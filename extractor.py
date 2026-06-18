#!/usr/bin/env python3
"""
extractor.py
============
Provider-agnostic PDF → table extraction engine.

Strategy ladder (tried in order, best result wins):
  1. pdfplumber tables       – text-layer grid/border table extraction
  2. pdfplumber text-parse   – extract_text() → smart column splitting
  3. camelot lattice         – bordered tables (requires ghostscript)
  4. camelot stream          – borderless tables (requires ghostscript)
  5. tabula-py               – Java-based fallback (requires Java)
  6. pdfminer text heuristic – raw text parsing with delimiter detection
  7. OCR fallback            – for scanned/image PDFs (requires tesseract)

Engines 1, 2, and 6 have ZERO system dependencies beyond Python.
The app will always work even without Ghostscript, Java, or Tesseract.
"""

from __future__ import annotations

import io
import re
import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedTable:
    source: str
    page: int
    df: pd.DataFrame
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    tables: List[ExtractedTable] = field(default_factory=list)
    strategy_used: str = ""
    warnings: List[str] = field(default_factory=list)
    engine_log: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Quality scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_df(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    rows, cols = df.shape
    if rows < 1 or cols < 1:
        return 0.0
    non_empty = df.notna().sum().sum()
    total = rows * cols
    fill_ratio = non_empty / total if total else 0
    row_score = min(rows / 20, 1.0)
    col_score = min(cols / 3, 1.0)
    return round(fill_ratio * 0.5 + row_score * 0.3 + col_score * 0.2, 3)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _stringify = lambda v: str(v).strip() if pd.notna(v) else ""
    try:
        df = df.map(_stringify)
    except AttributeError:
        df = df.applymap(_stringify)

    # Replace 'nan', 'None' strings
    df = df.replace({"nan": "", "None": "", "none": ""})

    # Drop columns that are entirely empty
    df = df.loc[:, (df != "").any(axis=0)]
    # Drop rows that are entirely empty
    df = df.loc[(df != "").any(axis=1)]
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 – pdfplumber extract_tables() (grid/border detection)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pdfplumber_tables(pdf_path: str) -> List[ExtractedTable]:
    results: List[ExtractedTable] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    # Use first row as header if it looks like one
                    header = tbl[0]
                    if header and all(h is not None for h in header):
                        df = pd.DataFrame(tbl[1:], columns=header)
                    else:
                        df = pd.DataFrame(tbl)
                    df = _clean_df(df)
                    score = _score_df(df)
                    if score > 0:
                        results.append(ExtractedTable(
                            source="pdfplumber-tables", page=page_num,
                            df=df, confidence=score
                        ))
    except Exception as exc:
        logger.warning("pdfplumber-tables failed: %s", exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 – pdfplumber extract_text() + smart column splitting
# This is the KEY strategy that works on virtually all text-layer PDFs
# even without borders/grids, and has ZERO extra dependencies.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pdfplumber_text(pdf_path: str) -> List[ExtractedTable]:
    """
    Use pdfplumber's word-level extraction to reconstruct table columns
    by detecting consistent vertical alignment of text elements.
    """
    results: List[ExtractedTable] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Try word-based column detection first
                table = _words_to_table(page)
                if table is None:
                    # Fallback: split extract_text() by whitespace gaps
                    table = _text_to_table(page)
                if table is not None:
                    df = _clean_df(table)
                    score = _score_df(df)
                    if score > 0:
                        results.append(ExtractedTable(
                            source="pdfplumber-text", page=page_num,
                            df=df, confidence=score
                        ))
    except Exception as exc:
        logger.warning("pdfplumber-text failed: %s", exc)
    return results


def _words_to_table(page) -> Optional[pd.DataFrame]:
    """
    Extract words with their x-coordinates, cluster into columns,
    then group by y-coordinate into rows.
    """
    try:
        words = page.extract_words(
            x_tolerance=3, y_tolerance=3,
            keep_blank_chars=True, use_text_flow=False
        )
        if not words or len(words) < 4:
            return None

        # Group words into rows by y-coordinate (top)
        row_groups = _group_by_y(words, tolerance=5)
        if len(row_groups) < 2:
            return None

        # Detect column boundaries from x-positions across all rows
        col_boundaries = _detect_column_boundaries(words, row_groups)
        if not col_boundaries or len(col_boundaries) < 2:
            return None

        # Build table
        rows = []
        for _, row_words in sorted(row_groups.items()):
            row = _assign_words_to_columns(row_words, col_boundaries)
            rows.append(row)

        if len(rows) < 2:
            return None

        # Filter: keep only rows with the most common column count
        col_counts = [len(r) for r in rows]
        if not col_counts:
            return None
        mode_cols = Counter(col_counts).most_common(1)[0][0]
        if mode_cols < 2:
            return None
        rows = [r for r in rows if len(r) == mode_cols]

        if len(rows) < 2:
            return None

        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df

    except Exception:
        return None


def _group_by_y(words: list, tolerance: float = 5) -> dict:
    """Group words into row clusters based on their y-coordinate."""
    groups = {}
    for w in words:
        y = round(w["top"])
        # Find existing group within tolerance
        matched = False
        for gy in list(groups.keys()):
            if abs(y - gy) <= tolerance:
                groups[gy].append(w)
                matched = True
                break
        if not matched:
            groups[y] = [w]
    return groups


def _detect_column_boundaries(words: list, row_groups: dict) -> List[float]:
    """
    Detect column left-edge boundaries by finding clusters of word x0 positions.
    """
    all_x0 = sorted([w["x0"] for w in words])
    if not all_x0:
        return []

    # Cluster x0 positions: gap > page_width * 0.03 means new column
    page_width = max(w.get("x1", w["x0"] + 50) for w in words)
    min_gap = max(page_width * 0.03, 8)

    boundaries = [all_x0[0]]
    x0_sorted = sorted(set(round(x, 1) for x in all_x0))

    for i in range(1, len(x0_sorted)):
        if x0_sorted[i] - x0_sorted[i - 1] > min_gap:
            boundaries.append(x0_sorted[i])

    return boundaries


def _assign_words_to_columns(row_words: list, boundaries: List[float]) -> List[str]:
    """Assign each word in a row to its nearest column boundary."""
    n_cols = len(boundaries)
    cells = [""] * n_cols

    for w in sorted(row_words, key=lambda x: x["x0"]):
        # Find closest column
        best_col = 0
        best_dist = abs(w["x0"] - boundaries[0])
        for i, b in enumerate(boundaries):
            dist = abs(w["x0"] - b)
            if dist < best_dist:
                best_dist = dist
                best_col = i
        text = w.get("text", "").strip()
        if cells[best_col]:
            cells[best_col] += " " + text
        else:
            cells[best_col] = text

    return cells


def _text_to_table(page) -> Optional[pd.DataFrame]:
    """
    Fallback: use extract_text() and split lines by detecting
    consistent multi-space gaps.
    """
    text = page.extract_text()
    if not text or len(text.strip()) < 10:
        return None

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return None

    # Detect if lines have consistent multi-space separation
    delim = _detect_delimiter(lines)
    if not delim:
        return None

    rows = []
    for ln in lines:
        parts = [p.strip() for p in re.split(delim, ln)]
        parts = [p for p in parts if p != ""]
        if parts:
            rows.append(parts)

    if len(rows) < 2:
        return None

    # Normalise column count
    col_counts = [len(r) for r in rows]
    mode_cols = Counter(col_counts).most_common(1)[0][0]
    if mode_cols < 2:
        return None

    # Keep rows matching mode, allow ±1 (pad/trim)
    normalised = []
    for r in rows:
        if len(r) == mode_cols:
            normalised.append(r)
        elif len(r) == mode_cols - 1:
            normalised.append(r + [""])
        elif len(r) == mode_cols + 1:
            # Try merging last two cells
            normalised.append(r[:mode_cols - 1] + [" ".join(r[mode_cols - 1:])])
        # Skip rows with very different column count

    if len(normalised) < 2:
        return None

    df = pd.DataFrame(normalised[1:], columns=normalised[0])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3/4 – camelot (lattice + stream) — needs ghostscript
# ─────────────────────────────────────────────────────────────────────────────

def _extract_camelot(pdf_path: str, flavor: str) -> List[ExtractedTable]:
    results: List[ExtractedTable] = []
    try:
        import camelot
        tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
        for t in tables:
            df = _clean_df(t.df)
            score = _score_df(df) * (t.accuracy / 100.0)
            if score > 0:
                results.append(ExtractedTable(
                    source=f"camelot-{flavor}", page=t.page,
                    df=df, confidence=score
                ))
    except ImportError:
        logger.info("camelot not installed, skipping")
    except Exception as exc:
        logger.warning("camelot-%s failed: %s", flavor, exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5 – tabula-py — needs Java
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tabula(pdf_path: str) -> List[ExtractedTable]:
    results: List[ExtractedTable] = []
    # Quick check: skip if Java not available
    if not shutil.which("java"):
        logger.info("Java not found, skipping tabula")
        return results
    try:
        import tabula
        dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True,
                              silent=True, pandas_options={"dtype": str})
        for i, df in enumerate(dfs):
            df = _clean_df(df)
            score = _score_df(df)
            if score > 0:
                results.append(ExtractedTable(
                    source="tabula", page=i + 1, df=df, confidence=score
                ))
    except ImportError:
        logger.info("tabula not installed, skipping")
    except Exception as exc:
        logger.warning("tabula failed: %s", exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 6 – pdfminer text heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_heuristic(pdf_path: str) -> List[ExtractedTable]:
    results: List[ExtractedTable] = []
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        if not text or len(text.strip()) < 10:
            return results

        lines = [ln for ln in text.split("\n") if ln.strip()]
        if len(lines) < 2:
            return results

        delim = _detect_delimiter(lines)
        if not delim:
            return results

        rows = []
        for ln in lines:
            parts = [p.strip() for p in re.split(delim, ln)]
            parts = [p for p in parts if p != ""]
            if parts:
                rows.append(parts)

        if len(rows) < 2:
            return results

        col_counts = [len(r) for r in rows]
        mode_cols = Counter(col_counts).most_common(1)[0][0]
        if mode_cols < 2:
            return results

        rows = [r for r in rows if len(r) == mode_cols]
        if len(rows) < 2:
            return results

        df = pd.DataFrame(rows[1:], columns=rows[0])
        df = _clean_df(df)
        score = _score_df(df) * 0.6
        if score > 0:
            results.append(ExtractedTable(
                source="text-heuristic", page=1, df=df, confidence=score
            ))

    except ImportError:
        logger.info("pdfminer not installed, skipping")
    except Exception as exc:
        logger.warning("text heuristic failed: %s", exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 7 – OCR fallback (for scanned / image-only PDFs)
# ─────────────────────────────────────────────────────────────────────────────

def _is_scanned_pdf(pdf_path: str) -> bool:
    """Detect if a PDF is image-based (no extractable text)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_chars = 0
            for page in pdf.pages[:3]:  # check first 3 pages
                text = page.extract_text() or ""
                total_chars += len(text.strip())
            return total_chars < 50  # almost no text = scanned
    except Exception:
        return False


def _ocr_availability() -> tuple[bool, List[str]]:
    """Return (available, missing_components) for the OCR pipeline."""
    missing: List[str] = []
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        missing.append("pytesseract (pip install pytesseract)")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow (pip install pillow)")
    if not shutil.which("tesseract"):
        missing.append(
            "tesseract binary (macOS: brew install tesseract, "
            "Debian/Ubuntu: apt-get install tesseract-ocr)"
        )
    return (len(missing) == 0, missing)


def _render_page_image(page):
    """Render a pdfplumber page to a PIL image, or None on failure."""
    try:
        return page.to_image(resolution=300).original
    except Exception as exc:
        logger.warning("page render failed: %s", exc)
        return None


def _ocr_detect_columns(words: list, row_groups: dict) -> List[float]:
    """
    Detect column left-edge boundaries from whitespace gutters.

    For each row we walk words left-to-right and start a new column whenever
    the blank space between the previous word's right edge (x1) and the next
    word's left edge (x0) exceeds a gutter threshold. Column-start positions
    are then clustered across all rows so the columns line up vertically.
    This keeps multi-word cells intact while still splitting on real gutters.
    """
    if not words:
        return []
    page_width = max(w.get("x1", w["x0"] + 50) for w in words)
    gutter = max(page_width * 0.015, 15)   # min blank space for a column break
    tol = max(page_width * 0.02, 20)        # clustering tolerance for col starts

    starts: List[float] = []
    for _, row in row_groups.items():
        rw = sorted(row, key=lambda x: x["x0"])
        if not rw:
            continue
        starts.append(rw[0]["x0"])
        for i in range(1, len(rw)):
            if rw[i]["x0"] - rw[i - 1].get("x1", rw[i - 1]["x0"]) > gutter:
                starts.append(rw[i]["x0"])

    if not starts:
        return []
    starts.sort()
    boundaries = [starts[0]]
    for s in starts[1:]:
        if s - boundaries[-1] > tol:
            boundaries.append(s)
    return boundaries


def _ocr_data_to_table(data: dict, min_conf: float = 30.0) -> Optional[pd.DataFrame]:
    """
    Reconstruct a table from pytesseract image_to_data() output.

    Instead of splitting raw OCR text by delimiters (fragile on multi-column
    layouts), we use each recognised word's bounding box: group words into
    rows by their vertical position, then assign each word to a column by its
    left x-edge — the same clustering approach used for digital text layers.
    """
    texts = data.get("text", [])
    n = len(texts)
    if n == 0:
        return None

    words = []
    for i in range(n):
        text = (texts[i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError, KeyError):
            conf = -1.0
        if conf < min_conf:
            continue
        left = float(data["left"][i])
        top = float(data["top"][i])
        width = float(data["width"][i])
        words.append({
            "text": text,
            "x0": left,
            "x1": left + width,
            "top": top,
        })

    if len(words) < 4:
        return None

    # Group words into rows; OCR boxes need a slightly looser y-tolerance
    row_groups = _group_by_y(words, tolerance=10)
    if len(row_groups) < 2:
        return None

    # Detect columns from genuine whitespace gutters. We do NOT reuse the
    # digital-text detector here: it clusters word *left-edges*, so a wide
    # word followed by a normal space yields a large x0-to-x0 gap and splits
    # a multi-word cell (e.g. a long product name) into spurious columns.
    # Instead we look at the whitespace between a word's right edge and the
    # next word's left edge — a real column gutter, not just a wide word.
    col_boundaries = _ocr_detect_columns(words, row_groups)
    if not col_boundaries or len(col_boundaries) < 2:
        return None

    rows = []
    for _, row_words in sorted(row_groups.items()):
        rows.append(_assign_words_to_columns(row_words, col_boundaries))

    if len(rows) < 2:
        return None

    col_counts = [len(r) for r in rows]
    mode_cols = Counter(col_counts).most_common(1)[0][0]
    if mode_cols < 2:
        return None
    rows = [r for r in rows if len(r) == mode_cols]
    if len(rows) < 2:
        return None

    return pd.DataFrame(rows[1:], columns=rows[0])


def _extract_ocr(pdf_path: str) -> List[ExtractedTable]:
    """OCR fallback for scanned / image-only PDFs (no text layer)."""
    results: List[ExtractedTable] = []
    if not _is_scanned_pdf(pdf_path):
        return results

    available, missing = _ocr_availability()
    if not available:
        # The user-facing reason is surfaced by extract_tables(); just stop.
        logger.info("OCR unavailable, missing: %s", ", ".join(missing))
        return results

    import pytesseract
    from pytesseract import Output

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                pil_img = _render_page_image(page)
                if pil_img is None:
                    continue

                data = pytesseract.image_to_data(
                    pil_img, output_type=Output.DICT
                )
                df = _ocr_data_to_table(data)
                if df is None:
                    continue

                df = _clean_df(df)
                score = _score_df(df) * 0.6  # OCR confidence discount
                if score > 0:
                    results.append(ExtractedTable(
                        source="ocr", page=page_num, df=df, confidence=score
                    ))

    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_delimiter(lines: list) -> Optional[str]:
    """Return the most likely delimiter regex, or None."""
    candidates = {
        r"\t":       sum(ln.count("\t") for ln in lines),
        r"\|":       sum(ln.count("|") for ln in lines),
        r",":        sum(ln.count(",") for ln in lines),
        r"  {2,}":   sum(1 for ln in lines if re.search(r"  {2,}", ln)),
    }
    best = max(candidates, key=candidates.get)
    if candidates[best] >= max(2, len(lines) * 0.3):
        return best
    # Be more lenient with multi-space: if majority of lines have 2+ spaces
    if candidates[r"  {2,}"] >= len(lines) * 0.5:
        return r"  {2,}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _fingerprint(df: pd.DataFrame) -> str:
    sub = df.iloc[:3, :5]
    vals = [str(v) for v in sub.values.flatten().tolist()]
    return "|".join(vals)


def _deduplicate(tables: List[ExtractedTable]) -> List[ExtractedTable]:
    seen: set[str] = set()
    deduped: List[ExtractedTable] = []
    for t in tables:
        fp = _fingerprint(t.df)
        if fp not in seen:
            seen.add(fp)
            deduped.append(t)
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables(pdf_path: str) -> ExtractionResult:
    """
    Run the full strategy ladder and return the best set of tables.
    """
    result = ExtractionResult()
    all_candidates: List[ExtractedTable] = []

    strategies = [
        ("pdfplumber-tables", lambda: _extract_pdfplumber_tables(pdf_path)),
        ("pdfplumber-text",   lambda: _extract_pdfplumber_text(pdf_path)),
        ("camelot-lattice",   lambda: _extract_camelot(pdf_path, "lattice")),
        ("camelot-stream",    lambda: _extract_camelot(pdf_path, "stream")),
        ("tabula",            lambda: _extract_tabula(pdf_path)),
        ("text-heuristic",    lambda: _extract_text_heuristic(pdf_path)),
        ("ocr",               lambda: _extract_ocr(pdf_path)),
    ]

    for name, fn in strategies:
        try:
            tables = fn()
        except Exception as exc:
            msg = f"{name}: FAILED ({exc})"
            result.engine_log.append(msg)
            result.warnings.append(msg)
            continue

        if tables:
            all_candidates.extend(tables)
            msg = f"{name}: found {len(tables)} table(s)"
            result.engine_log.append(msg)
            logger.info(msg)
        else:
            result.engine_log.append(f"{name}: no tables found")

    if not all_candidates:
        if _is_scanned_pdf(pdf_path):
            available, missing = _ocr_availability()
            if not available:
                result.warnings.append(
                    "This PDF appears to be scanned / image-only (it has no "
                    "text layer), so it can only be read with OCR — but OCR is "
                    "not available in this environment. Missing: "
                    + "; ".join(missing) + ". Install the missing component(s) "
                    "and try again."
                )
            else:
                result.warnings.append(
                    "This PDF appears to be scanned / image-only. OCR ran but "
                    "could not reconstruct a table from the page image — the "
                    "scan may be low-resolution, skewed/rotated, or not a "
                    "table. Engine details: " + "; ".join(result.engine_log)
                )
        else:
            result.warnings.append(
                "No tables found by any engine. "
                "Engine details: " + "; ".join(result.engine_log)
            )
        return result

    # Deduplicate across strategies
    unique = _deduplicate(all_candidates)

    # For each page, keep the highest-confidence extraction
    by_page: dict[int, ExtractedTable] = {}
    for t in unique:
        if t.page not in by_page or t.confidence > by_page[t.page].confidence:
            by_page[t.page] = t

    result.tables = sorted(by_page.values(), key=lambda t: t.page)
    result.strategy_used = ", ".join(sorted({t.source for t in result.tables}))
    return result
