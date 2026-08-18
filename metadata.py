#!/usr/bin/env python3
"""
metadata.py
===========
Layout-aware document metadata for the PDF → spreadsheet pipeline.

Direct extraction converts the whole document into an ordered content
stream — text rows and tables in reading order — and every output is
rendered from that stream, so the spreadsheet mirrors the source document
in ONE sheet with zero manual rearranging:

  * Table ROWS come from ruled horizontal lines.
  * Table COLUMNS come from interior ruled vertical lines when the table
    has them, otherwise from whitespace gutters — vertical strips that no
    text ever crosses (the layout of most real-world ordering documents,
    which rule the outline but separate columns with whitespace only).
  * Inside a ruled row, a text line that has content in 2+ columns is a
    new row (parallel values), while a line with content in a single
    column is a wrapped continuation and is joined to that cell.
  * A table whose leading rows repeat the previous table's leading rows
    is a cross-page continuation: the repeated header is dropped, leading
    single-cell wrap rows are joined into the previous table's last row,
    and the remaining rows are appended.
  * Text outside tables (headings, addresses, notes) becomes plain rows,
    split into cells on large horizontal gaps.

Digital PDFs read words from the text layer; scanned PDFs read them with
OCR (word bounding boxes). The geometry logic is shared.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union

import pdfplumber

from extractor import (
    _find_grid_positions, _grid_regions, _is_scanned_pdf, _line_masks,
    _ocr_availability, _render_page_gray,
)

logger = logging.getLogger(__name__)

MIN_OCR_CONF = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Model — an ordered stream of text rows and tables
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextRowMeta:
    kind: str           # always "text"
    page: int
    cells: List[str]


@dataclass
class TableMeta:
    kind: str           # always "table"
    index: int          # 1-based, source order (after continuation merges)
    page: int           # page the table starts on
    n_cols: int
    rows: List[List[str]]
    col_widths_pt: List[float]

    @property
    def n_rows(self) -> int:
        return len(self.rows)


@dataclass
class DocumentMeta:
    source_file: str
    page_count: int
    engine: str = ""
    content: List[Union[TextRowMeta, TableMeta]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def tables(self) -> List[TableMeta]:
        return [c for c in self.content if isinstance(c, TableMeta)]

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "page_count": self.page_count,
            "engine": self.engine,
            "warnings": list(self.warnings),
            "content": [asdict(c) for c in self.content],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Word sources — same shape for both paths: {text, x0, x1, top, bottom}
# ─────────────────────────────────────────────────────────────────────────────

def _words_digital(page):
    words = page.extract_words(x_tolerance=1.5, y_tolerance=2,
                               keep_blank_chars=False, use_text_flow=False)
    return [{"text": w["text"], "x0": float(w["x0"]), "x1": float(w["x1"]),
             "top": float(w["top"]), "bottom": float(w["bottom"])}
            for w in words if w["text"].strip()]


def _ocr_pass(gray, config):
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(gray, output_type=Output.DICT,
                                     config=config)
    words = []
    for i, raw in enumerate(data.get("text", [])):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < MIN_OCR_CONF:
            continue
        if text in ("|", "¦", "¡"):
            continue  # ruled-line artifacts (keep real "-" hyphens)
        left, top = float(data["left"][i]), float(data["top"][i])
        w = {"text": text, "x0": left,
             "x1": left + float(data["width"][i]),
             "top": top, "bottom": top + float(data["height"][i]),
             "conf": conf}
        # Ink check: real print is dark (a word's box is 15-35% dark
        # pixels); scan ghosts and bleed-through are faint. Filters
        # hallucinated words regardless of tesseract's confidence.
        crop = gray[int(top):int(w["bottom"]), int(left):int(w["x1"])]
        if crop.size == 0 or (crop < 150).mean() < 0.05:
            continue
        words.append(w)
    return words


def _words_ocr(gray):
    """Union of two tesseract segmentation passes.

    psm 6 (uniform block) has better recall on sparse table cells but can
    skip side regions entirely; psm 3 (auto) covers full-page layout but
    low-confidences isolated numerals. Overlapping detections are
    deduplicated keeping the higher-confidence word.
    """
    words = sorted(_ocr_pass(gray, "--psm 6") + _ocr_pass(gray, "--psm 3"),
                   key=lambda w: -w["conf"])
    kept, bins = [], {}
    for w in words:
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        key = (int(cx // 24), int(cy // 24))
        clash = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for o in bins.get((key[0] + dx, key[1] + dy), ()):
                    ix = min(w["x1"], o["x1"]) - max(w["x0"], o["x0"])
                    min_w = min(w["x1"] - w["x0"], o["x1"] - o["x0"])
                    h1 = w["bottom"] - w["top"]
                    h2 = o["bottom"] - o["top"]
                    dyc = abs((w["top"] + w["bottom"]) / 2
                              - (o["top"] + o["bottom"]) / 2)
                    # Same reading position → duplicate detection from the
                    # other pass (possibly garbled): x-ranges overlapping
                    # and vertical centres within one glyph height.
                    if ix > 0.5 * min_w and dyc < 0.8 * min(h1, h2):
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            kept.append(w)
            bins.setdefault(key, []).append(w)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (unit-agnostic: everything scales off median word height)
# ─────────────────────────────────────────────────────────────────────────────

def _is_datalike(text: str) -> bool:
    """Mostly-numeric cell content (quantities, prices, terms)."""
    digits = sum(ch.isdigit() for ch in text)
    if not digits:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    return alpha <= max(2, 0.3 * len(text))


def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2] if s else 0.0


def _line_height(words):
    return _median([w["bottom"] - w["top"] for w in words]) or 10.0


def _group_lines(words, tol):
    """Group words into visual text lines by vertical-centre proximity."""
    lines = []
    for w in sorted(words, key=lambda w: ((w["top"] + w["bottom"]) / 2, w["x0"])):
        cy = (w["top"] + w["bottom"]) / 2
        if lines and abs(cy - lines[-1][0]) <= tol:
            prev_cy, ws = lines[-1]
            ws.append(w)
            lines[-1] = ((prev_cy * (len(ws) - 1) + cy) / len(ws), ws)
        else:
            lines.append((cy, [w]))
    out = []
    for cy, ws in lines:
        ws.sort(key=lambda w: w["x0"])
        out.append(ws)
    return out


def _gutter_boundaries(words, x_left, x_right, min_gap):
    """
    Column boundaries from whitespace gutters: vertical strips inside
    [x_left, x_right] that no word ever crosses and that are at least
    min_gap wide. Returns the centre x of each gutter.
    """
    spans = sorted((max(w["x0"], x_left), min(w["x1"], x_right))
                   for w in words if w["x1"] > x_left and w["x0"] < x_right)
    if not spans:
        return []
    boundaries = []
    cur_end = spans[0][1]
    for s, e in spans[1:]:
        if s > cur_end:
            if s - cur_end >= min_gap:
                boundaries.append((cur_end + s) / 2)
            cur_end = e
        else:
            cur_end = max(cur_end, e)
    return boundaries


def _assign_cells(line_words, boundaries, n_cols):
    """Assign a visual line's words to columns by word-centre x."""
    import bisect
    cells = [""] * n_cols
    for w in line_words:
        cx = (w["x0"] + w["x1"]) / 2
        col = min(bisect.bisect(boundaries, cx), n_cols - 1)
        cells[col] = f"{cells[col]} {w['text']}".strip()
    return cells


def _split_line_on_gaps(line_words, gap):
    """Split one text line into cells wherever the horizontal gap is large."""
    cells, cur = [], [line_words[0]]
    for w in line_words[1:]:
        if w["x0"] - cur[-1]["x1"] > gap:
            cells.append(" ".join(x["text"] for x in cur))
            cur = [w]
        else:
            cur.append(w)
    cells.append(" ".join(x["text"] for x in cur))
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Table assembly
# ─────────────────────────────────────────────────────────────────────────────

def _build_table(region_words, row_edges, interior_vlines, x_left, x_right,
                 page_num, line_h, px_to_pt, cell_ocr=None):
    """
    Assemble one table from its words, ruled row edges, and column info.

    Rows come from the ruled bands. Within a band, a visual line filling
    2+ columns is a new row; a single-column line is a wrap continuation
    joined into that cell of the band's current row.
    """
    if interior_vlines:
        boundaries = sorted(interior_vlines)
    else:
        boundaries = _gutter_boundaries(
            region_words, x_left, x_right, min_gap=max(line_h * 0.75, 8.0))
    boundaries = [b for b in boundaries if x_left < b < x_right]
    n_cols = len(boundaries) + 1
    edges = [x_left] + list(boundaries) + [x_right]

    rows: List[List[str]] = []
    extents: List[tuple] = []   # per-row (y_lo, y_hi) for cell recovery
    bands: List[int] = []       # per-row source band index
    band_idx = -1

    def _new_row(cells, y_lo, y_hi):
        rows.append(cells)
        extents.append((y_lo, y_hi))
        bands.append(band_idx)
        return cells

    for y0, y1 in zip(row_edges[:-1], row_edges[1:]):
        band_idx += 1
        band = [w for w in region_words
                if y0 <= (w["top"] + w["bottom"]) / 2 < y1]
        if not band:
            if cell_ocr is not None and (y1 - y0) < line_h * 4:
                # Ruled band with no OCR words: the page-level OCR may have
                # missed it — leave a placeholder row for per-cell recovery.
                _new_row([""] * n_cols, y0 + 2, y1 - 2)
            continue
        lines = []
        for line_words in _group_lines(band, tol=line_h * 0.55):
            cy = _median([(w["top"] + w["bottom"]) / 2 for w in line_words])
            cells = _assign_cells(line_words, boundaries, n_cols)
            if any(cells):
                lines.append((cy, cells))
        if not lines:
            continue
        # Ruled lines are hard row breaks. Within a band, logical rows are
        # reconstructed from column occupancy:
        #   * a line whose filled columns are all empty in the current row
        #     completes it (item name above, values below → one row);
        #   * a line filling 3+ columns, or 2 columns with numeric data,
        #     is parallel data → a new row;
        #   * a line filling 1-2 text columns is a wrap continuation and
        #     is joined into the current row — unless it looks like an
        #     item name whose values follow on the next line, which
        #     starts a new row instead.
        infos = []
        for cy, cells in lines:
            infos.append((cells, {i for i, c in enumerate(cells) if c}))
        cur = None

        def _ext(cy):
            return (max(cy - line_h * 0.8, y0 + 2),
                    min(cy + line_h * 0.8, y1 - 2))

        def _widen(cy):
            lo, hi = _ext(cy)
            plo, phi = extents[-1]
            extents[-1] = (min(plo, lo), max(phi, hi))

        for idx, (cells, filled) in enumerate(infos):
            cy = lines[idx][0]
            if cur is None:
                cur = _new_row(cells, *_ext(cy))
                continue
            cur_filled = {i for i, c in enumerate(cur) if c}
            if not (filled & cur_filled):
                for i in filled:
                    cur[i] = cells[i]
                _widen(cy)
                continue
            if len(filled) >= 3 or (
                    len(filled) == 2
                    and any(_is_datalike(cells[i]) for i in filled)):
                cur = _new_row(cells, *_ext(cy))
                continue
            if len(filled) == 1:
                col = next(iter(filled))
                if cur_filled != {col}:
                    # Look past further wrap lines in the same column: if a
                    # multi-column line follows with this column empty, this
                    # line is an item name whose values come below → new row.
                    j = idx + 1
                    while j < len(infos) and infos[j][1] == {col}:
                        j += 1
                    nxt = infos[j] if j < len(infos) else None
                    if (nxt is not None and col not in nxt[1]
                            and len(nxt[1]) >= 2):
                        cur = _new_row(cells, *_ext(cy))
                        continue
            for i in filled:
                cur[i] = f"{cur[i]} {cells[i]}".strip()
            _widen(cy)
    if cell_ocr is not None:
        # Recovery pass: OCR empty cells individually. The callback is
        # cheap for truly blank cells (ink check short-circuits), so this
        # only pays for cells the page-level OCR genuinely missed.
        for r, (row, (y_lo, y_hi)) in enumerate(zip(rows, extents)):
            for c in range(n_cols):
                if not row[c]:
                    text = cell_ocr(y_lo, y_hi, edges[c] + 3, edges[c + 1] - 3)
                    if text:
                        row[c] = text
        keep = [i for i, row in enumerate(rows) if any(row)]
        rows[:] = [rows[i] for i in keep]
        bands[:] = [bands[i] for i in keep]

    if not rows:
        return None
    t = TableMeta(
        kind="table", index=0, page=page_num, n_cols=n_cols, rows=rows,
        col_widths_pt=[(edges[i + 1] - edges[i]) * px_to_pt
                       for i in range(n_cols)],
    )
    # Private rebuild context (not serialised): lets a cross-page
    # continuation be re-columnised with its parent table's boundaries.
    t._bands = bands
    t._ctx = {"words": region_words, "row_edges": row_edges,
              "x_left": x_left, "x_right": x_right, "line_h": line_h,
              "px_to_pt": px_to_pt, "cell_ocr": cell_ocr,
              "edges_pt": [e * px_to_pt for e in edges]}
    return t


def _norm_row(cells):
    return tuple("".join(ch for ch in c.lower() if ch.isalnum()) for c in cells)


def _norm_join(cells):
    return "".join(ch for ch in " ".join(cells).lower() if ch.isalnum())


def _header_similar(a_cells, b_cells) -> bool:
    import difflib
    a, b = _norm_join(a_cells), _norm_join(b_cells)
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def _rebuild_with_edges(t: TableMeta, edges_pt) -> Optional[TableMeta]:
    """Re-columnise a table using its parent table's column edges."""
    ctx = getattr(t, "_ctx", None)
    if not ctx:
        return None
    px = [e / ctx["px_to_pt"] for e in edges_pt]
    boundaries = px[1:-1]
    n_cols = len(boundaries) + 1
    rebuilt = _build_table(ctx["words"], ctx["row_edges"], boundaries or None,
                           ctx["x_left"], ctx["x_right"], t.page,
                           ctx["line_h"], ctx["px_to_pt"],
                           cell_ocr=ctx.get("cell_ocr"))
    if rebuilt is None or rebuilt.n_cols != n_cols:
        return None
    return rebuilt


def _try_merge_continuation(prev: TableMeta, nxt: TableMeta) -> bool:
    """If nxt repeats prev's leading rows (cross-page continuation), merge
    nxt into prev: drop the repeated header, join leading single-cell wrap
    rows into prev's last row, append the rest. Returns True if merged."""
    if nxt.page <= prev.page or not prev.rows or not nxt.rows:
        return False
    if prev.n_cols != nxt.n_cols:
        # Same table, drifted column detection (sparser page): re-columnise
        # the continuation with the parent's edges if headers agree.
        if not _header_similar(prev.rows[0], nxt.rows[0]):
            return False
        ctx = getattr(prev, "_ctx", None)
        rebuilt = _rebuild_with_edges(nxt, ctx["edges_pt"]) if ctx else None
        if rebuilt is None or rebuilt.n_cols != prev.n_cols:
            return False
        nxt = rebuilt
    matched = 0
    for a, b in zip(prev.rows, nxt.rows):
        if any(c for c in a) and (_norm_row(a) == _norm_row(b)
                                  or _header_similar(a, b)):
            matched += 1
        else:
            break
    if matched < 1:
        # No repeated header: still a continuation if the column geometry
        # is the same (borderless invoices repeat no header mid-table).
        pc = getattr(prev, "_ctx", {}).get("edges_pt")
        nc = getattr(nxt, "_ctx", {}).get("edges_pt")
        if (not pc or not nc or len(pc) != len(nc)
                or max(abs(a - b) for a, b in zip(pc, nc)) > 24):
            return False
    body = nxt.rows[matched:]
    nbands = getattr(nxt, "_bands", [0] * len(nxt.rows))[matched:]
    # Leading wrap rows: one filled cell, joined into prev's last row
    while body and prev.rows:
        filled = [i for i, c in enumerate(body[0]) if c]
        if len(filled) != 1:
            break
        col = filled[0]
        prev.rows[-1][col] = f"{prev.rows[-1][col]} {body[0][col]}".strip()
        body = body[1:]
        nbands = nbands[1:]
    pbands = getattr(prev, "_bands", [0] * len(prev.rows))
    # Seam stitch: an item split mid-row by the page break leaves prev's
    # last row holding only the name half; the continuation's first row
    # carries the rest of the name plus the values — join them.
    if body and prev.rows:
        pf = [i for i, c in enumerate(prev.rows[-1]) if c]
        bf = [i for i, c in enumerate(body[0]) if c]
        if len(pf) == 1 and pf[0] in bf and len(bf) >= 2:
            c = pf[0]
            body[0][c] = f"{prev.rows[-1][c]} {body[0][c]}".strip()
            prev.rows.pop()
            if pbands:
                pbands.pop()
    offset = (max(pbands) + 1) if pbands else 0
    prev._bands = pbands + [b + offset for b in nbands]
    prev.rows.extend(body)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Page stream builders
# ─────────────────────────────────────────────────────────────────────────────

def _page_stream(words, regions, page_num, page_w, px_to_pt, cell_ocr=None):
    """Build the ordered content stream for one page.

    regions: list of (y0, y1, x0, x1, row_edges, interior_vlines) in the
    same units as the words.
    """
    if not words and not regions:
        return []
    line_h = _line_height(words) if words else page_w * 0.012

    def in_region(w):
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        for i, (y0, y1, x0, x1, *_rest) in enumerate(regions):
            if y0 <= cy <= y1 and x0 <= cx <= x1:
                return i
        return -1

    outside, per_region = [], [[] for _ in regions]
    for w in words:
        i = in_region(w)
        (outside if i < 0 else per_region[i]).append(w)

    items = []  # (sort_y, item)
    for i, (y0, y1, x0, x1, row_edges, vlines) in enumerate(regions):
        t = _build_table(per_region[i], row_edges, vlines, x0, x1,
                         page_num, line_h, px_to_pt, cell_ocr=cell_ocr)
        if t is not None:
            items.append((y0, t))

    gap_cell = max(line_h * 2.2, page_w * 0.03)
    for line_words in _group_lines(outside, tol=line_h * 0.55):
        y = min(w["top"] for w in line_words)
        cells = _split_line_on_gaps(line_words, gap_cell)
        items.append((y, TextRowMeta(kind="text", page=page_num, cells=cells)))

    items.sort(key=lambda p: p[0])
    return [it for _, it in items]


def _ink_gutters(gray, horiz, vert, y0, y1, x0, x1, min_gap=None):
    """Column boundaries of a region from ink occupancy: vertical strips
    that contain no text ink (ruled lines removed) are column gutters."""
    import cv2
    import numpy as np

    crop = gray[y0:y1, x0:x1]
    thr = cv2.adaptiveThreshold(
        cv2.bitwise_not(crop), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, -2)
    kill = cv2.dilate(
        cv2.add(horiz[y0:y1, x0:x1], vert[y0:y1, x0:x1]),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    ink = cv2.bitwise_and(thr, cv2.bitwise_not(kill))
    profile = (ink > 0).sum(axis=0)
    if min_gap is None:
        min_gap = max(int((x1 - x0) * 0.015), 18)
    inside = np.where(profile > 1)[0]
    if len(inside) == 0:
        return []
    lo, hi = int(inside[0]), int(inside[-1])
    boundaries, run = [], None
    for i in range(lo, hi + 1):
        if profile[i] <= 1:
            run = i if run is None else run
        else:
            if run is not None and i - run >= min_gap:
                boundaries.append(x0 + (run + i) / 2)
            run = None
    return boundaries


def _open_regions(horiz, closed_boxes, line_h, page_w, page_h, words=()):
    """Table regions for borderless tables: long horizontal separator
    rules (with no surrounding grid) grouped into one region; rows are the
    separator bands, columns come from whitespace gutters later."""
    import cv2
    import numpy as np

    n, _labels, stats, _cents = cv2.connectedComponentsWithStats(
        (horiz > 0).astype(np.uint8), connectivity=8)
    seps = []
    for i in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[i])
        if w < page_w * 0.25 or h > max(line_h, 8):
            continue
        cy, cx = y + h // 2, x + w // 2
        if any(ry0 - 6 <= cy <= ry1 + 6 and rx0 - 6 <= cx <= rx1 + 6
               for (ry0, ry1, rx0, rx1) in closed_boxes):
            continue
        seps.append((cy, x, x + w))
    seps.sort()

    groups = []
    for s in seps:
        if groups:
            last = groups[-1][-1]
            overlap = min(last[2], s[2]) - max(last[1], s[1])
            width = min(last[2] - last[1], s[2] - s[1])
            if s[0] - last[0] < line_h * 12 and overlap > 0.6 * width:
                groups[-1].append(s)
                continue
        groups.append([s])

    regions = []
    for g in groups:
        if len(g) < 2:
            continue
        ys = [s[0] for s in g]
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        pad = int(_median(gaps)) if gaps else int(line_h * 3)
        x0, x1 = min(s[1] for s in g), max(s[2] for s in g)
        # Adaptive top: the tail of an item cut at the previous page break
        # sits above the first separator, so climb upward line by line
        # while the vertical gaps stay tight — and stop at the first big
        # gap, which separates the table from page furniture whose words
        # would bridge the column gutters.
        top = max(0, ys[0] - min(pad, int(line_h * 2)))
        line_ys = sorted({round((w["top"] + w["bottom"]) / 2)
                          for w in words
                          if w["top"] < ys[0] and x0 <= (w["x0"] + w["x1"]) / 2 <= x1},
                         reverse=True)
        prev_y = ys[0]
        for ly in line_ys:
            # threshold is line PITCH (leading), not glyph height: text
            # lines sit ~1.8-2x the glyph height apart; the gap to page
            # furniture above the table is 3.5x+
            if prev_y - ly > line_h * 2.7:
                break
            prev_y = ly
            top = max(0, int(ly - line_h * 0.9))
        bottom = min(page_h - 1, ys[-1] + pad)
        regions.append((top, bottom, x0, x1, [top] + ys + [bottom], []))
    return regions


def _ocr_cell_strict(cell, min_conf=45.0):
    """Per-cell recovery OCR with a confidence gate.

    Recovery crops are mostly blank paper; plain OCR hallucinates on scan
    noise (Otsu turns faint specks into glyph shapes). Keep only words
    tesseract itself is confident about, and reject the all-lowercase
    letter-salad pattern the noise produces — real short cells are
    numeric or capitalised.
    """
    import cv2
    import pytesseract
    from pytesseract import Output

    if cell is None or cell.size == 0 or (cell < 128).sum() < 4:
        return ""
    if cell.shape[0] < 100:
        cell = cv2.resize(cell, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)
    cell = cv2.copyMakeBorder(cell, 12, 12, 12, 12,
                              cv2.BORDER_CONSTANT, value=255)
    _, cell = cv2.threshold(cell, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(cell, config="--psm 6",
                                     output_type=Output.DICT)
    parts = []
    for i, raw in enumerate(data.get("text", [])):
        text = (raw or "").strip().strip("|¦")
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= min_conf:
            parts.append(text)
    out = " ".join(parts).strip()
    compact = out.replace(" ", "")
    if compact and compact.isalpha():
        tokens = out.split()
        # letter-salad ghosts: all-lowercase, or stray single letters, or
        # nothing but 1-2 letter fragments
        if compact.islower() and len(compact) <= 8:
            return ""
        if any(len(t) == 1 for t in tokens):
            return ""
        if all(len(t) <= 2 for t in tokens):
            return ""
    return out


def _region_vlines(gray, y, x, w, h, ys):
    """Interior vertical ruled lines of one region, using a kernel sized
    to the region's own row height — the page-level mask uses a long
    kernel that erases dividers broken into per-row segments by the
    crossing horizontal lines."""
    import cv2

    crop = gray[y:y + h, x:x + w]
    thr = cv2.adaptiveThreshold(
        cv2.bitwise_not(crop), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, -2)
    band_h = min((b - a for a, b in zip(ys, ys[1:])), default=h)
    k = max(int(band_h * 0.75), 25)
    fine = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    return _find_grid_positions(fine, axis=1, min_frac=0.55)


def _scanned_page_stream(page, page_num):
    import cv2  # noqa: F401

    gray, res = _render_page_gray(page)
    px_to_pt = 72.0 / res
    horiz, vert = _line_masks(gray)

    # OCR a copy with the ruled lines erased: tesseract otherwise reads
    # the line pixels themselves as junk words, which both pollutes cells
    # and bridges the whitespace gutters used for column detection.
    import cv2
    import numpy as np
    kill = cv2.dilate(cv2.add(horiz, vert),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    gray_ocr = gray.copy()
    gray_ocr[kill > 0] = 255
    words = _words_ocr(gray_ocr)

    regions = []
    for y, x, w, h in _grid_regions(horiz, vert):
        ys = _find_grid_positions(horiz[y:y + h, x:x + w], axis=0)
        if len(ys) < 1:
            continue
        xs = list(_find_grid_positions(vert[y:y + h, x:x + w], axis=1))
        for v in _region_vlines(gray, y, x, w, h, ys):
            if all(abs(v - u) > 8 for u in xs):
                xs.append(v)
        xs.sort()
        interior = [x + v for v in xs if 6 < v < w - 6]
        edges = [y + e for e in ys]
        # A table that continues onto the next page has no bottom border
        # (and a continued one no top border): close the region explicitly.
        if edges[0] - y > 12:
            edges.insert(0, y)
        if (y + h) - edges[-1] > 12:
            edges.append(y + h)
        if len(edges) < 2:
            continue
        regions.append((y, y + h, x, x + w, edges, interior))

    lh = _line_height(words) if words else gray.shape[1] * 0.012
    closed_boxes = [(r[0], r[1], r[2], r[3]) for r in regions]
    regions.extend(_open_regions(horiz, closed_boxes, lh,
                                 gray.shape[1], gray.shape[0], words))

    # Full-page OCR sometimes returns nothing for an isolated region
    # (tesseract segmentation quirk). For such regions, derive column
    # boundaries from the ink itself — vertical strips free of text ink —
    # and let per-cell recovery OCR the cells afterwards.
    for ri, (ry0, ry1, rx0, rx1, _e, vlines) in enumerate(regions):
        has_words = any(
            ry0 <= (w["top"] + w["bottom"]) / 2 <= ry1
            and rx0 <= (w["x0"] + w["x1"]) / 2 <= rx1 for w in words)
        if not has_words and not vlines:
            for b in _ink_gutters(gray, horiz, vert, ry0, ry1, rx0, rx1):
                vlines.append(b)

    def cell_ocr(y0, y1, x0, x1):
        y0, x0 = max(int(y0), 0), max(int(x0), 0)
        y1, x1 = min(int(y1), gray.shape[0]), min(int(x1), gray.shape[1])
        if y1 - y0 < 4 or x1 - x0 < 4:
            return ""
        return _ocr_cell_strict(gray[y0:y1, x0:x1])

    return _page_stream(words, regions, page_num, gray.shape[1], px_to_pt,
                        cell_ocr=cell_ocr)


def _digital_page_stream(page, page_num):
    words = _words_digital(page)
    regions = []
    found = page.find_tables()
    found.sort(key=lambda t: (round(t.bbox[1]), round(t.bbox[0])))
    for tbl in found:
        rects = tbl.cells
        if not rects:
            continue
        xs = sorted({round(v, 1) for r in rects for v in (r[0], r[2])})
        ys = sorted({round(v, 1) for r in rects for v in (r[1], r[3])})
        if len(ys) < 2 or len(xs) < 2:
            continue
        x0, x1 = xs[0], xs[-1]
        interior = [v for v in xs if x0 + 2 < v < x1 - 2]
        regions.append((ys[0], ys[-1], x0, x1, ys, interior))
    return _page_stream(words, regions, page_num, float(page.width), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _merge_prefix_rows(t: TableMeta) -> None:
    """Within one ruled band, a row holding only one text cell directly
    above a full row that also fills that column is a prefix (an item code
    printed above its own line): join it in. Runs after cross-page merges
    so genuine page-tail wraps are consumed by those first."""
    bands = getattr(t, "_bands", None)
    if not bands or len(bands) != len(t.rows):
        return
    i = 0
    while i < len(t.rows) - 1:
        row, nxt = t.rows[i], t.rows[i + 1]
        filled = [c for c, v in enumerate(row) if v]
        if len(filled) == 1 and bands[i] == bands[i + 1]:
            c = filled[0]
            if nxt[c] and sum(1 for v in nxt if v) >= 2:
                nxt[c] = f"{row[c]} {nxt[c]}".strip()
                del t.rows[i]
                del bands[i]
                continue
        i += 1


def build_document_metadata(pdf_path: str) -> DocumentMeta:
    doc = DocumentMeta(source_file=pdf_path, page_count=0)
    scanned = _is_scanned_pdf(pdf_path)

    if scanned:
        available, missing = _ocr_availability()
        if not available:
            doc.warnings.append(
                "This PDF appears to be scanned / image-only (no text "
                "layer), so it can only be read with OCR — but OCR is not "
                "available. Missing: " + "; ".join(missing) + ".")
            return doc

    raw: List[Union[TextRowMeta, TableMeta]] = []
    with pdfplumber.open(pdf_path) as pdf:
        doc.page_count = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                if scanned:
                    stream = _scanned_page_stream(page, page_num)
                else:
                    stream = _digital_page_stream(page, page_num)
            except Exception as exc:
                logger.warning("page %d failed: %s", page_num, exc)
                doc.warnings.append(f"Page {page_num} could not be read: {exc}")
                continue
            raw.extend(stream)

    # Drop repeating page furniture (running headers/footers): the same
    # normalized text appearing on 3+ pages is chrome, not content.
    if doc.page_count >= 3:
        seen: dict = {}
        for item in raw:
            if isinstance(item, TextRowMeta):
                seen.setdefault(_norm_join(item.cells), set()).add(item.page)
        furniture = {k for k, pages in seen.items()
                     if len(pages) >= max(3, doc.page_count // 2) and k}
        import re as _re
        _page_no = _re.compile(r"^page\d+of\d+$")
        raw = [it for it in raw
               if not (isinstance(it, TextRowMeta)
                       and (_norm_join(it.cells) in furniture
                            or _page_no.match(_norm_join(it.cells))))]
        # Page furniture that a region pad swallowed into a table (running
        # address lines, "Page N of M") shows up as repeated table rows —
        # strip those too.
        if furniture:
            import re as _re
            page_pat = _re.compile(r"^page\d+of\d+$")
            for it in raw:
                if not isinstance(it, TableMeta):
                    continue
                bands = getattr(it, "_bands", None)
                keep = [i for i, row in enumerate(it.rows)
                        if _norm_join(row) not in furniture
                        and not page_pat.match(_norm_join(row))]
                if len(keep) != len(it.rows):
                    it.rows[:] = [it.rows[i] for i in keep]
                    if bands and len(bands) >= len(keep):
                        it._bands = [bands[i] for i in keep]

    content: List[Union[TextRowMeta, TableMeta]] = []
    for item in raw:
        if isinstance(item, TableMeta) and content:
            # A table may continue past intervening non-table rows only if
            # the previous item is the table itself (furniture removed).
            if (isinstance(content[-1], TableMeta)
                    and _try_merge_continuation(content[-1], item)):
                continue
        content.append(item)

    doc.content = content
    tables = doc.tables
    for t in tables:
        _merge_prefix_rows(t)
        t.__dict__.pop("_ctx", None)
        t.__dict__.pop("_bands", None)
    for i, t in enumerate(tables, start=1):
        t.index = i

    if tables:
        doc.engine = "ocr-grid" if scanned else "pdfplumber-grid"
        if scanned:
            doc.warnings.append(
                "This PDF is scanned, so text was read with OCR. Table "
                "structure comes from the ruled lines and whitespace "
                "columns and mirrors the source, but OCR character "
                "accuracy is not guaranteed — spot-check values against "
                "the PDF.")
    elif not doc.warnings:
        doc.warnings.append("No bordered/grid tables detected.")
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Renderers — ONE sheet, content in reading order
# ─────────────────────────────────────────────────────────────────────────────

def _iter_sheet_rows(doc: DocumentMeta):
    """Yield (cells, is_table_row) sheet rows with blank separators."""
    prev = None
    for item in doc.content:
        if prev is not None and (isinstance(item, TableMeta)
                                 or isinstance(prev, TableMeta)):
            yield [], False
        if isinstance(item, TableMeta):
            for row in item.rows:
                yield row, True
        else:
            yield item.cells, False
        prev = item


def metadata_to_xlsx_bytes(doc: DocumentMeta) -> bytes:
    """Render the whole document into a single sheet in reading order.
    Table rows get thin borders; every value is written as text so
    spreadsheet apps cannot coerce values on open ("007" stays "007")."""
    import io
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    thin = Side(style="thin", color="999999")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    wb = Workbook()
    ws = wb.active
    ws.title = "Extracted"

    r = 0
    max_cols = 1
    for cells, is_table in _iter_sheet_rows(doc):
        r += 1
        max_cols = max(max_cols, len(cells))
        for c, val in enumerate(cells, start=1):
            text = ILLEGAL_CHARACTERS_RE.sub("", val or "")
            xc = ws.cell(row=r, column=c, value=text)
            xc.alignment = Alignment(wrap_text=False, vertical="top")
            if is_table:
                xc.border = border
        if is_table and not cells:
            pass

    # Column widths: sized to content, capped so the sheet stays readable
    widths = {}
    for row in ws.iter_rows():
        for c in row:
            if c.value:
                widths[c.column] = max(widths.get(c.column, 0), len(str(c.value)))
    for col, wlen in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = min(60, max(9, wlen + 2))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def metadata_preview_text(doc: DocumentMeta) -> str:
    """CSV-style text of the single-sheet layout for the UI preview pane."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for cells, _ in _iter_sheet_rows(doc):
        writer.writerow(cells)
    return buf.getvalue().rstrip("\n")
