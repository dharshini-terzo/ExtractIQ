#!/usr/bin/env python3
"""
processor.py
============
Orchestrates the full PDF → CSV pipeline.

No dependency on any specific LLM provider. Extraction is entirely local;
LLM assistance is an optional enhancement for reference-sheet mapping and
is invoked only when API keys are available in environment variables.

Public API:
    process_pdf(pdf_path, reference_csv_path=None) -> (csv_text, report)
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from extractor import extract_tables, ExtractionResult
from metadata import (
    build_document_metadata, metadata_preview_text, metadata_to_xlsx_bytes,
)
from mapper import map_to_reference, parse_reference, combine_tables

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Processing report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessingReport:
    mode: str = ""                   # "direct" | "with_reference"
    strategy_used: str = ""
    pages_processed: int = 0
    tables_found: int = 0
    rows_output: int = 0
    columns_output: int = 0
    warnings: list = field(default_factory=list)
    llm_used: bool = False
    output_ext: str = ".csv"     # ".csv" | ".xlsx"
    preview_text: str = ""       # CSV-style text for the UI preview pane
    metadata: Optional[dict] = None  # document metadata (direct mode)

    def summary(self) -> str:
        parts = [
            f"Mode: {self.mode}",
            f"Extraction engine(s): {self.strategy_used}",
            f"Tables found: {self.tables_found}",
            f"Output: {self.rows_output} rows × {self.columns_output} columns",
        ]
        if self.llm_used:
            parts.append("LLM column mapping: yes (optional enhancement)")
        if self.warnings:
            parts.append("Warnings: " + "; ".join(self.warnings))
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def df_to_csv(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a clean CSV string."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL)
    return buf.getvalue().strip()


def clean_csv_output(raw: str) -> str:
    """Strip any accidental markdown fences."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def _tables_to_csv_blocks(tables) -> str:
    """
    Serialise verbatim tables to CSV, one block per table in source order.

    Every grid row (including the source header row) is written as-is via
    csv.writer, so cell values survive untouched — embedded commas, quotes
    and newlines are handled by CSV quoting rather than by rewriting values.
    With a single table the output is plain CSV with no extra rows; with
    multiple tables each block is preceded by a one-cell marker row
    "=== Table N (page P) ===" and blocks are separated by a blank line so
    the file can be reviewed side-by-side with the PDF.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    multi = len(tables) > 1
    for i, t in enumerate(tables, start=1):
        if multi:
            if i > 1:
                buf.write("\n")
            writer.writerow([f"=== Table {i} (page {t.page}) ==="])
        for row in t.df.itertuples(index=False, name=None):
            writer.writerow(list(row))
    return buf.getvalue().rstrip("\n")


def tables_to_xlsx_bytes(tables) -> bytes:
    """
    Serialise verbatim tables to an .xlsx workbook, one sheet per table in
    source order ("Table 1 (page 1)", …). Every cell is written as a string
    so spreadsheet apps cannot coerce values on open (e.g. "007" stays
    "007", "1,000.50" stays "1,000.50") — which CSV import cannot guarantee.
    """
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    wb = Workbook()
    wb.remove(wb.active)
    for i, t in enumerate(tables, start=1):
        ws = wb.create_sheet(title=f"Table {i} (page {t.page})"[:31])
        for row in t.df.itertuples(index=False, name=None):
            ws.append([
                ILLEGAL_CHARACTERS_RE.sub("", "" if v is None else str(v))
                for v in row
            ])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def validate_csv(text: str) -> tuple[bool, str]:
    try:
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            return False, "Empty output"
        return True, ""
    except csv.Error as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    reference_csv_path: Optional[str] = None,
) -> tuple[str, ProcessingReport]:
    """
    Extract tables from a PDF and return clean CSV text.

    Args:
        pdf_path:            Path to the input PDF.
        reference_csv_path:  Optional path to a reference CSV that defines
                             the desired output schema.

    Returns:
        (csv_text, report)  where csv_text is valid CSV and report contains
                            metadata about the extraction.
    """
    report = ProcessingReport()

    has_ref = reference_csv_path and os.path.exists(reference_csv_path)

    # ── Direct mode: verbatim, structure-preserving extraction ───────────────
    # Each detected table is emitted exactly as it appears in the PDF:
    # same rows, same columns, same header labels, cell values untouched.
    # Multi-table documents produce one CSV block per table in source order.
    if not has_ref:
        report.mode = "direct"
        # Stage 1: PDF → document metadata (tables, grid geometry, cells
        # with spans and verbatim text). Stage 2: render outputs from the
        # metadata only, so the spreadsheet mirrors the source structure.
        doc = build_document_metadata(pdf_path)
        report.strategy_used = doc.engine or "none"
        report.tables_found = len(doc.tables)
        report.warnings.extend(doc.warnings)
        report.metadata = doc.to_dict()

        if not doc.tables:
            return "", report

        report.pages_processed = len({t.page for t in doc.tables})

        xlsx_bytes = metadata_to_xlsx_bytes(doc)
        report.output_ext = ".xlsx"
        report.preview_text = metadata_preview_text(doc)

        report.rows_output = sum(t.n_rows for t in doc.tables)
        report.columns_output = max(t.n_cols for t in doc.tables)
        return xlsx_bytes, report

    # ── Step 1: Extract tables (fully local, no API required) ────────────────
    extraction: ExtractionResult = extract_tables(pdf_path)
    report.strategy_used = extraction.strategy_used or "none"
    report.tables_found = len(extraction.tables)
    report.warnings.extend(extraction.warnings)

    if not extraction.tables:
        report.mode = "with_reference"
        return "", report

    report.pages_processed = len({t.page for t in extraction.tables})

    # ── Step 2: Combine all page tables into one DataFrame ───────────────────
    combined_df = combine_tables(extraction.tables)

    if combined_df.empty:
        report.mode = "with_reference"
        return "", report

    # Promote first row to header BEFORE mapping if columns look auto-generated
    if _header_looks_like_data(combined_df):
        new_header = combined_df.iloc[0].tolist()
        combined_df = combined_df.iloc[1:].reset_index(drop=True)
        combined_df.columns = [str(h).strip() for h in new_header]

    # ── Step 3: Reference-sheet mapping ──────────────────────────────────────
    report.mode = "with_reference"
    with open(reference_csv_path, newline="", encoding="utf-8-sig") as f:
        ref_text = f.read()

    ref_df = parse_reference(ref_text)

    if ref_df.empty:
        report.warnings.append("Could not parse reference sheet; returning unmapped extraction.")
        out_df = combined_df
    else:
        # Determine whether LLM refinement is available
        llm_available = bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        out_df = map_to_reference(combined_df, ref_df, use_llm=llm_available)
        report.llm_used = llm_available

    # ── Step 4: Finalise and validate ────────────────────────────────────────
    out_df = out_df.fillna("")

    # Promote first row to header if header looks like data
    # (happens when extractor couldn't identify a proper header row)
    if _header_looks_like_data(out_df):
        new_header = out_df.iloc[0].tolist()
        out_df = out_df.iloc[1:].reset_index(drop=True)
        out_df.columns = [str(h).strip() for h in new_header]

    csv_text = df_to_csv(out_df)
    csv_text = clean_csv_output(csv_text)
    report.preview_text = csv_text

    valid, err = validate_csv(csv_text)
    if not valid:
        report.warnings.append(f"CSV validation warning: {err}")

    report.rows_output = len(out_df)
    report.columns_output = len(out_df.columns)

    return csv_text, report


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header_looks_like_data(df: pd.DataFrame) -> bool:
    """
    Return True if the DataFrame's column names look like auto-generated
    indices (0, 1, 2, … or '0', '1', '2', …) which means the real header
    is in the first data row.
    """
    cols = [str(c) for c in df.columns]
    return all(re.match(r"^\d+$", c) for c in cols)
