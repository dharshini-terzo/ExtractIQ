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

    # ── Step 1: Extract tables (fully local, no API required) ────────────────
    extraction: ExtractionResult = extract_tables(pdf_path)
    report.strategy_used = extraction.strategy_used or "none"
    report.tables_found = len(extraction.tables)
    report.warnings.extend(extraction.warnings)

    if not extraction.tables:
        report.mode = "direct"
        return "", report

    report.pages_processed = len({t.page for t in extraction.tables})

    # ── Step 2: Combine all page tables into one DataFrame ───────────────────
    combined_df = combine_tables(extraction.tables)

    if combined_df.empty:
        report.mode = "direct"
        return "", report

    # Promote first row to header BEFORE mapping if columns look auto-generated
    if _header_looks_like_data(combined_df):
        new_header = combined_df.iloc[0].tolist()
        combined_df = combined_df.iloc[1:].reset_index(drop=True)
        combined_df.columns = [str(h).strip() for h in new_header]

    # ── Step 3: Optional reference-sheet mapping ─────────────────────────────
    has_ref = reference_csv_path and os.path.exists(reference_csv_path)

    if has_ref:
        report.mode = "with_reference"
        with open(reference_csv_path, newline="", encoding="utf-8-sig") as f:
            ref_text = f.read()

        ref_df = parse_reference(ref_text)

        if ref_df.empty:
            report.warnings.append("Could not parse reference sheet; falling back to direct mode.")
            report.mode = "direct"
            out_df = combined_df
        else:
            # Determine whether LLM refinement is available
            llm_available = bool(
                os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
            )
            out_df = map_to_reference(combined_df, ref_df, use_llm=llm_available)
            report.llm_used = llm_available
    else:
        report.mode = "direct"
        out_df = combined_df

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
