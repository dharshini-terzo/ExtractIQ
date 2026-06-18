#!/usr/bin/env python3
"""
mapper.py
=========
Maps extracted DataFrames to a reference CSV schema.

Two modes:
  1. Rule-based (always available, no API key needed)
     – Column-name fuzzy matching
     – Structural pattern replication (section headers, subtotals)
     – Data-type alignment

  2. LLM-assisted (optional, activated only when ANTHROPIC_API_KEY or
     OPENAI_API_KEY env vars are present AND the rule-based score is low)
     – Passes column names + sample rows to a language model for
       disambiguation of ambiguous column mappings.
     – Falls back gracefully to rule-based if the API call fails or
       hits rate limits.

The caller always gets a valid CSV back regardless of LLM availability.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Column-name similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s) -> str:
    """Normalise a column name for comparison."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _best_match(target: str, candidates: List[str]) -> Tuple[Optional[str], float]:
    """Return (best_candidate, score) for a target column name."""
    best_col, best_score = None, 0.0
    for c in candidates:
        s = _similarity(target, c)
        if s > best_score:
            best_score = s
            best_col = c
    return best_col, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Reference sheet parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_reference(ref_csv_text: str) -> pd.DataFrame:
    """Parse reference CSV text into a DataFrame."""
    try:
        return pd.read_csv(io.StringIO(ref_csv_text), dtype=str).fillna("")
    except Exception:
        # Try with different encodings / separators
        for sep in [";", "\t", "|"]:
            try:
                return pd.read_csv(io.StringIO(ref_csv_text), sep=sep, dtype=str).fillna("")
            except Exception:
                pass
        return pd.DataFrame()


def reference_columns(ref_df: pd.DataFrame) -> List[str]:
    return list(ref_df.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based column mapping
# ─────────────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.35  # below this, leave column unmapped


def build_column_map(
    source_cols: List[str],
    target_cols: List[str],
    positional_fallback: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Return {target_col: source_col_or_None} for each target column.
    Uses greedy best-match; each source col is used at most once.

    When positional_fallback=True and all similarity scores are very low
    (likely because column names are semantically different, e.g. "Product"
    vs "Name"), fall back to positional alignment so data is still mapped.
    """
    remaining_sources = list(source_cols)
    mapping: Dict[str, Optional[str]] = {}

    for tgt in target_cols:
        best_src, best_score = _best_match(tgt, remaining_sources)
        if best_score >= SIMILARITY_THRESHOLD:
            mapping[tgt] = best_src
            remaining_sources.remove(best_src)
        else:
            mapping[tgt] = None

    # --- Positional fallback ---
    # If less than half the columns mapped by name and we have a 1:1 count,
    # assume columns are in the same order despite different names.
    mapped_count = sum(1 for v in mapping.values() if v is not None)
    if (positional_fallback
            and mapped_count < len(target_cols) / 2
            and len(source_cols) == len(target_cols)):
        positional: Dict[str, Optional[str]] = {}
        for tgt, src in zip(target_cols, source_cols):
            positional[tgt] = src
        return positional

    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Structural pattern detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_section_header_row(row: pd.Series, num_cols: int) -> bool:
    """Heuristic: row where only 1-2 cells are non-empty = section header."""
    non_empty = sum(1 for v in row if str(v).strip())
    return 0 < non_empty <= max(2, num_cols // 4)


def _is_numeric(val: str) -> bool:
    return bool(re.match(r"^-?[\d,\.]+%?$", str(val).strip()))


def _detect_row_types(df: pd.DataFrame) -> List[str]:
    """
    Assign a type tag to each row: 'header', 'data', 'section', 'subtotal', 'empty'.
    """
    types = []
    num_cols = len(df.columns)
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row]
        non_empty = [v for v in vals if v]
        if not non_empty:
            types.append("empty")
        elif _is_section_header_row(row, num_cols):
            types.append("section")
        elif any(kw in " ".join(vals).lower() for kw in ("total", "subtotal", "sum", "grand")):
            types.append("subtotal")
        else:
            types.append("data")
    return types


# ─────────────────────────────────────────────────────────────────────────────
# LLM-assisted mapping (optional, graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _llm_refine_mapping(
    source_cols: List[str],
    target_cols: List[str],
    sample_rows: List[List[str]],
    rule_mapping: Dict[str, Optional[str]],
) -> Dict[str, Optional[str]]:
    """
    Ask an LLM to improve low-confidence column mappings.
    Returns the original rule_mapping on any failure.
    """
    # Determine which provider to use
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not anthropic_key and not openai_key:
        return rule_mapping

    prompt = (
        "You are a data schema mapping expert.\n\n"
        f"SOURCE COLUMNS (from extracted PDF):\n{json.dumps(source_cols)}\n\n"
        f"TARGET COLUMNS (from reference sheet):\n{json.dumps(target_cols)}\n\n"
        "SAMPLE SOURCE ROWS (first 3):\n"
        + "\n".join(json.dumps(r) for r in sample_rows[:3])
        + "\n\n"
        "CURRENT MAPPING (target → source, null means unmapped):\n"
        + json.dumps(rule_mapping, indent=2)
        + "\n\n"
        "Improve this mapping. Rules:\n"
        "- Map each target column to the most semantically appropriate source column.\n"
        "- Each source column may be used at most once.\n"
        "- If no source column fits a target, keep it null.\n"
        "- Return ONLY a JSON object with the same structure as CURRENT MAPPING.\n"
        "- No explanation, no markdown fences.\n"
    )

    try:
        if anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",  # cheapest / fastest
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:
            import urllib.request
            payload = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            }).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            raw = data["choices"][0]["message"]["content"].strip()

        # Strip potential fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
        refined = json.loads(raw)

        # Validate: must be same keys
        if set(refined.keys()) == set(rule_mapping.keys()):
            return refined
    except Exception as exc:
        logger.warning("LLM mapping refinement failed (falling back to rule-based): %s", exc)

    return rule_mapping


# ─────────────────────────────────────────────────────────────────────────────
# Main mapping function
# ─────────────────────────────────────────────────────────────────────────────

def map_to_reference(
    extracted_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    use_llm: bool = True,
) -> pd.DataFrame:
    """
    Map extracted_df into the column structure of ref_df.

    Steps:
      1. Build rule-based column mapping (always runs)
      2. Optionally refine with LLM if API keys present and mapping quality is low
      3. Apply mapping and produce output DataFrame with ref_df's column order
    """
    if extracted_df.empty or ref_df.empty:
        return extracted_df

    source_cols = list(extracted_df.columns)
    target_cols = list(ref_df.columns)

    # --- Step 1: rule-based mapping ---
    col_map = build_column_map(source_cols, target_cols)

    # --- Step 2: optional LLM refinement ---
    mapped_count = sum(1 for v in col_map.values() if v is not None)
    coverage = mapped_count / len(target_cols) if target_cols else 1.0

    if use_llm and coverage < 0.7:
        sample_rows = extracted_df.head(3).values.tolist()
        col_map = _llm_refine_mapping(source_cols, target_cols, sample_rows, col_map)

    # --- Step 3: build output ---
    out_rows = []
    row_types = _detect_row_types(extracted_df)

    for i, (_, row) in enumerate(extracted_df.iterrows()):
        rtype = row_types[i]
        out_row = {}
        for tgt_col in target_cols:
            src_col = col_map.get(tgt_col)
            if rtype == "section":
                # Preserve section header in first column, blank the rest
                if tgt_col == target_cols[0]:
                    # Find the non-empty value
                    non_empty = [str(v).strip() for v in row if str(v).strip()]
                    out_row[tgt_col] = non_empty[0] if non_empty else ""
                else:
                    out_row[tgt_col] = ""
            elif src_col and src_col in extracted_df.columns:
                out_row[tgt_col] = str(row[src_col]).strip()
            else:
                out_row[tgt_col] = ""
        out_rows.append(out_row)

    return pd.DataFrame(out_rows, columns=target_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Combine multiple page tables into one coherent DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def combine_tables(tables) -> pd.DataFrame:
    """
    Merge a list of ExtractedTable objects into a single DataFrame.

    Handles:
    - Tables with the same columns: simple concat
    - Tables with different columns: align by best-match before concat
    """
    if not tables:
        return pd.DataFrame()

    dfs = [t.df for t in tables if not t.df.empty]
    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0]

    # Check if all tables share the same columns
    first_cols = list(dfs[0].columns)
    if all(list(df.columns) == first_cols for df in dfs[1:]):
        combined = pd.concat(dfs, ignore_index=True)
        return combined

    # Columns differ — try to align them
    # Use the largest table's columns as the reference
    dfs_sorted = sorted(dfs, key=lambda d: d.shape[0], reverse=True)
    ref_cols = list(dfs_sorted[0].columns)

    aligned = []
    for df in dfs:
        col_map = build_column_map(list(df.columns), ref_cols)
        renamed = {}
        for tgt, src in col_map.items():
            if src and src in df.columns:
                renamed[src] = tgt
        df_renamed = df.rename(columns=renamed)
        # Add any missing columns
        for col in ref_cols:
            if col not in df_renamed.columns:
                df_renamed[col] = ""
        aligned.append(df_renamed[ref_cols])

    return pd.concat(aligned, ignore_index=True)
