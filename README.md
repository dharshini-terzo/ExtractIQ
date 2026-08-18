# PDF → CSV Converter

A **provider-agnostic**, local-first PDF table extraction application.  
No API key is required to run it. The app works out of the box with five
local extraction engines and produces clean, validated CSV output for any
PDF containing table data.

---

## Architecture overview

```
PDF file
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  extractor.py  — Strategy Ladder (all local, no API)        │
│                                                             │
│  1. pdfplumber   – fast text-layer extraction               │
│  2. camelot-lattice – bordered / grid tables                │
│  3. camelot-stream  – borderless / whitespace tables        │
│  4. tabula-py    – Java-based fallback                      │
│  5. text-heuristic  – raw text + delimiter detection        │
│                                                             │
│  Each table is scored for quality; best results survive.    │
│  Near-duplicates across engines are deduplicated.           │
└──────────────────────┬──────────────────────────────────────┘
                       │  List[ExtractedTable]
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  mapper.py  — Column Mapping                                │
│                                                             │
│  Rule-based (always available):                             │
│    – Fuzzy column-name matching (SequenceMatcher)           │
│    – Structural row detection (section / subtotal / data)   │
│    – Multi-table alignment by best-match headers            │
│                                                             │
│  LLM assist (optional, only when env key present):          │
│    – ANTHROPIC_API_KEY → claude-haiku (cheapest tier)       │
│    – OPENAI_API_KEY    → gpt-4o-mini                        │
│    – Only triggered when rule-based coverage < 70 %         │
│    – Falls back to rule-based on any API error / rate limit │
└──────────────────────┬──────────────────────────────────────┘
                       │  pd.DataFrame
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  processor.py  — Orchestration + Output                     │
│    – Combines page tables into one DataFrame                │
│    – Applies reference-sheet mapping when provided          │
│    – Validates CSV; returns (csv_text, ProcessingReport)    │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
              Clean, validated CSV
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) set an LLM key for enhanced column mapping
export ANTHROPIC_API_KEY=sk-ant-...   # OR
export OPENAI_API_KEY=sk-...

# 3. Run
python app.py

# 4. Open http://127.0.0.1:5000
```

---

## Usage

### Direct extraction (no reference sheet)
Upload a PDF → click **Extract & Download**.  
Direct mode converts the whole document into an ordered metadata stream
(metadata.py) — text rows and tables in reading order — and renders it
into **one xlsx sheet** that mirrors the source with no manual
rearranging:

* Table **rows** come from ruled horizontal lines; logical rows spanning
  several text lines (wrapped names, name-above-values layouts) are
  reassembled from column occupancy.
* Table **columns** come from interior ruled vertical lines when present,
  otherwise from whitespace gutters — vertical strips no text crosses
  (the layout of most real ordering documents).
* Tables that continue across pages (repeated headers) are merged into
  one table; repeating page headers/footers are dropped.
* Headings and text between tables become plain rows in reading order.
* Every cell is written as text so spreadsheet apps cannot coerce values
  ("007" stays "007", "1,000.50" stays "1,000.50").

The metadata JSON (tables, grid geometry, cells) is saved next to each
output and served at `/metadata/<job_id>` for inspection or downstream
modules.

**Scanned PDFs** are supported when tesseract is installed: words are
read with a dual-pass OCR (two segmentation modes, deduplicated), and any
table cell the page-level OCR missed is recovered by OCR'ing that cell's
crop individually. Structure always comes from the ruled lines and
whitespace gutters — never inferred from an LLM or fuzzy matching. A
warning reminds the user that OCR character accuracy on scans is not
guaranteed and values should be spot-checked.

### Reference-sheet mode
Upload a PDF **and** a reference CSV that shows the desired column
structure → click **Extract & Download CSV**.  
The extractor maps PDF columns to reference columns using fuzzy name
matching. If an LLM key is configured and the automatic match is weak
(< 70 % coverage), the LLM refines the mapping — otherwise rule-based
matching is used and the result is still clean.

---

## Why no hard API dependency?

| Concern | How it's addressed |
|---|---|
| API key not available | All five extraction engines are fully local |
| Rate limit / quota exhaustion | LLM step is optional; app never blocks on it |
| Token cost | LLM only runs on *column name disambiguation*, never on full document |
| Provider changes | Supports both Anthropic and OpenAI; trivial to add others |
| Different PDF structures | Five complementary engines cover: text-layer, bordered tables, borderless tables, Java-parsed, and raw text |

---

## File structure

```
pdfcsv_app/
├── app.py            Flask web application
├── processor.py      Orchestration layer
├── extractor.py      Multi-engine PDF extraction
├── mapper.py         Rule-based + optional LLM column mapping
├── requirements.txt
├── templates/
│   └── index.html
├── uploads/          Created at runtime (temp job files)
└── outputs/          Created at runtime (generated CSVs)
```
