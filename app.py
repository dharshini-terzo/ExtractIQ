#!/usr/bin/env python3
"""
app.py — PDF to CSV Converter
==============================
Provider-agnostic extraction pipeline with an asynchronous job model.

Each upload starts a background job identified by a unique job_id. The page
submits over AJAX, polls /status/<job_id> for progress, and then fetches a
fresh result + download for that exact job — so a new upload can never show a
previous upload's output, the progress spinner always clears, and the long
OCR step never blocks the request/UI.

Run:
    python3 app.py
Then open http://127.0.0.1:5001
"""

import json
import os
import time
import uuid
import threading
import traceback

from flask import (
    Flask, request, render_template, send_file, jsonify, abort, make_response,
)
from werkzeug.utils import secure_filename

from processor import process_pdf

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_PDF = {".pdf"}
ALLOWED_CSV = {".csv"}
MAX_CONTENT = 64 * 1024 * 1024  # 64 MB

app = Flask(__name__)
app.secret_key = "pdf-csv-converter-2025-generic"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT

# ── In-memory job registry ───────────────────────────────────────────────────
# job_id -> {state, message, error, out_path, filename, report, created}
JOBS: dict = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 60 * 60  # forget finished jobs after an hour


def _allowed(filename, exts):
    return os.path.splitext(filename.lower())[1] in exts


def _set_job(job_id, **kw):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def _get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def _prune_jobs():
    now = time.time()
    with JOBS_LOCK:
        stale = [j for j, v in JOBS.items()
                 if v.get("state") in ("done", "error")
                 and now - v.get("created", now) > JOB_TTL_SECONDS]
        for j in stale:
            JOBS.pop(j, None)


def _run_job(job_id, pdf_path, ref_path, out_path, filename):
    """Background worker: run the (slow) extraction and record the result."""
    try:
        _set_job(job_id, state="running", message="Extracting tables from the PDF…")
        output, report = process_pdf(pdf_path, ref_path)

        if not output:
            detail = " | ".join(report.warnings) if report.warnings else ""
            _set_job(job_id, state="error",
                     error=f"No tables could be extracted from the PDF. {detail}".strip())
            return

        # Direct mode returns .xlsx bytes; reference mode returns CSV text.
        if isinstance(output, bytes):
            with open(out_path, "wb") as f:
                f.write(output)
        else:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                f.write(output)

        preview_text = getattr(report, "preview_text", "") or (
            output if isinstance(output, str) else "")
        preview_lines = preview_text.splitlines()

        # Persist the document metadata (grid geometry, cells, spans) next
        # to the output so it can be inspected via /metadata/<job_id>.
        meta_path = ""
        meta = getattr(report, "metadata", None)
        if meta:
            meta_path = os.path.splitext(out_path)[0] + "_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        _set_job(
            job_id, state="done", message="Done.", out_path=out_path,
            filename=filename, rows=len(preview_lines),
            preview=preview_lines[:300], meta_path=meta_path,
            strategy=getattr(report, "strategy_used", ""),
            warnings=list(getattr(report, "warnings", []) or []),
        )
    except Exception as exc:
        traceback.print_exc()
        _set_job(job_id, state="error", error=f"Processing error: {exc}")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    _prune_jobs()
    pdf_file = request.files.get("pdf_file")
    ref_file = request.files.get("ref_file")

    if not pdf_file or pdf_file.filename == "":
        return jsonify(error="Please upload a PDF file."), 400
    if not _allowed(pdf_file.filename, ALLOWED_PDF):
        return jsonify(error="Only .pdf files are accepted."), 400

    has_ref = ref_file and ref_file.filename != ""
    if has_ref and not _allowed(ref_file.filename, ALLOWED_CSV):
        return jsonify(error="The reference sheet must be a .csv file."), 400

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    pdf_name = secure_filename(pdf_file.filename) or "input.pdf"
    pdf_path = os.path.join(job_dir, pdf_name)
    pdf_file.save(pdf_path)

    ref_path = None
    if has_ref:
        ref_name = secure_filename(ref_file.filename) or "reference.csv"
        ref_path = os.path.join(job_dir, ref_name)
        ref_file.save(ref_path)

    # Direct mode delivers a real spreadsheet (one sheet per table);
    # reference mode delivers a single mapped CSV.
    out_ext = ".csv" if has_ref else ".xlsx"
    base_name = os.path.splitext(pdf_name)[0]
    filename = f"{base_name}_output{out_ext}"
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}_{filename}")

    _set_job(job_id, state="queued", message="Queued…", created=time.time(),
             filename=filename)
    threading.Thread(
        target=_run_job, args=(job_id, pdf_path, ref_path, out_path, filename),
        daemon=True,
    ).start()

    resp = jsonify(job_id=job_id)
    resp.headers["Cache-Control"] = "no-store"
    return resp, 202


@app.route("/status/<job_id>")
def status(job_id):
    job = _get_job(job_id)
    if not job:
        return jsonify(state="unknown", error="Unknown job."), 404
    payload = {
        "state": job.get("state"),
        "message": job.get("message", ""),
        "error": job.get("error", ""),
        "rows": job.get("rows"),
        "strategy": job.get("strategy", ""),
        "warnings": job.get("warnings", []),
        "filename": job.get("filename", ""),
    }
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/preview/<job_id>")
def preview(job_id):
    job = _get_job(job_id)
    if not job or job.get("state") != "done":
        return jsonify(error="Result not ready."), 404
    # Preview lines are stored on the job (the output file itself may be
    # binary .xlsx); fall back to reading the file for legacy CSV jobs.
    lines = job.get("preview")
    total = job.get("rows", 0)
    if lines is None:
        out_path = job.get("out_path")
        if not out_path or not os.path.exists(out_path):
            return jsonify(error="Result file missing."), 404
        with open(out_path, encoding="utf-8") as f:
            all_lines = f.read().splitlines()
        lines, total = all_lines[:300], len(all_lines)
    resp = jsonify(filename=job.get("filename", "output.csv"),
                   total_rows=total, preview=lines)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/download/<job_id>")
def download(job_id):
    job = _get_job(job_id)
    if not job or job.get("state") != "done":
        abort(404)
    out_path = job.get("out_path")
    if not out_path or not os.path.exists(out_path):
        abort(404)
    download_name = job.get("filename", "output.csv")
    mimetype = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if download_name.endswith(".xlsx") else "text/csv"
    )
    resp = make_response(send_file(
        out_path, as_attachment=True,
        download_name=download_name, mimetype=mimetype,
    ))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/metadata/<job_id>")
def metadata(job_id):
    job = _get_job(job_id)
    if not job or job.get("state") != "done":
        return jsonify(error="Result not ready."), 404
    meta_path = job.get("meta_path")
    if not meta_path or not os.path.exists(meta_path):
        return jsonify(error="No metadata for this job."), 404
    resp = make_response(send_file(meta_path, mimetype="application/json"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # threaded=True so background jobs and status polls run concurrently.
    app.run(debug=True, port=int(os.environ.get("PORT", 5001)), threaded=True)
