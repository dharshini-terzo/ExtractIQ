#!/usr/bin/env python3
"""
app.py — PDF to CSV Converter
==============================
Provider-agnostic extraction pipeline.

Run:
    python3 app.py

Then open http://127.0.0.1:5000
"""

import os
import uuid
import traceback

from flask import (
    Flask, request, render_template,
    send_file, flash, redirect, url_for, jsonify
)
from werkzeug.utils import secure_filename

from processor import process_pdf

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_PDF  = {".pdf"}
ALLOWED_CSV  = {".csv"}
MAX_CONTENT  = 64 * 1024 * 1024  # 64 MB

app = Flask(__name__)
app.secret_key = "pdf-csv-converter-2025-generic"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT


def _allowed(filename, exts):
    return os.path.splitext(filename.lower())[1] in exts


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    pdf_file = request.files.get("pdf_file")
    ref_file = request.files.get("ref_file")

    if not pdf_file or pdf_file.filename == "":
        flash("Please upload a PDF file.")
        return redirect(url_for("index"))
    if not _allowed(pdf_file.filename, ALLOWED_PDF):
        flash("Only .pdf files are accepted.")
        return redirect(url_for("index"))

    has_ref = ref_file and ref_file.filename != ""
    if has_ref and not _allowed(ref_file.filename, ALLOWED_CSV):
        flash("The reference sheet must be a .csv file.")
        return redirect(url_for("index"))

    job_id  = uuid.uuid4().hex
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

    try:
        csv_text, report = process_pdf(pdf_path, ref_path)

        if not csv_text:
            detail = ""
            if report.warnings:
                detail = " | ".join(report.warnings)
            flash(f"No tables could be extracted from the PDF. {detail}")
            return redirect(url_for("index"))

        base_name    = os.path.splitext(pdf_name)[0]
        out_filename = f"{base_name}_output.csv"
        out_path     = os.path.join(OUTPUT_DIR, f"{job_id}_{out_filename}")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write(csv_text)

        return send_file(
            out_path,
            as_attachment=True,
            download_name=out_filename,
            mimetype="text/csv",
        )

    except Exception as exc:
        traceback.print_exc()
        flash(f"Processing error: {exc}")
        return redirect(url_for("index"))


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
