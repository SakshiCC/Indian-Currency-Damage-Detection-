"""
csv_writer.py
============================================================
CSV writer matching the required multi-label contrastive schema:
image_path, relative_path, filename, denomination, note_generation,
Torn, Folded, Burnt, Stain, Normal, predicted_labels,
clip_torn_positive_score, clip_torn_negative_score, torn_evidence_score,
clip_folded_positive_score, clip_folded_negative_score, folded_evidence_score,
clip_burnt_positive_score, clip_burnt_negative_score, burnt_evidence_score,
clip_stain_positive_score, clip_stain_negative_score, stain_evidence_score,
classification_status
============================================================
"""

import os
import csv
from pathlib import Path
from typing import Dict, Any, List

MAIN_CSV_HEADERS = [
    "image_path",
    "relative_path",
    "filename",
    "denomination",
    "note_generation",
    "Torn",
    "Folded",
    "Burnt",
    "Stain",
    "Normal",
    "predicted_labels",
    "clip_torn_positive_score",
    "clip_torn_negative_score",
    "torn_evidence_score",
    "clip_folded_positive_score",
    "clip_folded_negative_score",
    "folded_evidence_score",
    "clip_burnt_positive_score",
    "clip_burnt_negative_score",
    "burnt_evidence_score",
    "clip_stain_positive_score",
    "clip_stain_negative_score",
    "stain_evidence_score",
    "classification_status"
]

ERROR_CSV_HEADERS = [
    "image_path",
    "relative_path",
    "error_type",
    "error_message"
]

AMBIGUOUS_CSV_HEADERS = [
    "image_path",
    "relative_path",
    "filename",
    "predicted_labels",
    "torn_evidence_score",
    "folded_evidence_score",
    "burnt_evidence_score",
    "stain_evidence_score",
    "classification_status"
]


class IncrementalCSVWriter:
    def __init__(self, filepath: str, overwrite: bool = False):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.filepath.exists() and self.filepath.stat().st_size > 0

        mode = "w" if overwrite or not file_exists else "a"
        self.file = open(self.filepath, mode, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=MAIN_CSV_HEADERS)

        if mode == "w":
            self.writer.writeheader()
            self.file.flush()

    def write_rows(self, rows: List[Dict[str, Any]]):
        for row in rows:
            clean_row = {k: row.get(k, "") for k in MAIN_CSV_HEADERS}
            self.writer.writerow(clean_row)
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class ErrorLogWriter:
    def __init__(self, filepath: str, overwrite: bool = False):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.filepath.exists() and self.filepath.stat().st_size > 0

        mode = "w" if overwrite or not file_exists else "a"
        self.file = open(self.filepath, mode, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=ERROR_CSV_HEADERS)

        if mode == "w":
            self.writer.writeheader()
            self.file.flush()

    def log_error(self, image_path: str, relative_path: str, error_type: str, error_msg: str):
        self.writer.writerow({
            "image_path": image_path,
            "relative_path": relative_path,
            "error_type": error_type,
            "error_message": str(error_msg).replace("\n", " ")
        })
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class AmbiguousCSVWriter:
    def __init__(self, filepath: str, overwrite: bool = False):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.filepath.exists() and self.filepath.stat().st_size > 0

        mode = "w" if overwrite or not file_exists else "a"
        self.file = open(self.filepath, mode, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=AMBIGUOUS_CSV_HEADERS)

        if mode == "w":
            self.writer.writeheader()
            self.file.flush()

    def write_rows(self, rows: List[Dict[str, Any]]):
        for row in rows:
            clean_row = {
                "image_path": row.get("image_path", ""),
                "relative_path": row.get("relative_path", ""),
                "filename": row.get("filename", ""),
                "predicted_labels": row.get("predicted_labels", ""),
                "torn_evidence_score": row.get("torn_evidence_score", ""),
                "folded_evidence_score": row.get("folded_evidence_score", ""),
                "burnt_evidence_score": row.get("burnt_evidence_score", ""),
                "stain_evidence_score": row.get("stain_evidence_score", ""),
                "classification_status": row.get("classification_status", "")
            }
            self.writer.writerow(clean_row)
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()
