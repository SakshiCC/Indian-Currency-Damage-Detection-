"""
resume_manager.py
============================================================
Manages resume capability by inspecting existing CSV outputs.
Ensures zero duplicate rows on pipeline restart.
============================================================
"""

import os
import csv
from pathlib import Path
from typing import Set


def get_already_processed_keys(csv_path: str) -> Set[str]:
    """
    Reads existing output CSV and returns a set of unique relative_paths
    that have already been processed.
    """
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()

    processed = set()
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "relative_path" in reader.fieldnames:
                for row in reader:
                    rel_p = row.get("relative_path", "").strip()
                    if rel_p:
                        processed.add(rel_p)
    except Exception as e:
        print(f"[RESUME] Warning: Could not read existing CSV for resume: {e}")

    return processed
