"""
zip_extractor.py
============================================================
Safely extracts local ZIP dataset preventing path traversal (ZipSlip).
Extracts only once if files already exist.
============================================================
"""

import os
import zipfile
from pathlib import Path
from typing import Tuple


def safe_extract_zip(zip_path: str, extract_to: str, force: bool = False) -> Tuple[str, int]:
    """
    Safely extracts zip_path to extract_to directory.
    Guarantees that files do not escape extract_to (anti-ZipSlip).

    Returns:
        (extracted_directory_path, count_of_extracted_images)
    """
    zip_p = Path(zip_path).resolve()
    if not zip_p.exists():
        raise FileNotFoundError(f"ZIP file not found at: {zip_p}")

    out_dir = Path(extract_to).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_images = [
        p for p in out_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    ]

    if existing_images and not force:
        print(f"[ZIP] Using already extracted dataset at: {out_dir} ({len(existing_images)} images found)")
        return str(out_dir), len(existing_images)

    print(f"[ZIP] Extracting {zip_p} to {out_dir} ...")
    extracted_count = 0
    with zipfile.ZipFile(zip_p, "r") as z:
        for member in z.infolist():
            # Anti-ZipSlip path check
            target_path = (out_dir / member.filename).resolve()
            if not str(target_path).startswith(str(out_dir)):
                raise ValueError(f"Security error: Zip traversal attempt detected in {member.filename}")

            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

            if target_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                extracted_count += 1

    print(f"[ZIP] Extraction complete: {extracted_count} valid images extracted to {out_dir}")
    return str(out_dir), extracted_count
