"""
dataset_scanner.py
============================================================
Recursively locates all valid image files in extracted dataset directory.
Preserves relative path and filename.
============================================================
"""

import os
from pathlib import Path
from typing import List, Dict
from . import config


def scan_images(dataset_dir: str) -> List[Dict[str, str]]:
    """
    Recursively scans dataset_dir for all supported image formats.
    
    Returns:
        List of dicts:
        {
            "image_path": absolute_path,
            "relative_path": relative_path_from_dataset_dir,
            "filename": basename
        }
    """
    root = Path(dataset_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    image_entries = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in config.SUPPORTED_IMAGE_EXTENSIONS:
                full_path = Path(dirpath) / f
                try:
                    rel_path = full_path.relative_to(root)
                except ValueError:
                    rel_path = f
                image_entries.append({
                    "image_path": str(full_path),
                    "relative_path": str(rel_path).replace("\\", "/"),
                    "filename": f
                })

    # Sort deterministically by relative path
    image_entries.sort(key=lambda x: x["relative_path"])
    return image_entries
