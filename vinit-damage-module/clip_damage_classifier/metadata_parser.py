"""
metadata_parser.py
============================================================
Safely parses banknote denomination and generation from image path.
DOES NOT use denomination as damage classification.
Returns blank strings if not confidently inferred.
============================================================
"""

import re
from pathlib import Path
from typing import Dict


def parse_banknote_metadata(relative_path: str, filename: str) -> Dict[str, str]:
    """
    Extracts denomination (10, 20, 50, 100, 200, 500, 2000) and
    note_generation ('old', 'new') from directory structure and filename.

    Does NOT guess; if uncertain, leaves fields blank.
    """
    path_clean = relative_path.replace("\\\\", "/").lower()
    fn_clean = filename.lower()
    combined = f"{path_clean} {fn_clean}"

    # Denomination extraction
    denomination = ""
    # Look for patterns like '10 rupees', 'new 10', '10_new', etc.
    # Check largest denominations first to avoid prefix confusion
    for denom in ["2000", "500", "200", "100", "50", "20", "10"]:
        pattern = rf"(?:^|[_\s/\\\-])(?:new|old)?\s*{denom}\s*(?:rupees|rs|rupee)?(?:[_\s/\\\-]|$)"
        if re.search(pattern, combined):
            denomination = denom
            break

    # Note generation extraction with strict word boundaries
    has_new = bool(re.search(r"(?:^|[_\s/\\\-])new(?:[_\s/\\\-]|$)", combined))
    has_old = bool(re.search(r"(?:^|[_\s/\\\-])old(?:[_\s/\\\-]|$)", combined))

    note_generation = ""
    if has_new and not has_old:
        note_generation = "new"
    elif has_old and not has_new:
        note_generation = "old"

    return {
        "denomination": denomination,
        "note_generation": note_generation
    }
