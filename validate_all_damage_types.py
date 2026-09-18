"""
validate_all_damage_types.py
============================================================
One-shot validation script for all damage types on available
test images. Run from project root.
============================================================
"""

import sys
import json
from pathlib import Path

SEVERITY_DIR = Path(__file__).resolve().parent / "severity-module"
if str(SEVERITY_DIR) not in sys.path:
    sys.path.insert(0, str(SEVERITY_DIR))

from interface import assess_damage

IMG_DIR = SEVERITY_DIR / "test_images"


def run(label, image_name, damage, denomination, confidence=0.95):
    img = IMG_DIR / image_name
    if not img.exists():
        print(f"\n[{label}] IMAGE NOT FOUND: {img}")
        return None

    result = assess_damage(
        image_input=img,
        damage_labels=[damage],
        currency=denomination,
        denomination_confidence=confidence,
        preprocessing_metadata={"note_detected": True},
        debug=True,
    )
    r = result.get("result", result)
    print(f"\n[{label}]")
    print(f"  IMAGE          : {image_name}")
    print(f"  DAMAGE TYPE    : {damage}")
    print(f"  DAMAGED PIXELS : {r.get('damaged_pixels', 0)}")
    print(f"  TOTAL PIXELS   : {r.get('total_note_pixels', 0)}")
    print(f"  DAMAGE %       : {r.get('damage_percentage', 0.0):.2f}%")
    print(f"  SEVERITY       : {r.get('severity', '?')}")
    dbg = r.get("debug_paths", {})
    vis = r.get("visualization", {})
    mask_key = f"{damage.lower()}_mask"
    print(f"  MASK PATH      : {dbg.get(mask_key, dbg.get('combined_mask', 'N/A'))}")
    print(f"  ANNOTATED PATH : {vis.get('annotated_path', 'N/A')}")
    return r


# ── STAIN ────────────────────────────────────────────────────
# img_2 is the smallest image — likely a clean or mildly stained note
stain_res = run("STAIN / img_2", "img_2.jpeg", "Stain", "500", 0.90)

# ── BURNT ────────────────────────────────────────────────────
# No burnt image present in repo — report accordingly
burnt_img = IMG_DIR / "burnt_note.jpeg"
if not burnt_img.exists():
    print("\n[BURNT]\n  REAL TEST IMAGE REQUIRED")

# ── FOLDED ───────────────────────────────────────────────────
# sample_note.jpeg — small clean note, tests for false positives
folded_res = run("FOLDED / sample_note", "sample_note.jpeg", "Folded", "500", 0.85)

# ── NORMAL REGRESSION ────────────────────────────────────────
normal_res = run("NORMAL / sample_note", "sample_note.jpeg", "Normal", "500", 0.95)

# ── TORN REGRESSION ──────────────────────────────────────────
torn_res = run("TORN REGRESSION / img_3", "img_3.jpeg", "Torn", "10", 0.97)

print("\n" + "=" * 60)
print("DONE")
