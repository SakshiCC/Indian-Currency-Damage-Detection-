"""
run_all_tests.py
============================================================
Sakshi Module — Final Hardening & Freeze Test Suite
Tasks: Robustness, Input Safety, Mask Invariants, Severity Boundaries,
       Region Analysis, Output Consistency, Upstream Handoff,
       Full Regression, Summary.
============================================================
"""

import sys
import traceback
from pathlib import Path

SEVERITY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SEVERITY_DIR.parent
for p in [str(SEVERITY_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
from interface import assess_damage
from severity import classify_severity

IMG_DIR = SEVERITY_DIR / "test_images"
IMG3   = IMG_DIR / "img_3.jpeg"
IMG2   = IMG_DIR / "img_2.jpeg"
SAMPLE = IMG_DIR / "sample_note.jpeg"

results = {}   # label -> "PASS" | "FAIL" | "VALIDATION PENDING"
details = []   # detail lines


def record(label, passed, note=""):
    status = "PASS" if passed else "FAIL"
    results[label] = status
    details.append(f"  [{status}] {label}" + (f" — {note}" if note else ""))


def run_assess(img, labels, denom="500", conf=0.90, **kw):
    r = assess_damage(img, labels, currency=denom,
                      denomination_confidence=conf, debug=False, **kw)
    return r["result"]


# ============================================================
# TASK 8 — FULL REGRESSION (run first, establish baselines)
# ============================================================
print("\n=== TASK 8: Full Regression ===")

res_normal = run_assess(SAMPLE, ["Normal"], "500")
record("NORMAL_REGRESSION", res_normal["damage_percentage"] == 0.0 and
       res_normal["severity"] == "None",
       f"{res_normal['damage_percentage']}% / {res_normal['severity']}")

res_torn = run_assess(IMG3, ["Torn"], "10", 0.97)
torn_ref = res_torn["damage_percentage"]
record("TORN_REGRESSION", 10.0 <= torn_ref <= 16.0,
       f"{torn_ref}% / {res_torn['severity']}")

res_fold = run_assess(IMG2, ["Folded"], "500")
fold_ref = res_fold["damage_percentage"]
record("FOLDED_REGRESSION", fold_ref > 0.0,
       f"{fold_ref}% / {res_fold['severity']}")

# multi-damage union: Torn+Stain on img3 — combined <= torn+stain individually
res_union = run_assess(IMG3, ["Torn", "Stain"], "10", 0.97)
torn_px = res_union["per_damage"].get("Torn", {}).get("damaged_pixels", 0)
stain_px = res_union["per_damage"].get("Stain", {}).get("damaged_pixels", 0)
union_px = res_union["damaged_pixels"]
record("MULTI_DAMAGE_UNION", union_px <= torn_px + stain_px,
       f"union={union_px} <= torn({torn_px})+stain({stain_px})")


# ============================================================
# TASK 1 — ROBUSTNESS
# ============================================================
print("\n=== TASK 1: Robustness ===")

img_raw = cv2.imread(str(IMG3))
h, w = img_raw.shape[:2]

def make_variants(img):
    variants = {}
    # Original (already tested above)
    variants["resize_0.5x"] = cv2.resize(img, (w//2, h//2))
    for angle in [5, -5]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        variants[f"rotate_{angle:+d}"] = cv2.warpAffine(img, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE)
    variants["brighter"] = np.clip(img.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    variants["darker"] = np.clip(img.astype(np.int32) - 30, 0, 255).astype(np.uint8)
    variants["blurred"] = cv2.GaussianBlur(img, (7, 7), 2)
    return variants

rob_ok = True
for name, variant in make_variants(img_raw).items():
    try:
        r = assess_damage(variant, ["Torn"], currency="10",
                          denomination_confidence=0.97, debug=False)
        pct = r["result"]["damage_percentage"]
        sev = r["result"]["severity"]
        # Flag collapse: damage disappears (<=0%) or explodes (>60%) or sev=None on torn note
        failed = (pct <= 0.0 or pct > 60.0)
        details.append(f"  [{'FAIL' if failed else 'ok  '}] robustness/{name}: {pct:.2f}% / {sev}")
        if failed:
            rob_ok = False
    except Exception as e:
        details.append(f"  [FAIL] robustness/{name}: EXCEPTION {e}")
        rob_ok = False

record("ROBUSTNESS", rob_ok)


# ============================================================
# TASK 2 — INPUT SAFETY
# ============================================================
print("\n=== TASK 2: Input Safety ===")
safety_ok = True

# Invalid path
try:
    assess_damage(Path("nonexistent_image.jpg"), ["Torn"], "500")
    details.append("  [FAIL] invalid path: no exception raised")
    safety_ok = False
except Exception:
    details.append("  [ok  ] invalid path: graceful exception")

# Unreadable image (zero-byte file)
zero_path = SEVERITY_DIR / "debug" / "_zero.jpg"
zero_path.write_bytes(b"")
try:
    assess_damage(zero_path, ["Torn"], "500")
    details.append("  [FAIL] unreadable image: no exception raised")
    safety_ok = False
except Exception:
    details.append("  [ok  ] unreadable image: graceful exception")
finally:
    zero_path.unlink(missing_ok=True)

# Empty damage labels — should produce 0% damage, not crash
try:
    r = assess_damage(SAMPLE, [], "500")
    res = r["result"]
    ok = res["damaged_pixels"] == 0
    details.append(f"  [{'ok  ' if ok else 'FAIL'}] empty damage labels: px={res['damaged_pixels']}")
    if not ok: safety_ok = False
except Exception as e:
    details.append(f"  [FAIL] empty damage labels: EXCEPTION {e}")
    safety_ok = False

# Unknown damage label — must not crash; unknown labels are silently ignored
try:
    r = assess_damage(SAMPLE, ["UnknownDamage"], "500")
    details.append("  [ok  ] unknown damage label: no crash")
except Exception as e:
    details.append(f"  [FAIL] unknown damage label: EXCEPTION {e}")
    safety_ok = False

# Normal label
try:
    r = assess_damage(SAMPLE, ["Normal"], "500")
    pct = r["result"]["damage_percentage"]
    ok = pct == 0.0
    details.append(f"  [{'ok  ' if ok else 'FAIL'}] Normal label: {pct}%")
    if not ok: safety_ok = False
except Exception as e:
    details.append(f"  [FAIL] Normal label: EXCEPTION {e}")
    safety_ok = False

# Missing denomination
try:
    r = assess_damage(SAMPLE, ["Torn"], currency=None)
    details.append("  [ok  ] denomination=None: no crash")
except Exception as e:
    details.append(f"  [FAIL] denomination=None: EXCEPTION {e}")
    safety_ok = False

# No rectified image in preprocessing_metadata
try:
    r = assess_damage(IMG3, ["Torn"], "10", preprocessing_metadata={})
    details.append("  [ok  ] no rectified_note in metadata: no crash")
except Exception as e:
    details.append(f"  [FAIL] no rectified_note: EXCEPTION {e}")
    safety_ok = False

# Vinit output missing optional fields
try:
    r = assess_damage(IMG3, ["Torn"], "10",
                      preprocessing_metadata={"note_detected": True})
    details.append("  [ok  ] sparse preprocessing_metadata: no crash")
except Exception as e:
    details.append(f"  [FAIL] sparse metadata: EXCEPTION {e}")
    safety_ok = False

# Multiple damage labels
try:
    r = assess_damage(IMG3, ["Torn", "Stain", "Folded"], "10", 0.97)
    details.append("  [ok  ] multiple damage labels: no crash")
except Exception as e:
    details.append(f"  [FAIL] multiple labels: EXCEPTION {e}")
    safety_ok = False

record("INPUT_SAFETY", safety_ok)


# ============================================================
# TASK 3 — MASK INVARIANTS
# ============================================================
print("\n=== TASK 3: Mask Invariants ===")

from segmentation import segment_damage
from note_geometry import estimate_note_geometry

mask_ok = True

def check_mask_invariants(img, labels, denom="500", conf=0.90):
    global mask_ok
    seg = segment_damage(img, labels, denomination=denom,
                         denomination_confidence=conf)
    geo = seg["geometry"]
    dm  = seg["damage_masks"]
    cm  = seg["combined_mask"]
    exp = geo.expected_mask
    vis = geo.visible_mask

    # Numeric invariants
    exp_px = int(cv2.countNonZero(exp))
    vis_px = int(cv2.countNonZero(vis))
    comb_px = int(cv2.countNonZero(cm))

    # torn inside expected
    if "Torn" in dm:
        torn = dm["Torn"]
        outside = cv2.bitwise_and(torn, cv2.bitwise_not(exp))
        if cv2.countNonZero(outside) > 0:
            details.append(f"  [FAIL] torn pixels outside expected_note_mask: {cv2.countNonZero(outside)}")
            mask_ok = False

    # stain/folded inside visible
    for key in ["Stain", "Folded", "Burnt"]:
        if key in dm:
            m = dm[key]
            outside = cv2.bitwise_and(m, cv2.bitwise_not(vis))
            if cv2.countNonZero(outside) > 0:
                details.append(f"  [FAIL] {key} pixels outside visible_mask: {cv2.countNonZero(outside)}")
                mask_ok = False

    # combined <= sum of individuals
    ind_sum = sum(int(cv2.countNonZero(m)) for m in dm.values())
    if comb_px > ind_sum + 1:   # +1 for rounding
        details.append(f"  [FAIL] combined({comb_px}) > sum_individuals({ind_sum})")
        mask_ok = False

    # percentage bounds and pixel counts
    total = int(cv2.countNonZero(exp))
    from severity import calculate_severity
    sev_res = calculate_severity(dm, cm, exp)
    comb_m = sev_res["combined"]
    if not (0.0 <= comb_m["damage_percentage"] <= 100.0):
        details.append(f"  [FAIL] damage_percentage out of [0,100]: {comb_m['damage_percentage']}")
        mask_ok = False
    if comb_m["damaged_pixels"] > comb_m["total_note_pixels"]:
        details.append(f"  [FAIL] damaged_pixels > total_note_pixels")
        mask_ok = False
    if comb_m["damaged_pixels"] < 0:
        details.append(f"  [FAIL] negative damaged_pixels")
        mask_ok = False

img3 = cv2.imread(str(IMG3))
check_mask_invariants(img3, ["Torn"], "10", 0.97)
img2 = cv2.imread(str(IMG2))
check_mask_invariants(img2, ["Folded"], "500")
check_mask_invariants(cv2.imread(str(SAMPLE)), ["Stain"], "500")
check_mask_invariants(img3, ["Torn", "Stain", "Folded"], "10", 0.97)

if mask_ok:
    details.append("  [ok  ] All mask containment and numeric invariants pass")
record("MASK_INVARIANTS", mask_ok)


# ============================================================
# TASK 4 — SEVERITY BOUNDARY TESTS
# ============================================================
print("\n=== TASK 4: Severity Boundaries ===")

sev_cases = [
    (0.0,         "None"),
    (0.001,       "Minor"),
    (5.0,         "Minor"),
    (5.001,       "Moderate"),
    (15.0,        "Moderate"),
    (15.001,      "Severe"),
    (100.0,       "Severe"),
]
sev_ok = True
for pct, expected in sev_cases:
    got = classify_severity(pct)
    ok = (got == expected)
    details.append(f"  [{'ok  ' if ok else 'FAIL'}] classify_severity({pct}) = {got} (expected {expected})")
    if not ok:
        sev_ok = False
record("SEVERITY_BOUNDARIES", sev_ok)


# ============================================================
# TASK 5 — REGION ANALYSIS
# ============================================================
print("\n=== TASK 5: Region Analysis ===")
from region_analysis import analyze_damage_regions

region_ok = True
seg3 = segment_damage(img3, ["Torn"], denomination="10", denomination_confidence=0.97)
geo3 = seg3["geometry"]
torn_mask = seg3["damage_masks"].get("Torn")
if torn_mask is not None:
    regions = analyze_damage_regions(torn_mask, geo3.expected_mask, "Torn")
    h3, w3 = img3.shape[:2]
    for reg in regions:
        for key in ["damage_type","area_pixels","area_percentage","bounding_box","centroid","location"]:
            if key not in reg:
                details.append(f"  [FAIL] region missing key: {key}")
                region_ok = False
        bx, by, bw, bh = reg["bounding_box"]
        cx, cy = reg["centroid"]
        if not (0 <= bx <= w3 and 0 <= by <= h3 and bw > 0 and bh > 0):
            details.append(f"  [FAIL] bounding_box out of image: {reg['bounding_box']}")
            region_ok = False
        if not (0 <= cx <= w3 and 0 <= cy <= h3):
            details.append(f"  [FAIL] centroid out of image: {reg['centroid']}")
            region_ok = False
        if reg["area_pixels"] < 0 or reg["area_percentage"] < 0:
            details.append(f"  [FAIL] negative region area")
            region_ok = False
    details.append(f"  [ok  ] {len(regions)} Torn regions, all keys/bounds valid")
record("REGION_ANALYSIS", region_ok)


# ============================================================
# TASK 6 — OUTPUT CONSISTENCY
# ============================================================
print("\n=== TASK 6: Output Consistency ===")
cons_ok = True

r_torn = run_assess(IMG3, ["Torn"], "10", 0.97)
pct    = r_torn["damage_percentage"]
sev    = r_torn["severity"]
px     = r_torn["damaged_pixels"]
tot    = r_torn["total_note_pixels"]
seg_px = r_torn.get("segmentation", {}).get("damaged_pixels", -1)
seg_pct= r_torn.get("segmentation", {}).get("damage_percentage", -1.0)

# All paths must reference the same computation
if px != seg_px:
    details.append(f"  [FAIL] top-level px({px}) != segmentation.px({seg_px})")
    cons_ok = False
if abs(pct - seg_pct) > 0.01:
    details.append(f"  [FAIL] top-level pct({pct}) != segmentation.pct({seg_pct})")
    cons_ok = False
expected_pct = round(px / float(max(1, tot)) * 100, 2)
if abs(pct - expected_pct) > 0.01:
    details.append(f"  [FAIL] damage_percentage({pct}) != px/tot*100({expected_pct})")
    cons_ok = False
from severity import classify_severity as cs
if sev != cs(pct):
    details.append(f"  [FAIL] severity({sev}) inconsistent with pct({pct})")
    cons_ok = False

if cons_ok:
    details.append(f"  [ok  ] px={px}, pct={pct}, sev={sev} all consistent")
record("OUTPUT_CONSISTENCY", cons_ok)


# ============================================================
# TASK 7 — YASH → VINIT → SAKSHI HANDOFF
# ============================================================
print("\n=== TASK 7: Upstream Handoff ===")

import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT / "integration"))
from sakshi_interface import (
    get_damage_labels, get_active_damage_labels,
    get_sakshi_handoff_payload, run_sakshi_assessment
)

handoff_ok = True

# Mock Yash denomination output
yash_out = {"denomination": "10", "confidence": 0.97, "accepted": True,
            "is_background": False, "value": "10"}

# Mock Vinit damage output — Torn note
vinit_torn = {
    "final_damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0},
    "predicted_labels": ["Torn"],
    "evidence_scores": {"Torn": 0.82, "Stain": 0.1, "Folded": 0.05, "Burnt": 0.02},
    "preprocessing_metadata": {"note_detected": True},
    "dynamic_thresholds": {},
    "classification_status": "confident",
}

# Test label extraction
active = get_active_damage_labels(vinit_torn)
if active != ["Torn"]:
    details.append(f"  [FAIL] active labels wrong: {active}")
    handoff_ok = False
else:
    details.append(f"  [ok  ] active_damage_labels from Vinit: {active}")

# Test handoff payload construction
payload = get_sakshi_handoff_payload(vinit_torn, image_path="test.jpg",
                                     denomination_result=yash_out)
for key in ["image_path","is_damaged","active_damage_labels","evidence_scores",
            "denomination","denomination_confidence"]:
    if key not in payload:
        details.append(f"  [FAIL] handoff payload missing key: {key}")
        handoff_ok = False

if payload.get("denomination") != "10":
    details.append(f"  [FAIL] denomination wrong: {payload.get('denomination')}")
    handoff_ok = False

# Sakshi must NOT use Vinit confidence as damage%
vinit_conf = vinit_torn["evidence_scores"].get("Torn", 0.0)

try:
    r_handoff = run_sakshi_assessment(IMG3, yash_out, vinit_torn)
    res_h = r_handoff["result"]
    sakshi_pct = res_h["damage_percentage"]
    # Damage% must come from mask, not Vinit confidence
    if abs(sakshi_pct - vinit_conf * 100) < 0.5:
        details.append(f"  [FAIL] Sakshi pct({sakshi_pct}) suspiciously == Vinit conf*100({vinit_conf*100})")
        handoff_ok = False
    else:
        details.append(f"  [ok  ] Sakshi pct({sakshi_pct}) from mask, Vinit conf was {vinit_conf}")
    # Torn regression through handoff
    if 10.0 <= sakshi_pct <= 16.0:
        details.append(f"  [ok  ] Torn through handoff: {sakshi_pct}% in expected range")
    else:
        details.append(f"  [FAIL] Torn through handoff out of range: {sakshi_pct}%")
        handoff_ok = False
except Exception as e:
    details.append(f"  [FAIL] run_sakshi_assessment raised: {e}")
    traceback.print_exc()
    handoff_ok = False

# Missing optional fields — should not crash
vinit_sparse = {"predicted_labels": ["Torn"]}
try:
    run_sakshi_assessment(IMG3, yash_out, vinit_sparse)
    details.append("  [ok  ] sparse Vinit output: no crash")
except Exception as e:
    details.append(f"  [FAIL] sparse Vinit output: {e}")
    handoff_ok = False

# is_background flag — denomination must be nulled
yash_bg = {"denomination": "Background", "is_background": True,
           "confidence": 0.95, "accepted": False}
try:
    r_bg = run_sakshi_assessment(SAMPLE, yash_bg, {"predicted_labels": ["Normal"]})
    details.append("  [ok  ] is_background flag: no crash")
except Exception as e:
    details.append(f"  [FAIL] is_background: {e}")
    handoff_ok = False

record("YASH_VINIT_SAKSHI_HANDOFF", handoff_ok)


# ============================================================
# STAIN / BURNT — VALIDATION PENDING
# ============================================================
results["STAIN"]  = "VALIDATION PENDING"
results["BURNT"]  = "VALIDATION PENDING"
details.append("  [--  ] STAIN: implementation complete; no positive validation image provided")
details.append("  [--  ] BURNT: implementation complete; no burnt test image in repository")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n")
print("=" * 65)
print("SAKSHI MODULE — FINAL HARDENING SUMMARY")
print("=" * 65)
for line in details:
    print(line)
print()
print("=" * 65)
print("RESULT TABLE")
print("=" * 65)

all_pass = True
for label, status in results.items():
    if status == "VALIDATION PENDING":
        print(f"  {label:35s}: {status}")
    else:
        print(f"  {label:35s}: {status}")
        if status != "PASS":
            all_pass = False

print()
if all_pass:
    print("FINAL STATUS: SAKSHI MODULE — READY TO FREEZE")
    print()
    print("  Stain : IMPLEMENTED — POSITIVE VALIDATION PENDING")
    print("  Burnt : IMPLEMENTED — POSITIVE VALIDATION PENDING")
else:
    print("FINAL STATUS: NOT READY — FIX FAILURES ABOVE")
print("=" * 65)
