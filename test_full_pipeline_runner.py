import sys
from pathlib import Path
import json
import cv2
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent
SEVERITY_MODULE_DIR = PROJECT_ROOT / "severity-module"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SEVERITY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SEVERITY_MODULE_DIR))

from integration.currency_pipeline import currency_pipeline
from integration.sakshi_interface import get_active_damage_labels, run_sakshi_assessment, get_sakshi_handoff_payload
from interface import assess_damage

test_images_dir = SEVERITY_MODULE_DIR / "test_images"

print("=" * 70)
print("FINAL END-TO-END INTEGRATION TEST — CURRENCY NOTE ASSESSMENT SYSTEM")
print("=" * 70)

# ------------------------------------------------------------
# 1. FULL PIPELINE TEST ON REAL TEST IMAGES
# ------------------------------------------------------------
test_cases = [
    ("TORN PIPELINE", test_images_dir / "img_3.jpeg"),
    ("FOLDED PIPELINE", test_images_dir / "img_2.jpeg"),
    ("NORMAL PIPELINE", test_images_dir / "sample_note.jpeg"),
    ("STAIN PIPELINE", test_images_dir / "Ink_Stain.jpeg" if (test_images_dir / "Ink_Stain.jpeg").exists() else test_images_dir / "Stain.jpeg"),
]

pipeline_results = {}
all_passed = True

for test_name, img_path in test_cases:
    print(f"\n==================================================")
    print(f"TEST: {test_name} ({img_path.name})")
    print(f"==================================================")
    
    try:
        pipeline_res = currency_pipeline.analyze(str(img_path))
        
        yash_denom = pipeline_res.get("denomination", {})
        vinit_damage = pipeline_res.get("damage", {})
        sakshi_active = pipeline_res.get("sakshi_active_damages", [])
        severity_res = pipeline_res.get("severity_assessment", {})
        severity_err = pipeline_res.get("severity_error")
        
        denom_val = yash_denom.get("value", "Unknown")
        denom_conf = yash_denom.get("confidence", 0.0)
        
        vinit_labels = vinit_damage.get("predicted_labels", [])
        
        if severity_res is None:
            damage_pct = 0.0
            damaged_px = 0
            total_px = 0
            severity = "Unknown"
            rec = "Unknown"
            annotated_path = "N/A"
            status = "FAIL"
            all_passed = False
        else:
            damage_pct = severity_res.get("damage_percentage", 0.0)
            damaged_px = severity_res.get("damaged_pixels", 0)
            total_px = severity_res.get("total_note_pixels", 0)
            severity = severity_res.get("severity", "None")
            rec = severity_res.get("recommendation", "N/A")
            viz = severity_res.get("visualization", {})
            annotated_path = viz.get("annotated_path", "N/A")
            status = "PASS" if severity_err is None else "FAIL"
            if status == "FAIL":
                all_passed = False

        pipeline_results[test_name] = {
            "status": status,
            "yash_denomination": denom_val,
            "yash_confidence": denom_conf,
            "vinit_labels": vinit_labels,
            "sakshi_active": sakshi_active,
            "damaged_pixels": damaged_px,
            "total_pixels": total_px,
            "damage_pct": damage_pct,
            "severity": severity,
            "recommendation": rec,
            "annotated_path": annotated_path,
        }

        print(f"IMAGE: {img_path.name}")
        print(f"YASH DENOMINATION: {denom_val}")
        print(f"YASH CONFIDENCE: {denom_conf}")
        print(f"VINIT DAMAGE LABEL(S): {vinit_labels}")
        print(f"SAKSHI ACTIVE SEGMENTATION(S): {sakshi_active}")
        print(f"DAMAGED PIXELS: {damaged_px}")
        print(f"TOTAL/EXPECTED NOTE PIXELS: {total_px}")
        print(f"ESTIMATED DAMAGE %: {damage_pct:.2f}%")
        print(f"SEVERITY: {severity}")
        print(f"RECOMMENDATION: {rec}")
        print(f"ANNOTATED OUTPUT PATH: {annotated_path}")
        print(f"PIPELINE STATUS: {status}")

    except Exception as e:
        print(f"Pipeline error for {img_path.name}: {e}")
        traceback.print_exc()
        pipeline_results[test_name] = {
            "status": "FAIL",
            "yash_denomination": "Error",
            "yash_confidence": 0.0,
            "vinit_labels": [],
            "sakshi_active": [],
            "damaged_pixels": 0,
            "total_pixels": 0,
            "damage_pct": 0.0,
            "severity": "Error",
            "recommendation": "Error",
            "annotated_path": "N/A",
        }
        all_passed = False

# ------------------------------------------------------------
# 2. AUTOMATIC HANDOFF VERIFICATION (Yash -> Vinit -> Sakshi)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("VERIFYING HANDOFF INVARIANTS")
print("=" * 70)

handoff_tests_passed = True

# Test 2.1: Yash Denomination handoff
yash_mock_result = {"denomination": "10", "confidence": 0.98, "accepted": True, "available": True}
vinit_mock_torn = {"final_damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0}}

assessment = run_sakshi_assessment(
    image_input=str(test_images_dir / "img_3.jpeg"),
    denomination_result=yash_mock_result,
    damage_result=vinit_mock_torn
)
res = assessment.get("result", {})
denom_check = (res.get("denomination") == "10")
label_check = (res.get("damage_types") == ["Torn"])
pct_check = (res.get("damage_percentage") > 0.0) # Comes from mask, not confidence

print(f"Handoff Test 1 (Yash Denom '10' -> Sakshi): {'PASS' if denom_check else 'FAIL'}")
print(f"Handoff Test 2 (Vinit Label 'Torn' -> Sakshi Auto Segment): {'PASS' if label_check else 'FAIL'}")
print(f"Handoff Test 3 (Sakshi Damage % from Mask = {res.get('damage_percentage'):.2f}%): {'PASS' if pct_check else 'FAIL'}")

if not (denom_check and label_check and pct_check):
    handoff_tests_passed = False

# ------------------------------------------------------------
# 3. MULTI-DAMAGE HANDOFF & UNION TEST
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("VERIFYING MULTI-DAMAGE HANDOFF & UNION")
print("=" * 70)

vinit_mock_multi = {"final_damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 1, "Normal": 0}}
active_multi = get_active_damage_labels(vinit_mock_multi)
multi_labels_ok = (active_multi == ["Torn", "Stain"])

assessment_multi = run_sakshi_assessment(
    image_input=str(test_images_dir / "img_3.jpeg"),
    denomination_result=yash_mock_result,
    damage_result=vinit_mock_multi
)
res_multi = assessment_multi.get("result", {})
multi_damaged_px = res_multi.get("damaged_pixels", 0)

# Run individual segmentations
res_torn = run_sakshi_assessment(
    image_input=str(test_images_dir / "img_3.jpeg"),
    denomination_result=yash_mock_result,
    damage_result={"final_damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0}}
).get("result", {})

res_stain = run_sakshi_assessment(
    image_input=str(test_images_dir / "img_3.jpeg"),
    denomination_result=yash_mock_result,
    damage_result={"final_damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 1, "Normal": 0}}
).get("result", {})

torn_px = res_torn.get("damaged_pixels", 0)
stain_px = res_stain.get("damaged_pixels", 0)
sum_px = torn_px + stain_px

union_check = (multi_damaged_px <= sum_px)

print(f"Vinit Multi-label extraction: {active_multi} -> {'PASS' if multi_labels_ok else 'FAIL'}")
print(f"Combined Damaged Pixels ({multi_damaged_px}) <= Sum ({sum_px}) -> {'PASS' if union_check else 'FAIL'}")

multi_damage_passed = multi_labels_ok and union_check

# ------------------------------------------------------------
# 4. FINAL OUTPUT CONSISTENCY
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("VERIFYING OUTPUT CONSISTENCY")
print("=" * 70)

res_single = res_torn
dict_pct = res_single.get("damage_percentage")
sev_str = res_single.get("severity")
rec_str = res_single.get("recommendation")
viz = res_single.get("visualization", {})
annotated_file = viz.get("annotated_path", "")

consistency_passed = True
if not (dict_pct is not None and sev_str and rec_str and annotated_file):
    consistency_passed = False

print(f"Dictionary damage_percentage: {dict_pct:.2f}%")
print(f"Severity: {sev_str}")
print(f"Recommendation: {rec_str}")
print(f"Visualization path: {annotated_file}")
print(f"Final Output Consistency Check: {'PASS' if consistency_passed else 'FAIL'}")

# ------------------------------------------------------------
# 5. FAILURE SAFETY
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("VERIFYING FAILURE SAFETY")
print("=" * 70)

failure_safety_passed = True
try:
    # Optional outputs missing / empty labels / background denomination
    safe_res_1 = run_sakshi_assessment(
        image_input=str(test_images_dir / "sample_note.jpeg"),
        denomination_result={},
        damage_result={}
    )
    safe_res_2 = run_sakshi_assessment(
        image_input=str(test_images_dir / "sample_note.jpeg"),
        denomination_result={"is_background": True},
        damage_result={"final_damage_labels": {"Normal": 1}}
    )
    if safe_res_1.get("result") is None or safe_res_2.get("result") is None:
        failure_safety_passed = False
except Exception as e:
    print(f"Failure safety test exception: {e}")
    failure_safety_passed = False

print(f"Failure Safety Check: {'PASS' if failure_safety_passed else 'FAIL'}")

# ------------------------------------------------------------
# SUMMARY PRINT FOR EACH IMAGE AS REQUIRED BY PROMPT ITEM 7
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY RESULTS FOR PROMPT COMPLIANCE")
print("=" * 70)

for test_name, img_path in test_cases:
    data = pipeline_results[test_name]
    print(f"\nIMAGE: {img_path.name}")
    print(f"YASH DENOMINATION: {data['yash_denomination']}")
    print(f"YASH CONFIDENCE: {data['yash_confidence']}")
    print(f"VINIT DAMAGE LABEL(S): {data['vinit_labels']}")
    print(f"SAKSHI ACTIVE SEGMENTATION(S): {data['sakshi_active']}")
    print(f"DAMAGED PIXELS: {data['damaged_pixels']}")
    print(f"TOTAL/EXPECTED NOTE PIXELS: {data['total_pixels']}")
    print(f"ESTIMATED DAMAGE %: {data['damage_pct']:.2f}%")
    print(f"SEVERITY: {data['severity']}")
    print(f"RECOMMENDATION: {data['recommendation']}")
    print(f"ANNOTATED OUTPUT PATH: {data['annotated_path']}")
    print(f"PIPELINE STATUS: {data['status']}")

