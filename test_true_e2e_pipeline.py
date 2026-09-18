import sys
from pathlib import Path
import json
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent
SEVERITY_MODULE_DIR = PROJECT_ROOT / "severity-module"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SEVERITY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SEVERITY_MODULE_DIR))

from integration.currency_pipeline import currency_pipeline
from integration.yash_adapter import get_yash_adapter
from integration.vinit_adapter import get_vinit_adapter

test_images_dir = SEVERITY_MODULE_DIR / "test_images"

test_cases = [
    ("TORN REAL PIPELINE", test_images_dir / "img_3.jpeg"),
    ("FOLDED REAL PIPELINE", test_images_dir / "img_2.jpeg"),
    ("NORMAL REAL PIPELINE", test_images_dir / "sample_note.jpeg"),
    ("STAIN REAL PIPELINE", test_images_dir / "Ink_Stain.jpeg" if (test_images_dir / "Ink_Stain.jpeg").exists() else test_images_dir / "Stain.jpeg"),
]

results = {}

print("=" * 70)
print("TRUE END-TO-END INTEGRATION TEST (REAL MODELS)")
print("=" * 70)

yash_adapter = get_yash_adapter()
vinit_adapter = get_vinit_adapter()

yash_loaded = yash_adapter.is_available
vinit_loaded = vinit_adapter.is_available

print(f"YASH MODEL LOADED GLOBALLY: {yash_loaded}")
print(f"VINIT MODEL LOADED GLOBALLY: {vinit_loaded}")

for test_name, img_path in test_cases:
    print(f"\n" + "=" * 50)
    print(f"TEST: {test_name} ({img_path.name})")
    print("=" * 50)
    
    try:
        res = currency_pipeline.analyze(str(img_path))
        
        yash_info = res.get("denomination", {})
        vinit_info = res.get("damage", {})
        sakshi_active = res.get("sakshi_active_damages", [])
        severity_info = res.get("severity_assessment", {})
        severity_err = res.get("severity_error")
        
        yash_denom = yash_info.get("value", "Unknown")
        yash_conf = yash_info.get("confidence", 0.0)
        yash_available = yash_info.get("available", False)
        
        vinit_predicted = vinit_info.get("predicted_labels", [])
        vinit_final = vinit_info.get("final_labels", {})
        vinit_status = vinit_info.get("classification_status", "")
        vinit_available = (vinit_status != "error")
        
        fallback_used = (not yash_available) or (not vinit_available)
        
        if severity_info is None:
            damage_pct = 0.0
            severity = "Unknown"
            rec = "Unknown"
            status = "FAIL"
        else:
            damage_pct = severity_info.get("damage_percentage", 0.0)
            severity = severity_info.get("severity", "None")
            rec = severity_info.get("recommendation", "N/A")
            status = "PASS" if (severity_err is None and not fallback_used) else "FAIL"

        results[test_name] = {
            "status": status,
            "yash_loaded": yash_available,
            "denomination": yash_denom,
            "yash_confidence": yash_conf,
            "vinit_loaded": vinit_available,
            "vinit_labels": vinit_predicted,
            "sakshi_active": sakshi_active,
            "damage_pct": damage_pct,
            "severity": severity,
            "recommendation": rec,
            "fallback_used": fallback_used,
        }

        print(f"IMAGE: {img_path.name}")
        print(f"YASH MODEL LOADED: {yash_available}")
        print(f"YASH DENOMINATION: {yash_denom}")
        print(f"YASH CONFIDENCE: {yash_conf:.4f}")
        print(f"VINIT MODEL LOADED: {vinit_available}")
        print(f"VINIT DAMAGE LABELS: {vinit_predicted}")
        print(f"SAKSHI ACTIVE SEGMENTATION: {sakshi_active}")
        print(f"SAKSHI DAMAGE %: {damage_pct:.2f}%")
        print(f"SEVERITY: {severity}")
        print(f"RECOMMENDATION: {rec}")
        print(f"FALLBACK USED: {fallback_used}")
        print(f"PIPELINE STATUS: {status}")

    except Exception as e:
        print(f"Pipeline crashed for {img_path.name}: {e}")
        traceback.print_exc()
        results[test_name] = {
            "status": "FAIL",
            "yash_loaded": False,
            "denomination": "Error",
            "yash_confidence": 0.0,
            "vinit_loaded": False,
            "vinit_labels": [],
            "sakshi_active": [],
            "damage_pct": 0.0,
            "severity": "Error",
            "recommendation": "Error",
            "fallback_used": True,
        }

print("\n" + "=" * 70)
print("FINAL SUMMARY REPORT FOR PROMPT COMPLIANCE")
print("=" * 70)
for k, v in results.items():
    print(f"\n{k}:")
    print(f"Status: {v['status']}")
    print(f"Denomination: {v['denomination']}")
    print(f"Vinit labels: {v['vinit_labels']}")
    print(f"Damage %: {v['damage_pct']:.2f}%")
    print(f"Severity: {v['severity']}")
    print(f"Fallback used: {v['fallback_used']}")
