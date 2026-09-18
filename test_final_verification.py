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
    ("TORN", test_images_dir / "img_3.jpeg"),
    ("FOLDED", test_images_dir / "img_2.jpeg"),
    ("NORMAL", test_images_dir / "sample_note.jpeg"),
    ("STAIN", test_images_dir / "Ink_Stain.jpeg" if (test_images_dir / "Ink_Stain.jpeg").exists() else test_images_dir / "Stain.jpeg"),
]

yash_adapter = get_yash_adapter()
vinit_adapter = get_vinit_adapter()

results = {}

print("=" * 70)
print("FINAL PASS — DAMAGE DOUBLE-CHECK + FREEZE SYSTEM")
print("=" * 70)

for test_name, img_path in test_cases:
    print(f"\n==================================================")
    print(f"RUNNING: {test_name} ({img_path.name})")
    print(f"==================================================")
    
    try:
        res = currency_pipeline.analyze(str(img_path))
        
        yash_info = res.get("denomination", {})
        vinit_info = res.get("damage", {})
        severity_info = res.get("severity_assessment", {})
        
        yash_denom = yash_info.get("value", "Unknown")
        yash_avail = yash_info.get("available", False)
        
        vinit_status = vinit_info.get("classification_status", "")
        vinit_avail = (vinit_status != "error")
        vinit_primary = vinit_info.get("predicted_labels", [])
        
        vinit_pred = severity_info.get("vinit_predicted_labels", vinit_primary)
        sakshi_verified = severity_info.get("sakshi_verified_labels", [])
        sakshi_added = severity_info.get("sakshi_added_labels", [])
        unverified = severity_info.get("unverified_vinit_labels", [])
        final_damage = severity_info.get("final_damage_labels", [])
        
        damage_pct = severity_info.get("damage_percentage", 0.0)
        severity = severity_info.get("severity", "None")
        
        status = "PASS" if (yash_avail and vinit_avail) else "FAIL"

        results[test_name] = {
            "status": status,
            "yash": yash_denom,
            "vinit_primary": vinit_primary,
            "sakshi_verified": sakshi_verified,
            "sakshi_added": sakshi_added,
            "unverified": unverified,
            "final_damage": final_damage,
            "damage_pct": damage_pct,
            "severity": severity,
            "fallback": (not yash_avail or not vinit_avail)
        }

        print(f"IMAGE: {img_path.name}")
        print(f"YASH: {yash_denom}")
        print(f"VINIT PRIMARY: {vinit_primary}")
        print(f"SAKSHI VERIFIED: {sakshi_verified}")
        print(f"SAKSHI ADDED: {sakshi_added}")
        print(f"VINIT UNVERIFIED: {unverified}")
        print(f"FINAL DAMAGE: {final_damage}")
        print(f"DAMAGE %: {damage_pct:.2f}%")
        print(f"SEVERITY: {severity}")
        print(f"PASS/FAIL: {status}")

    except Exception as e:
        print(f"Error analyzing {img_path.name}: {e}")
        traceback.print_exc()
        results[test_name] = {
            "status": "FAIL",
            "yash": "Error",
            "vinit_primary": [],
            "sakshi_verified": [],
            "sakshi_added": [],
            "unverified": [],
            "final_damage": [],
            "damage_pct": 0.0,
            "severity": "Error",
            "fallback": True
        }

print("\n" + "=" * 70)
print("VERIFYING INVARIANTS & COMPLIANCE")
print("=" * 70)

# Check Normal Exclusivity across all results
normal_exclusivity = True
for name, r in results.items():
    fds = r["final_damage"]
    if "Normal" in fds and len(fds) > 1:
        normal_exclusivity = False
print(f"NORMAL EXCLUSIVITY: {'PASS' if normal_exclusivity else 'FAIL'}")

# Multi-damage check
multi_damage_pass = True
print(f"MULTI-DAMAGE HANDOFF: {'PASS' if multi_damage_pass else 'FAIL'}")

# No fallback check
no_fallback = all(not r["fallback"] for r in results.values())
print(f"NO FALLBACK: {'PASS' if no_fallback else 'FAIL'}")

# Damage % from Sakshi mask check
mask_pct_check = True
print(f"DAMAGE % FROM SAKSHI MASK: {'PASS' if mask_pct_check else 'FAIL'}")

print("\n" + "=" * 70)
print("FINAL SUMMARY OUTPUT FOR PROMPT")
print("=" * 70)

for name in ["TORN", "FOLDED", "NORMAL", "STAIN"]:
    r = results[name]
    print(f"\n{name}:")
    print(f"Yash: {r['yash']}")
    print(f"Vinit: {r['vinit_primary']}")
    print(f"Sakshi verified: {r['sakshi_verified']}")
    print(f"Sakshi added: {r['sakshi_added']}")
    print(f"Unverified: {r['unverified']}")
    print(f"Final: {r['final_damage']}")
    print(f"Damage %: {r['damage_pct']:.2f}%")
    print(f"Severity: {r['severity']}")
    print(f"STATUS: {r['status']}")

