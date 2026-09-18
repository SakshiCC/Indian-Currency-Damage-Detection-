import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SEVERITY_MODULE_DIR = PROJECT_ROOT / "severity-module"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SEVERITY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SEVERITY_MODULE_DIR))

from integration.yash_adapter import get_yash_adapter
from integration.vinit_adapter import get_vinit_adapter

print("=" * 60)
print("VERIFYING YASH ADAPTER MODEL LOADING")
print("=" * 60)

yash = get_yash_adapter()
print("Yash is_available:", yash.is_available)
print("Yash load_error:", yash.load_error)

test_img = SEVERITY_MODULE_DIR / "test_images" / "img_3.jpeg"
yash_res = yash.predict(str(test_img))
print("Yash prediction output:")
print(yash_res)

print("\n" + "=" * 60)
print("VERIFYING VINIT ADAPTER MODEL LOADING")
print("=" * 60)

vinit = get_vinit_adapter()
print("Vinit is_available:", vinit.is_available)

vinit_res = vinit.predict(str(test_img))
print("Vinit prediction output:")
print("Predicted labels:", vinit_res.get("predicted_labels"))
print("Final damage labels:", vinit_res.get("final_damage_labels"))
print("Evidence scores:", vinit_res.get("evidence_scores"))
print("Classification status:", vinit_res.get("classification_status"))
