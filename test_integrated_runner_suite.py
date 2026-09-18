import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent
SEVERITY_MODULE_DIR = PROJECT_ROOT / "severity-module"
test_images_dir = SEVERITY_MODULE_DIR / "test_images"

print("=" * 70)
print("TESTING INTEGRATED RUNNER SUITE")
print("=" * 70)

test_cases = [
    ("Torn", test_images_dir / "img_3.jpeg"),
    ("Folded", test_images_dir / "img_2.jpeg"),
    ("Normal", test_images_dir / "sample_note.jpeg"),
    ("Stain", test_images_dir / "Ink_Stain.jpeg"),
]

for name, path in test_cases:
    cmd = [sys.executable, "run.py", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    stdout = res.stdout
    print(f"\n--- Output for {name} ({path.name}) ---")
    print(stdout)
    
    # Assertions
    assert "Result Image" not in stdout, "Result Image path should NOT be printed in normal output!"
    assert "Detected Damage Types :" in stdout, "Detected Damage Types field missing!"
    if name == "Normal":
        assert "Detected Damage Types : Normal" in stdout, "Normal exclusivity violated!"

print("\n" + "=" * 70)
print("INTEGRATED RUNNER SUITE SUCCEEDED")
print("=" * 70)
