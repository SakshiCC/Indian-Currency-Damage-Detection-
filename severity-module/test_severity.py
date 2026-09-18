"""
test_severity.py
============================================================
Test runner for Sakshi's Damage Severity Assessment Module.

Usage:
  python severity-module/test_severity.py <image_path> <denomination> [confidence] [--damage DAMAGETYPE]

Examples:
  python severity-module/test_severity.py severity-module/test_images/sample_note.jpeg 500
  python severity-module/test_severity.py severity-module/test_images/img_2.jpeg 10
  python severity-module/test_severity.py severity-module/test_images/img_3.jpeg 10
============================================================
"""

import sys
from pathlib import Path

# Add severity-module directory to sys.path
SEVERITY_DIR = Path(__file__).resolve().parent
if str(SEVERITY_DIR) not in sys.path:
    sys.path.insert(0, str(SEVERITY_DIR))

from interface import assess_damage


def print_assessment(assessment: dict):
    res = assessment.get("result", assessment)
    print()
    print("=" * 65)
    print("SAKSHI - DAMAGE SEGMENTATION AND SEVERITY ASSESSMENT")
    print("=" * 65)
    print(f"Currency               : {res.get('denomination', 'Unknown')}")
    print(f"Damage Types           : {', '.join(res.get('damage_types', []))}")
    print(f"Damaged Pixels         : {res.get('damaged_pixels', 0)}")
    print(f"Total Note Pixels      : {res.get('total_note_pixels', 0)}")
    print(f"Damage %               : {res.get('damage_percentage', 0.0):.2f}%")
    print(f"Severity               : {res.get('severity', 'None')}")
    print(f"Recommendation         : {res.get('recommendation', '')}")

    regions = res.get("regions", [])
    if regions:
        print("-" * 65)
        print(f"DETECTED REGIONS ({len(regions)} total):")
        for i, reg in enumerate(regions, start=1):
            print(f"  [{i}] {reg['damage_type']} — {reg['area_percentage']:.2f}% | BBox: {reg['bounding_box']} | Centroid: {reg['centroid']} | Location: {reg['location']}")

    vis = res.get("visualization", {})
    if vis:
        print("-" * 65)
        print("VISUALIZATION OUTPUTS:")
        print(f"  Annotated Result     : {vis.get('annotated_path')}")
        print(f"  Damage Mask          : {vis.get('mask_path')}")

    print("=" * 65)
    print()


def main():
    if len(sys.argv) < 3:
        print("\nUsage: python severity-module/test_severity.py <image_path> <denomination> [confidence] [--damage DAMAGETYPE]\n")
        return

    image_path = Path(sys.argv[1])
    denomination = str(sys.argv[2])

    confidence = 0.95
    damage_labels = ["Torn"]

    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "--damage" and i + 1 < len(sys.argv):
            damage_labels = [d.strip() for d in sys.argv[i + 1].split(",")]
        elif not arg.startswith("--"):
            try:
                confidence = float(arg)
            except ValueError:
                pass

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\nRunning Sakshi Severity Test on: {image_path} (Denomination: Rs.{denomination}, Confidence: {confidence:.2f})\n")

    assessment = assess_damage(
        image_input=image_path,
        damage_labels=damage_labels,
        currency=denomination,
        denomination_confidence=confidence,
        preprocessing_metadata={"note_detected": True},
        debug=True,
    )

    print_assessment(assessment)


if __name__ == "__main__":
    main()