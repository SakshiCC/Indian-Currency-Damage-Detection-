import sys
from pathlib import Path
import argparse
import cv2
import numpy as np

# Ensure project root and severity-module are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SEVERITY_MODULE_DIR = PROJECT_ROOT / "severity-module"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SEVERITY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SEVERITY_MODULE_DIR))

from integration.currency_pipeline import currency_pipeline
from note_geometry import estimate_note_geometry
from segmentation import segment_torn

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Currency Note Damage Assessment System"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to input currency note image",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed internal debugging diagnostics",
    )
    return parser.parse_args()


def get_image_path(raw_arg: str = None) -> Path:
    if not raw_arg:
        print()
        raw_input = input("Enter currency note image path: ").strip()
    else:
        raw_input = raw_arg.strip()

    # Strip surrounding quotes if user pasted Windows path with quotes
    cleaned = raw_input.strip("\"'")
    img_path = Path(cleaned)

    if not img_path.exists():
        print(f"\nERROR: File does not exist: {cleaned}")
        sys.exit(1)

    if img_path.suffix.lower() not in VALID_EXTENSIONS:
        print(
            f"\nERROR: Unsupported image format '{img_path.suffix}'. "
            f"Allowed formats: {', '.join(sorted(VALID_EXTENSIONS))}"
        )
        sys.exit(1)

    return img_path.resolve()


def main():
    args = parse_args()
    img_path = get_image_path(args.image_path)

    # Run integrated assessment pipeline
    analysis_res = currency_pipeline.analyze(str(img_path))

    if not analysis_res.get("success", False):
        print(f"\nERROR: Pipeline execution failed.")
        sys.exit(1)

    # 1. Denomination Recognition Output
    denom_data = analysis_res.get("denomination", {})
    denom_val = denom_data.get("value") or denom_data.get("denomination") or "Unknown"
    denom_str = f"Rs. {denom_val}" if str(denom_val).lower() not in {"unknown", "none", "background"} else "Unknown"
    denom_conf = denom_data.get("confidence_percent", 0.0)

    # 2. Damage Classification & Verification Output
    severity_res = analysis_res.get("severity_assessment", {})
    vinit_data = analysis_res.get("damage", {})

    if severity_res is None:
        print(f"\nERROR: Damage severity assessment failed: {analysis_res.get('severity_error')}")
        sys.exit(1)

    final_damage_labels = severity_res.get("final_damage_labels", severity_res.get("damage_types", ["Normal"]))
    
    # Ensure Normal exclusivity for user display
    if "Normal" in final_damage_labels and len(final_damage_labels) > 1:
        final_damage_labels = [l for l in final_damage_labels if l != "Normal"]
    if not final_damage_labels:
        final_damage_labels = ["Normal"]

    damaged_px = severity_res.get("damaged_pixels", 0)
    total_px = severity_res.get("total_note_pixels", 0)
    damage_pct = severity_res.get("damage_percentage", 0.0)
    severity_label = severity_res.get("severity", "None")
    recommendation_text = severity_res.get("recommendation", "No significant physical damage detected.")
    geometry_score = severity_res.get("geometry_score", 0.0)

    # 3. Create Unique Output Paths in output/ directory
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    unique_result_path = output_dir / f"{stem}_assessment.jpg"
    unique_mask_path = output_dir / f"{stem}_damage_mask.png"

    # Get annotated image and save unique outputs
    viz_info = severity_res.get("visualization", {})
    annotated_source_path = viz_info.get("annotated_path")
    mask_source_path = viz_info.get("mask_path")

    if annotated_source_path and Path(annotated_source_path).exists():
        ann_img = cv2.imread(annotated_source_path)
        if ann_img is not None:
            cv2.imwrite(str(unique_result_path), ann_img)

    if mask_source_path and Path(mask_source_path).exists():
        mask_img = cv2.imread(mask_source_path)
        if mask_img is not None:
            cv2.imwrite(str(unique_mask_path), mask_img)

    # 4. Standard Clean Integrated User Terminal Output
    damage_types_display = ", ".join(final_damage_labels) if isinstance(final_damage_labels, list) else str(final_damage_labels)

    print()
    print("=" * 64)
    print("              CURRENCY NOTE DAMAGE ASSESSMENT")
    print("=" * 64)
    print()
    print(f"Input Image           : {img_path.name}")
    print()
    print(f"Denomination          : {denom_str}")
    print(f"Confidence            : {denom_conf:.2f}%")
    print()
    print(f"Detected Damage Types : {damage_types_display}")
    print()
    print(f"Damaged Pixels        : {damaged_px}")
    print(f"Estimated Note Pixels : {total_px}")
    print(f"Estimated Damage Area : {damage_pct:.2f}%")
    print()
    print(f"Severity              : {severity_label}")
    print(f"Recommendation        : {recommendation_text}")
    print()
    print(f"Geometry Confidence   : {geometry_score:.3f}")
    print()
    print("=" * 64)
    print()

    # 5. Optional Debug Mode Diagnostics (--debug)
    if args.debug:
        print("=" * 64)
        print("                  INTERNAL DEBUG DIAGNOSTICS")
        print("=" * 64)
        print(f"Vinit Primary Labels  : {vinit_data.get('predicted_labels', [])}")
        print(f"Classification Status : {vinit_data.get('classification_status', 'N/A')}")
        print(f"Vinit Evidence Scores : {vinit_data.get('evidence_scores', {})}")
        print(f"Sakshi Verified       : {severity_res.get('sakshi_verified_labels', [])}")
        print(f"Sakshi Added          : {severity_res.get('sakshi_added_labels', [])}")
        print(f"Unverified Labels     : {severity_res.get('unverified_vinit_labels', [])}")
        print(f"Background Flag       : {denom_data.get('is_background', False)}")
        print(f"Prior Used            : {severity_res.get('denomination_prior_used', False)}")

        # Load raw geometry metrics for debug
        raw_img = cv2.imread(str(img_path))
        if raw_img is not None:
            geom = estimate_note_geometry(raw_img, denomination=denom_val)
            exp_mask = geom.expected_mask
            vis_mask = geom.visible_mask
            exp_px = int(np.count_nonzero(exp_mask)) if exp_mask is not None else 0
            vis_px = int(np.count_nonzero(vis_mask)) if vis_mask is not None else 0

            raw_missing = cv2.subtract(exp_mask, cv2.bitwise_and(vis_mask, exp_mask)) if (exp_mask is not None and vis_mask is not None) else None
            raw_torn_px = int(np.count_nonzero(raw_missing)) if raw_missing is not None else 0

            torn_m = segment_torn(raw_img, geom)
            filt_torn_px = int(np.count_nonzero(torn_m))
            outside_m = cv2.bitwise_and(torn_m, cv2.bitwise_not(exp_mask)) if (torn_m is not None and exp_mask is not None) else None
            outside_px = int(np.count_nonzero(outside_m)) if outside_m is not None else 0

            prior_used = severity_res.get("denomination_prior_used", False)
            recon_method = "Aspect-Ratio Prior + Convex Hull" if prior_used else "Convex Hull Polygon"

            print(f"Expected Note Pixels  : {exp_px}")
            print(f"Visible Note Pixels   : {vis_px}")
            print(f"Raw Torn Pixels       : {raw_torn_px}")
            print(f"Filtered Torn Pixels  : {filt_torn_px}")
            print(f"Final Torn Pixels     : {damaged_px if 'Torn' in final_damage_labels else 0}")
            print(f"Torn Outside Expected : {outside_px}")
            print(f"Reconstruction Method : {recon_method}")
        print("=" * 64)
        print()


if __name__ == "__main__":
    main()
