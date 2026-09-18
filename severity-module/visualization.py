"""
visualization.py
============================================================
Visualization & Output Generation Module for Sakshi's Severity Module.

Generates uncluttered annotated result images with damage contours,
bounding boxes, metadata text overlays, and debug mask saving.
============================================================
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import cv2
import numpy as np

from config import OUTPUT_DIR, DEBUG_DIR

DAMAGE_COLORS = {
    "Torn": (0, 0, 255),       # Red
    "Stain": (0, 165, 255),    # Orange
    "Burnt": (0, 0, 128),      # Dark Red / Maroon
    "Folded": (255, 0, 255),   # Magenta
    "Combined": (0, 0, 255),   # Red
}


def create_annotated_image(
    image: np.ndarray,
    regions: List[Dict[str, Any]],
    combined_mask: np.ndarray,
    denomination: Optional[str],
    damage_types: List[str],
    damage_percentage: float,
    severity: str,
    recommendation: str,
    max_regions_to_draw: int = 6,
) -> np.ndarray:
    """
    Draws translucent damage mask overlay using contour outlines (not bounding boxes),
    and a clean summary banner.  Detailed region info is kept in JSON only.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Draw transparent damage mask overlay + contour outlines per damage type
    damage_masks_by_type: Dict[str, np.ndarray] = {}
    for region in regions:
        dtype = region.get("damage_type", "Torn")
        # Build a per-type mask from regions' contour areas
        if dtype not in damage_masks_by_type:
            damage_masks_by_type[dtype] = np.zeros((h, w), dtype=np.uint8)

    # Use combined_mask for the overlay tint
    if combined_mask is not None and cv2.countNonZero(combined_mask) > 0:
        red_overlay = np.zeros_like(annotated)
        red_overlay[:, :] = (0, 0, 220)  # Red tint
        mask_indices = combined_mask > 0
        annotated[mask_indices] = cv2.addWeighted(
            annotated, 0.60, red_overlay, 0.40, 0
        )[mask_indices]

        # Draw damage contour outlines (no boxes)
        contours_found, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours_found, -1, (0, 0, 255), 2)

    # Draw a single per-damage-type summary label (top-left corner stacked)
    # Use authoritative damage_percentage — NOT summed region area_percentages
    unique_damage_types = list(dict.fromkeys(
        r.get("damage_type", "Torn") for r in regions
    )) if regions else damage_types

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.65, w / 900.0))
    label_y = 28
    for dtype in unique_damage_types:
        color = DAMAGE_COLORS.get(dtype, (0, 0, 255))
        # Use the authoritative combined damage_percentage for single-damage results,
        # or show damage_percentage directly (same value as banner line 2)
        label_text = f"{dtype} Damage: {damage_percentage:.2f}%"
        # Shadow
        cv2.putText(annotated, label_text, (11, label_y + 1),
                    font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        # Foreground
        cv2.putText(annotated, label_text, (10, label_y),
                    font, font_scale, color, 1, cv2.LINE_AA)
        label_y += int(font_scale * 40) + 6

    # Draw summary banner header box on bottom of image
    banner_height = max(90, int(h * 0.20))
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)

    denom_str = f"Rs.{denomination}" if denomination else "Unknown"
    damages_str = ", ".join(damage_types) if damage_types else "None"

    line1 = f"Currency: {denom_str}  |  Damage Type: {damages_str}"
    line2 = f"Damage Area: {damage_percentage:.2f}%  |  Severity: {severity}"
    line3 = f"Recommendation: {recommendation}"

    scale = max(0.42, min(0.62, w / 950.0))
    thickness = 1

    cv2.putText(banner, line1, (15, 26), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(banner, line2, (15, 54), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(banner, line3, (15, 82), font, scale, (200, 200, 250), thickness, cv2.LINE_AA)

    final_result = np.vstack([annotated, banner])
    return final_result


def save_visualization_outputs(
    image: np.ndarray,
    combined_mask: np.ndarray,
    annotated_image: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, str]:
    """Saves main output files: corrected image, damage mask, annotated result."""
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_path = output_dir / "original_or_corrected.jpg"
    mask_path = output_dir / "damage_mask.png"
    annotated_path = output_dir / "annotated_result.jpg"

    cv2.imwrite(str(orig_path), image)
    if combined_mask is not None:
        cv2.imwrite(str(mask_path), combined_mask)
    else:
        cv2.imwrite(str(mask_path), np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
    cv2.imwrite(str(annotated_path), annotated_image)

    return {
        "original_path": str(orig_path),
        "mask_path": str(mask_path),
        "annotated_path": str(annotated_path),
    }


def save_debug_outputs(
    visible_note_mask: np.ndarray,
    expected_note_mask: np.ndarray,
    damage_masks: Dict[str, np.ndarray],
    combined_mask: np.ndarray,
    debug_dir: Path = DEBUG_DIR,
) -> Dict[str, str]:
    """
    Saves intermediate debug masks:
      - visible_note_mask.png
      - expected_note_mask.png
      - torn_mask.png
      - stain_mask.png
      - burnt_mask.png
      - folded_mask.png
      - combined_mask.png
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    if visible_note_mask is not None:
        p = debug_dir / "visible_note_mask.png"
        cv2.imwrite(str(p), visible_note_mask)
        paths["visible_note_mask"] = str(p)

    if expected_note_mask is not None:
        p = debug_dir / "expected_note_mask.png"
        cv2.imwrite(str(p), expected_note_mask)
        paths["expected_note_mask"] = str(p)

    for dtype, mask in damage_masks.items():
        if mask is not None:
            p = debug_dir / f"{dtype.lower()}_mask.png"
            cv2.imwrite(str(p), mask)
            paths[f"{dtype.lower()}_mask"] = str(p)

    if combined_mask is not None:
        p = debug_dir / "combined_mask.png"
        cv2.imwrite(str(p), combined_mask)
        paths["combined_mask"] = str(p)

    return paths