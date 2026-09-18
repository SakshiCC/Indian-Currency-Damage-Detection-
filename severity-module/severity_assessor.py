"""
severity_assessor.py
============================================================
Main Controller for Sakshi's Damage Severity Assessment Module.

Orchestrates:
  1. Image & Rectified Note preprocessing
  2. Note Mask generation (visible vs expected note geometry)
  3. Damage-specific segmentation
  4. Per-damage & Combined Union mask creation
  5. Strict expected note mask clipping
  6. Region analysis (contours, bbox, centroid, location grid)
  7. Damaged pixel count & 2-decimal percentage calculation
  8. Project-level severity assessment
  9. Guidance recommendation generation
  10. Visualization overlay & debug mask saving
============================================================
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import cv2
import numpy as np

from segmentation import segment_damage
from region_analysis import analyze_damage_regions
from severity import calculate_severity
from recommendation import generate_recommendation
from visualization import (
    create_annotated_image,
    save_visualization_outputs,
    save_debug_outputs,
)


class SeverityAssessor:
    """Controller orchestrating Sakshi's damage severity assessment workflow."""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def assess(
        self,
        image_input: Union[str, Path, np.ndarray],
        damage_types: List[str],
        denomination: Optional[str] = None,
        denomination_confidence: Optional[float] = None,
        preprocessing_metadata: Optional[Dict[str, Any]] = None,
        debug: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Runs complete severity assessment on banknote image input.
        """
        is_debug = self.debug if debug is None else debug

        # Load image array
        if isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            image = cv2.imread(str(image_input))

        if image is None:
            raise ValueError(f"Could not load image input: {image_input}")

        # Check if rectified / cropped note is available in preprocessing_metadata
        prep_meta = preprocessing_metadata or {}
        rectified_note = prep_meta.get("rectified_note")
        if isinstance(rectified_note, np.ndarray) and rectified_note.size > 0:
            work_image = rectified_note.copy()
        else:
            work_image = image.copy()

        # ============================================================
        # 1. PRESERVE VINIT PRIMARY PREDICTED LABELS
        # ============================================================
        vinit_predicted_labels = list(damage_types) if damage_types else ["Normal"]

        # Normal exclusivity: if physical damage is present, filter Normal from candidate set
        vinit_candidates = [
            d for d in vinit_predicted_labels
            if d in ["Torn", "Folded", "Stain", "Burnt"]
        ]

        # ============================================================
        # 2. SAKSHI CONSERVATIVE DOUBLE-CHECK / VERIFICATION
        # ============================================================
        all_physical_categories = ["Torn", "Folded", "Stain", "Burnt"]
        sakshi_verified_labels: List[str] = []
        sakshi_added_labels: List[str] = []
        unverified_vinit_labels: List[str] = []

        from config import MIN_VERIFICATION_PCT, MIN_RECOVERY_PCT

        # Probe segmentation evidence for physical damage categories using frozen segment_damage
        probe_res = segment_damage(
            image=work_image,
            damage_labels=all_physical_categories,
            denomination=denomination,
            denomination_confidence=denomination_confidence,
            preprocessing_metadata=prep_meta,
        )

        geometry = probe_res["geometry"]
        probe_masks = probe_res["damage_masks"]
        expected_pixels = float(np.count_nonzero(geometry.expected_mask)) if geometry.expected_mask is not None else 1.0
        if expected_pixels <= 0:
            expected_pixels = float(work_image.shape[0] * work_image.shape[1])

        for cat in all_physical_categories:
            cat_mask = probe_masks.get(cat)
            cat_px = np.count_nonzero(cat_mask) if cat_mask is not None else 0
            cat_pct = (cat_px / expected_pixels) * 100.0

            ver_thresh = MIN_VERIFICATION_PCT.get(cat, 0.5) if isinstance(MIN_VERIFICATION_PCT, dict) else MIN_VERIFICATION_PCT
            rec_thresh = MIN_RECOVERY_PCT.get(cat, 1.0) if isinstance(MIN_RECOVERY_PCT, dict) else 1.0

            if cat in vinit_candidates:
                if cat_pct >= ver_thresh:
                    sakshi_verified_labels.append(cat)
                else:
                    unverified_vinit_labels.append(cat)
            else:
                if cat_pct >= rec_thresh:
                    sakshi_added_labels.append(cat)

        # ============================================================
        # 3. DETERMINE FINAL VERIFIED DAMAGE LABELS
        # ============================================================
        final_damage_labels = list(dict.fromkeys(sakshi_verified_labels + sakshi_added_labels))
        if not final_damage_labels:
            final_damage_labels = ["Normal"]

        # Run final damage-specific segmentation using final_damage_labels
        segmentation_res = segment_damage(
            image=work_image,
            damage_labels=final_damage_labels,
            denomination=denomination,
            denomination_confidence=denomination_confidence,
            preprocessing_metadata=prep_meta,
        )

        geometry = segmentation_res["geometry"]
        damage_masks = segmentation_res["damage_masks"]
        combined_mask = segmentation_res["combined_mask"]

        # Note mask: expected complete banknote geometry
        note_mask = geometry.expected_mask

        # Calculate severity metrics
        severity_res = calculate_severity(
            damage_masks=damage_masks,
            combined_mask=combined_mask,
            expected_note_mask=note_mask,
        )

        combined_metrics = severity_res["combined"]
        per_damage_metrics = severity_res["per_damage"]

        # Extract regions per active damage type
        all_regions: List[Dict[str, Any]] = []
        for dtype, dmask in damage_masks.items():
            r_list = analyze_damage_regions(
                damage_mask=dmask,
                note_mask=note_mask,
                damage_type=dtype,
            )
            all_regions.extend(r_list)

        # Recommendation text
        rec_text = generate_recommendation(
            severity=combined_metrics["severity"],
            active_damage_labels=final_damage_labels,
            damage_results=severity_res,
        )

        # Generate annotated output visualization
        annotated_img = create_annotated_image(
            image=work_image,
            regions=all_regions,
            combined_mask=combined_mask,
            denomination=denomination,
            damage_types=final_damage_labels,
            damage_percentage=combined_metrics["damage_percentage"],
            severity=combined_metrics["severity"],
            recommendation=rec_text,
        )

        # Save main output files
        vis_paths = save_visualization_outputs(
            image=work_image,
            combined_mask=combined_mask,
            annotated_image=annotated_img,
        )

        # Save intermediate debug files if debug=True
        debug_paths = {}
        if is_debug:
            debug_paths = save_debug_outputs(
                visible_note_mask=geometry.visible_mask,
                expected_note_mask=geometry.expected_mask,
                damage_masks=damage_masks,
                combined_mask=combined_mask,
            )

        denom_str = str(denomination) if denomination else "Unknown"

        return {
            "denomination": denom_str,
            "currency": denom_str,
            "denomination_confidence": denomination_confidence,
            "vinit_predicted_labels": vinit_predicted_labels,
            "sakshi_verified_labels": sakshi_verified_labels,
            "sakshi_added_labels": sakshi_added_labels,
            "unverified_vinit_labels": unverified_vinit_labels,
            "final_damage_labels": final_damage_labels,
            "damage_types": final_damage_labels,
            "damaged_pixels": combined_metrics["damaged_pixels"],
            "total_note_pixels": combined_metrics["total_note_pixels"],
            "damage_percentage": combined_metrics["damage_percentage"],
            "segmentation": {
                "damaged_pixels": combined_metrics["damaged_pixels"],
                "total_note_pixels": combined_metrics["total_note_pixels"],
                "damage_percentage": combined_metrics["damage_percentage"],
            },
            "per_damage": {
                dtype: {
                    "damaged_pixels": metrics["damaged_pixels"],
                    "damage_percentage": metrics["damage_percentage"],
                    "area_percentage": metrics["damage_percentage"],
                    "severity": metrics["severity"],
                }
                for dtype, metrics in per_damage_metrics.items()
            },
            "regions": all_regions,
            "severity": combined_metrics["severity"],
            "recommendation": rec_text,
            "visualization": vis_paths,
            "debug_paths": debug_paths,
            "geometry_score": round(geometry.geometry_score, 3),
            "orientation_degrees": round(geometry.orientation_degrees, 2),
            "expected_aspect_ratio": round(geometry.expected_aspect_ratio, 4),
            "denomination_prior_used": geometry.denomination_prior_used,
            "raw": {
                "geometry_score": geometry.geometry_score,
                "orientation_degrees": geometry.orientation_degrees,
                "denomination_prior_used": geometry.denomination_prior_used,
            },
        }


_DEFAULT_ASSESSOR: Optional[SeverityAssessor] = None


def get_severity_assessor(debug: bool = False) -> SeverityAssessor:
    global _DEFAULT_ASSESSOR
    if _DEFAULT_ASSESSOR is None:
        _DEFAULT_ASSESSOR = SeverityAssessor(debug=debug)
    return _DEFAULT_ASSESSOR
