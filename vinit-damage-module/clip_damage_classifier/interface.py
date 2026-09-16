"""
interface.py
============================================================
Unified, Integration-Ready Damage Classification Interface.
Combines OpenCV note preprocessing (detection, cropping, CLAHE)
with contrastive CLIP ViT-B/32 prompt ensemble evidence scoring
and dynamic class-specific threshold calibration.

Supports:
    - V3.1: Dynamic calibration with NEW Normal rule: Normal = 1 iff Burnt=0 AND Torn=0
            and domain rule: Burnt=1 ==> Stain=1 (default)
    - V3: Dynamic calibration with legacy Normal exclusivity rule
    - V2: Fixed contrastive evidence thresholds (backward compatible)

Provides:
    from clip_damage_classifier import DamageClassifier

Owner: Vinit (Currency Note Image Processing + Damage Detection)
============================================================
"""

from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
from PIL import Image
import torch

from clip_damage_classifier import config
from clip_damage_classifier.clip_model import get_clip_wrapper
from clip_damage_classifier.scoring import compute_contrastive_scores
from clip_damage_classifier.thresholds import apply_evidence_thresholds
from clip_damage_classifier.preprocessing.image_preprocessing import (
    load_image,
    is_valid_image,
    apply_clahe,
    bgr_to_pil,
)
from clip_damage_classifier.preprocessing.note_detection import detect_and_crop_note
from clip_damage_classifier.calibration.context_features import extract_context_features
from clip_damage_classifier.calibration.dynamic_thresholds import (
    DynamicThresholdCalculator,
    DEFAULT_V2_BASE_THRESHOLDS,
)


class DamageClassifier:
    """
    Public integration-ready classifier for Indian Banknote Damage Analysis.
    Supports single-image and batch prediction with multi-label damage output.
    """

    def __init__(
        self,
        device: str = config.DEVICE_PREFERENCE,
        thresholds: Optional[Dict[str, float]] = None,
        use_preprocessing: bool = True,
        ambiguity_band: float = config.AMBIGUITY_BAND,
        strong_margin: float = config.STRONG_PREDICTION_MARGIN,
        calibration_version: str = "v3.1",
    ):
        self.device = device
        self.use_preprocessing = use_preprocessing
        self.ambiguity_band = ambiguity_band
        self.strong_margin = strong_margin
        self.calibration_version = calibration_version.lower()

        # Fixed thresholds used for V2 or as anchors
        self.thresholds = thresholds or dict(config.CALIBRATED_EVIDENCE_THRESHOLDS)

        # Dynamic threshold calculator for V3 / V3.1
        self.dynamic_calculator = DynamicThresholdCalculator(
            base_thresholds=self.thresholds
        )

        # Load CLIP model wrapper (singleton with pre-encoded prompt ensembles)
        self.clip_wrapper = get_clip_wrapper(
            model_name=config.CLIP_MODEL_NAME,
            device=self.device,
        )

    def preprocess_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]):
        """
        Loads, validates, detects note boundary, crops, and applies CLAHE.
        Returns the processed PIL Image for CLIP, the BGR array, and metadata dict.
        """
        raw_bgr = load_image(image_input)
        orig_shape = raw_bgr.shape[:2]

        if not self.use_preprocessing:
            return bgr_to_pil(raw_bgr), raw_bgr, {
                "note_detected": False,
                "original_shape": orig_shape,
                "cropped_shape": orig_shape,
                "clahe_applied": False,
                "area_ratio": 1.0,
            }

        # Note contour detection and cropping
        cropped_bgr, detected = detect_and_crop_note(raw_bgr, apply_perspective_correction=True)
        crop_shape = cropped_bgr.shape[:2]

        # Area ratio
        orig_area = float(orig_shape[0] * orig_shape[1]) if (orig_shape[0] * orig_shape[1]) > 0 else 1.0
        crop_area = float(crop_shape[0] * crop_shape[1])
        area_ratio = crop_area / orig_area

        # Apply CLAHE on luminance channel for shadow normalization
        enhanced_bgr = apply_clahe(cropped_bgr)
        pil_image = bgr_to_pil(enhanced_bgr)

        metadata = {
            "note_detected": detected,
            "original_shape": orig_shape,
            "cropped_shape": crop_shape,
            "clahe_applied": True,
            "area_ratio": round(area_ratio, 5),
        }
        return pil_image, enhanced_bgr, metadata

    def classify(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Classifies physical damages on an Indian currency note image.
        Returns structured multi-label classification results.
        """
        # Validate input if path
        if isinstance(image_input, (str, Path)):
            is_val, reason = is_valid_image(image_input)
            if not is_val:
                return {
                    "success": False,
                    "error": f"Invalid image ({reason})",
                    "damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0},
                    "raw_damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0},
                    "final_damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0},
                    "predicted_labels": [],
                    "evidence_scores": {"Torn": 0.0, "Folded": 0.0, "Burnt": 0.0, "Stain": 0.0},
                    "dynamic_thresholds": dict(self.thresholds),
                    "threshold_adjustments": {"Torn": 0.0, "Folded": 0.0, "Burnt": 0.0, "Stain": 0.0},
                    "context_features": {},
                    "classification_status": config.STATUS_ERROR,
                    "preprocessing_metadata": {},
                    "normal_rule": "NOT(Burnt OR Torn)" if self.calibration_version in ("v3.1", "v3_1") else "NOT(Torn OR Folded OR Burnt OR Stain)",
                    "burnt_implies_stain": True if self.calibration_version in ("v3.1", "v3_1") else False,
                }

        try:
            pil_image, enhanced_bgr, prep_meta = self.preprocess_image(image_input)
        except Exception as e:
            return {
                "success": False,
                "error": f"Preprocessing failed: {e}",
                "damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0},
                "raw_damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0},
                "final_damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0},
                "predicted_labels": [],
                "evidence_scores": {"Torn": 0.0, "Folded": 0.0, "Burnt": 0.0, "Stain": 0.0},
                "dynamic_thresholds": dict(self.thresholds),
                "threshold_adjustments": {"Torn": 0.0, "Folded": 0.0, "Burnt": 0.0, "Stain": 0.0},
                "context_features": {},
                "classification_status": config.STATUS_ERROR,
                "preprocessing_metadata": {},
                "normal_rule": "NOT(Burnt OR Torn)" if self.calibration_version in ("v3.1", "v3_1") else "NOT(Torn OR Folded OR Burnt OR Stain)",
                "burnt_implies_stain": True if self.calibration_version in ("v3.1", "v3_1") else False,
            }

        # Encode image features
        img_feats = self.clip_wrapper.encode_images([pil_image])

        # Compute contrastive scores (pos - neg)
        scores_list = compute_contrastive_scores(
            image_features=img_feats,
            positive_embeddings=self.clip_wrapper.positive_embeddings,
            negative_embeddings=self.clip_wrapper.negative_embeddings,
            aggregation_method=config.PROMPT_AGGREGATION_METHOD,
        )
        scores = scores_list[0]

        evidence_dict = {
            "Torn": scores["torn_evidence_score"],
            "Folded": scores["folded_evidence_score"],
            "Burnt": scores["burnt_evidence_score"],
            "Stain": scores["stain_evidence_score"],
        }

        # Extract context features from processed image and evidence
        context_features = extract_context_features(
            image=enhanced_bgr,
            evidence_scores=evidence_dict,
            preprocessing_meta=prep_meta,
        )

        if self.calibration_version in ("v3.1", "v3_1", "v3"):
            version_str = "v3.1" if self.calibration_version in ("v3.1", "v3_1") else "v3"
            dyn_res = self.dynamic_calculator.apply_dynamic_thresholds(
                evidence_scores=evidence_dict,
                context_features=context_features,
                version=version_str,
            )
            damage_labels = dyn_res["damage_labels"]
            raw_damage_labels = dyn_res["raw_damage_labels"]
            final_damage_labels = dyn_res["final_damage_labels"]
            predicted_labels = dyn_res["predicted_labels"]
            dynamic_thresholds = dyn_res["dynamic_thresholds"]
            threshold_adjustments = dyn_res["threshold_adjustments"]

            # Determine classification status (uncertain / moderate / strong)
            min_margin = min(
                abs(evidence_dict[cat] - dynamic_thresholds[cat])
                for cat in ["Torn", "Folded", "Burnt", "Stain"]
            )
            if min_margin < self.ambiguity_band:
                status = config.STATUS_UNCERTAIN
            elif min_margin >= self.strong_margin:
                status = config.STATUS_STRONG
            else:
                status = config.STATUS_MODERATE

            normal_rule = dyn_res["normal_rule"]
            burnt_implies_stain = dyn_res["burnt_implies_stain"]
        else:
            # Baseline V2 fixed thresholds
            decisions = apply_evidence_thresholds(
                scores=scores,
                thresholds=self.thresholds,
                ambiguity_band=self.ambiguity_band,
                strong_margin=self.strong_margin,
            )
            raw_damage_labels = {
                "Torn": decisions["Torn"],
                "Folded": decisions["Folded"],
                "Burnt": decisions["Burnt"],
                "Stain": decisions["Stain"],
                "Normal": decisions["Normal"],
            }
            final_damage_labels = dict(raw_damage_labels)
            damage_labels = dict(raw_damage_labels)
            predicted_labels = [
                lbl.strip()
                for lbl in decisions["predicted_labels"].split(";")
                if lbl.strip()
            ]
            dynamic_thresholds = dict(self.thresholds)
            threshold_adjustments = {cat: 0.0 for cat in ["Torn", "Folded", "Burnt", "Stain"]}
            status = decisions["classification_status"]
            normal_rule = "NOT(Torn OR Folded OR Burnt OR Stain)"
            burnt_implies_stain = False

        return {
            "success": True,
            "damage_labels": final_damage_labels,
            "raw_damage_labels": raw_damage_labels,
            "final_damage_labels": final_damage_labels,
            "predicted_labels": predicted_labels,
            "evidence_scores": evidence_dict,
            "dynamic_thresholds": dynamic_thresholds,
            "threshold_adjustments": threshold_adjustments,
            "context_features": context_features,
            "raw_scores": scores,
            "classification_status": status,
            "preprocessing_metadata": prep_meta,
            "normal_rule": normal_rule,
            "burnt_implies_stain": burnt_implies_stain,
        }

    def classify_batch(
        self, image_inputs: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[Dict[str, Any]]:
        """Classifies a batch of images sequentially or in batches."""
        return [self.classify(img) for img in image_inputs]
