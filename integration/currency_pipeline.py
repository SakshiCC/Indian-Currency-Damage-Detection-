"""
currency_pipeline.py
============================================================
Top-Level Orchestration Pipeline for Indian Currency Note Analysis.

Coordinates:
  1. Yash's Denomination Predictor (via yash_adapter)
  2. Vinit's V3.1 Multi-Label Damage Classifier (via vinit_adapter)
  3. Sakshi's clean interface hook (via sakshi_interface)

Preprocessing separation:
  Original Image
     ├─> Yash (internal 224x224 RGB conversion & tensor creation)
     └─> Vinit (OpenCV note cropping, perspective correction, CLAHE, CLIP ViT-B/32)
============================================================
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import numpy as np
from PIL import Image

from integration.yash_adapter import get_yash_adapter, YashDenominationAdapter
from integration.vinit_adapter import get_vinit_adapter, VinitDamageAdapter
from integration.sakshi_interface import get_active_damage_labels, get_sakshi_handoff_payload

logger = logging.getLogger("integration.currency_pipeline")


class CurrencyPipeline:
    """
    Unified, multi-modal pipeline coordinating Yash's denomination recognition
    and Vinit's V3.1 multi-label damage classification.
    """

    def __init__(
        self,
        device: str = "cpu",
        calibration_version: str = "v3.1",
        use_preprocessing: bool = True,
        confidence_threshold: float = 0.0,
    ):
        self.device = device
        self.calibration_version = calibration_version
        self.use_preprocessing = use_preprocessing

        # Adapters maintain singletons to prevent repeated model reloading
        self.denomination_adapter = get_yash_adapter()
        self.damage_adapter = get_vinit_adapter(
            device=device,
            calibration_version=calibration_version,
        )

    def analyze(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """
        Run complete end-to-end analysis on an Indian currency note image.

        Flow:
          Input Image
             ├─> Yash Denomination Predictor (EfficientNetB0)
             └─> Vinit V3.1 Damage Classifier (CLIP ViT-B/32 + Dynamic Thresholds)
          Combined Schema
             └─> Sakshi Handoff Hook
        """
        img_str = str(image_input) if isinstance(image_input, (str, Path)) else None
        filename = Path(image_input).name if isinstance(image_input, (str, Path)) else "in_memory_image"

        # 1. Denomination Recognition (Yash)
        # Yash's module receives original image and applies its own 224x224 preprocessing
        denom_res = self.denomination_adapter.predict(image_input)

        # 2. Damage Classification (Vinit)
        # Vinit's module receives original image and applies note detection, CLAHE, and CLIP
        damage_res = self.damage_adapter.predict(image_input)

        # 3. Sakshi Handoff Hook
        sakshi_active = get_active_damage_labels(damage_res)
        sakshi_handoff = get_sakshi_handoff_payload(damage_res, image_path=img_str)

        # Format clean, consolidated denomination dictionary
        denom_formatted = {
            "value": denom_res.get("denomination", "Unknown"),
            "confidence": denom_res.get("confidence", 0.0),
            "confidence_percent": denom_res.get("confidence_percent", 0.0),
            "class_index": denom_res.get("class_index", -1),
            "is_background": denom_res.get("is_background", False),
            "accepted": denom_res.get("accepted", False),
            "available": denom_res.get("available", False),
        }
        if not denom_res.get("available", False) and "reason" in denom_res:
            denom_formatted["status_note"] = denom_res["reason"]

        # Format clean, consolidated damage dictionary
        damage_formatted = {
            "raw_labels": damage_res.get("raw_damage_labels", {}),
            "final_labels": damage_res.get("final_damage_labels", {}),
            "damage_labels": damage_res.get("damage_labels", {}),
            "predicted_labels": damage_res.get("predicted_labels", []),
            "evidence_scores": damage_res.get("evidence_scores", {}),
            "dynamic_thresholds": damage_res.get("dynamic_thresholds", {}),
            "classification_status": damage_res.get("classification_status", "uncertain"),
            "min_evidence_margin": damage_res.get("min_evidence_margin", 0.0),
            "calibration_version": damage_res.get("calibration_version", self.calibration_version),
            "normal_rule": damage_res.get("normal_rule", "NOT(Burnt OR Torn)"),
            "burnt_implies_stain": damage_res.get("burnt_implies_stain", True),
        }

        return {
            "success": bool(damage_res.get("success", False)),
            "image": {
                "path": img_str,
                "filename": filename,
            },
            "denomination": denom_formatted,
            "damage": damage_formatted,
            "preprocessing": damage_res.get("preprocessing_metadata", {}),
            "sakshi_active_damages": sakshi_active,
            "sakshi_handoff": sakshi_handoff,
        }

    def analyze_batch(
        self,
        image_inputs: List[Union[str, Path, np.ndarray, Image.Image]],
    ) -> List[Dict[str, Any]]:
        """Batch analysis across multiple images."""
        return [self.analyze(img) for img in image_inputs]


_DEFAULT_PIPELINE: Optional[CurrencyPipeline] = None


def get_default_pipeline(device: str = "cpu", calibration_version: str = "v3.1") -> CurrencyPipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = CurrencyPipeline(device=device, calibration_version=calibration_version)
    return _DEFAULT_PIPELINE


def run_currency_analysis(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    device: str = "cpu",
    calibration_version: str = "v3.1",
) -> Dict[str, Any]:
    """
    Convenience function for full banknote analysis.
    """
    pipeline = get_default_pipeline(device=device, calibration_version=calibration_version)
    return pipeline.analyze(image_input)
