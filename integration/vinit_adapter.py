"""
vinit_adapter.py
============================================================
Thin, Production Adapter for Vinit's Damage Classification Module.

Wraps DamageClassifier located in vinit-damage-module/clip_damage_classifier.
Preserves:
  - V3.1 context-aware calibration
  - Consistency rules: Burnt -> Stain, Normal = NOT(Burnt OR Torn)
  - Singleton model reuse (CLIP ViT-B/32 instantiated once)
  - Full output metadata: raw_labels, final_labels, evidence_scores, status
============================================================
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import numpy as np
from PIL import Image

logger = logging.getLogger("integration.vinit_adapter")

# Ensure vinit-damage-module is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VINIT_DIR = PROJECT_ROOT / "vinit-damage-module"
if str(VINIT_DIR) not in sys.path:
    sys.path.insert(0, str(VINIT_DIR))

from clip_damage_classifier.interface import DamageClassifier


class VinitDamageAdapter:
    """
    Thin adapter wrapping Vinit's DamageClassifier without altering V3.1 logic.
    """

    def __init__(
        self,
        device: str = "cpu",
        calibration_version: str = "v3.1",
        use_preprocessing: bool = True,
    ):
        self.device = device
        self.calibration_version = calibration_version
        self.use_preprocessing = use_preprocessing
        self._classifier: Optional[DamageClassifier] = None
        self._initialize()

    def _initialize(self):
        """Instantiate the underlying DamageClassifier singleton once."""
        try:
            logger.info(
                f"Initializing Vinit DamageClassifier (device={self.device}, version={self.calibration_version})..."
            )
            self._classifier = DamageClassifier(
                device=self.device,
                use_preprocessing=self.use_preprocessing,
                calibration_version=self.calibration_version,
            )
            logger.info("Vinit DamageClassifier initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Vinit DamageClassifier: {e}")
            self._classifier = None

    @property
    def is_available(self) -> bool:
        return self._classifier is not None

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """
        Run multi-label damage classification on an input image.

        Returns normalized dictionary preserving all V3.1 fields:
          - damage_labels: 5 classes (Torn, Folded, Burnt, Stain, Normal)
          - raw_damage_labels: 4 classes (pre-consistency rules)
          - final_damage_labels: 5 classes (post-consistency rules)
          - predicted_labels: list of active class strings
          - evidence_scores: dict of contrastive evidence
          - dynamic_thresholds: dict of applied thresholds
          - classification_status: strong_prediction / moderate_prediction / uncertain
          - min_evidence_margin: float
          - normal_rule: "NOT(Burnt OR Torn)"
          - burnt_implies_stain: True
          - calibration_version: "v3.1"
          - preprocessing_metadata: crop and note detection info
        """
        if self._classifier is None:
            self._initialize()
            if self._classifier is None:
                return {
                    "success": False,
                    "error": "DamageClassifier could not be initialized.",
                    "damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 1},
                    "raw_damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0},
                    "final_damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 1},
                    "predicted_labels": ["Normal"],
                    "evidence_scores": {},
                    "classification_status": "error",
                    "calibration_version": self.calibration_version,
                }

        # Delegate directly to Vinit's existing classify() method
        raw_res = self._classifier.classify(image_input)

        # Normalize output schema for integration
        return {
            "success": raw_res.get("success", True),
            "damage_labels": raw_res.get("damage_labels", {}),
            "raw_damage_labels": raw_res.get("raw_damage_labels", raw_res.get("damage_labels", {})),
            "final_damage_labels": raw_res.get("final_damage_labels", raw_res.get("damage_labels", {})),
            "predicted_labels": raw_res.get("predicted_labels", []),
            "evidence_scores": raw_res.get("evidence_scores", {}),
            "dynamic_thresholds": raw_res.get("dynamic_thresholds", {}),
            "threshold_adjustments": raw_res.get("threshold_adjustments", {}),
            "classification_status": raw_res.get("classification_status", "uncertain"),
            "min_evidence_margin": raw_res.get("min_evidence_margin", 0.0),
            "normal_rule": raw_res.get("normal_rule", "NOT(Burnt OR Torn)"),
            "burnt_implies_stain": raw_res.get("burnt_implies_stain", True),
            "calibration_version": raw_res.get("calibration_version", self.calibration_version),
            "preprocessing_metadata": raw_res.get("preprocessing_metadata", {}),
            "raw": raw_res,
        }


# Module singleton
_DEFAULT_VINIT_ADAPTER: Optional[VinitDamageAdapter] = None


def get_vinit_adapter(device: str = "cpu", calibration_version: str = "v3.1") -> VinitDamageAdapter:
    global _DEFAULT_VINIT_ADAPTER
    if _DEFAULT_VINIT_ADAPTER is None:
        _DEFAULT_VINIT_ADAPTER = VinitDamageAdapter(device=device, calibration_version=calibration_version)
    return _DEFAULT_VINIT_ADAPTER


def predict_damage_safe(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    device: str = "cpu",
    calibration_version: str = "v3.1",
) -> Dict[str, Any]:
    """Module-level convenience function."""
    return get_vinit_adapter(device=device, calibration_version=calibration_version).predict(image_input)
