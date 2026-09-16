"""
Dynamic Threshold Calculation Engine for Vinit V3 & V3.1 Damage Classifier.

Implements mathematically grounded dynamic threshold evaluation:
    logit(P(C=1)) = intercept_C + w_ev,C * evidence_C + sum(w_i,C * norm_context_i)
    T_dynamic_C(context) = -(intercept_C + sum(w_i,C * norm_context_i)) / w_ev,C

Ensures:
  1. w_ev,C > 0 (positive evidence coefficient constraint; safe fallback to base threshold if violated).
  2. Safety bounds clipping [-0.008, +0.008].
  3. Base threshold anchoring to V2 values.
  4. Multi-label independence: damage_C = 1 iff evidence_C >= T_dynamic_C.
  5. Configurable Normal rule:
     - V3 (legacy): Normal = 1 iff Torn=Folded=Burnt=Stain=0
     - V3.1 (new project rule): Normal = 1 iff Burnt=0 AND Torn=0 (Folded & Stain do not affect Normal)
  6. V3.1 Burnt -> Stain consistency: if Burnt=1 then Stain=1.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger("clip_damage_classifier.calibration.dynamic_thresholds")

# Anchored baseline V2 thresholds
DEFAULT_V2_BASE_THRESHOLDS: Dict[str, float] = {
    "Torn": -0.0005,
    "Folded": 0.0000,
    "Burnt": 0.0038,
    "Stain": 0.0000,
}

DAMAGE_CATEGORIES: List[str] = ["Torn", "Folded", "Burnt", "Stain"]

# Default safety bounds
DEFAULT_MIN_THRESHOLD: float = -0.0080
DEFAULT_MAX_THRESHOLD: float = +0.0080


class DynamicThresholdCalculator:
    """
    Computes dynamic thresholds per damage class from context features and CLIP evidence.
    Loads learned parameters from calibrated_dynamic_thresholds.json or falls back safely.
    """

    def __init__(
        self,
        calibration_file: Optional[Path] = None,
        base_thresholds: Optional[Dict[str, float]] = None,
        min_threshold: float = DEFAULT_MIN_THRESHOLD,
        max_threshold: float = DEFAULT_MAX_THRESHOLD,
        version: str = "v3.1",
    ):
        self.version = version
        self.base_thresholds = dict(base_thresholds or DEFAULT_V2_BASE_THRESHOLDS)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.calibration_file = calibration_file

        if calibration_file is not None and Path(calibration_file).exists():
            self.load_calibration(Path(calibration_file))
        else:
            # Check standard path
            default_p = (
                Path(__file__).resolve().parent.parent
                / "outputs"
                / "v3"
                / "calibrated_dynamic_thresholds.json"
            )
            if default_p.exists():
                self.load_calibration(default_p)
            else:
                logger.info(
                    "No calibrated_dynamic_thresholds.json found; initialized with base V2 thresholds."
                )

    def load_calibration(self, file_path: Path):
        """Load calibrated model coefficients from JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.models = data.get("classes", {})
            logger.info(f"Loaded calibration parameters for {list(self.models.keys())} from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load calibration from {file_path}: {e}")
            self.models = {}

    def calculate_dynamic_threshold(
        self,
        category: str,
        evidence_scores: Dict[str, float],
        context_features: Dict[str, float],
    ) -> Tuple[float, float, str]:
        """
        Calculate dynamic threshold for a specific category.

        Returns:
            Tuple of (dynamic_threshold, threshold_adjustment, decision_mode)
        """
        base_t = self.base_thresholds.get(category, 0.0)

        if category not in self.models:
            return base_t, 0.0, "v2_fallback_no_model"

        model_info = self.models[category]
        w_ev = float(model_info.get("w_evidence", 0.0))
        intercept = float(model_info.get("intercept", 0.0))
        feature_names = model_info.get("feature_names", [])
        w_context = model_info.get("w_context", [])
        scaler_means = model_info.get("scaler_means", {})
        scaler_stds = model_info.get("scaler_stds", {})
        status = model_info.get("status", "valid")

        # Constraint check: w_evidence must be strictly positive
        if w_ev <= 0.0 or status != "calibrated":
            logger.debug(
                f"Class {category}: w_evidence ({w_ev}) <= 0 or status '{status}'. Falling back to V2 base {base_t}."
            )
            return base_t, 0.0, f"v2_fallback_{status}"

        # Standardize and compute context dot product
        context_contrib = 0.0
        for name, w_ctx in zip(feature_names, w_context):
            raw_val = float(context_features.get(name, 0.0))
            mean_val = float(scaler_means.get(name, 0.0))
            std_val = float(scaler_stds.get(name, 1.0))
            if std_val < 1e-8:
                std_val = 1.0
            norm_val = (raw_val - mean_val) / std_val
            context_contrib += float(w_ctx) * norm_val

        # P=0.5 decision boundary:
        raw_dyn_t = -(intercept + context_contrib) / w_ev
        clipped_dyn_t = float(np.clip(raw_dyn_t, self.min_threshold, self.max_threshold))

        adjustment = round(clipped_dyn_t - base_t, 6)
        mode = "dynamic_calibrated"
        if clipped_dyn_t != raw_dyn_t:
            mode = "dynamic_clipped"

        return clipped_dyn_t, adjustment, mode

    def calculate_all_dynamic_thresholds(
        self,
        evidence_scores: Dict[str, float],
        context_features: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate dynamic thresholds for all damage categories.
        """
        results = {}
        for cat in DAMAGE_CATEGORIES:
            t_dyn, adj, mode = self.calculate_dynamic_threshold(
                cat, evidence_scores, context_features
            )
            results[cat] = {
                "dynamic_threshold": round(t_dyn, 6),
                "threshold_adjustment": adj,
                "mode": mode,
                "base_threshold": self.base_thresholds[cat],
            }
        return results

    def apply_dynamic_thresholds(
        self,
        evidence_scores: Dict[str, float],
        context_features: Dict[str, float],
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate damage decisions using dynamic class-specific thresholds.

        Args:
            evidence_scores: dict mapping category -> float evidence
            context_features: dict mapping feature name -> float
            version: "v3" (old Normal rule) or "v3.1" (new Normal rule + Burnt->Stain)
        """
        if version is None:
            version = self.version

        dyn_info = self.calculate_all_dynamic_thresholds(
            evidence_scores, context_features
        )

        raw_damage_labels: Dict[str, int] = {}
        dynamic_thresholds: Dict[str, float] = {}
        threshold_adjustments: Dict[str, float] = {}
        modes: Dict[str, str] = {}

        for cat in DAMAGE_CATEGORIES:
            t_dyn = dyn_info[cat]["dynamic_threshold"]
            ev = float(evidence_scores.get(cat, 0.0))
            is_active = 1 if ev >= t_dyn else 0

            raw_damage_labels[cat] = is_active
            dynamic_thresholds[cat] = t_dyn
            threshold_adjustments[cat] = dyn_info[cat]["threshold_adjustment"]
            modes[cat] = dyn_info[cat]["mode"]

        final_damage_labels = dict(raw_damage_labels)

        if version in ("v3.1", "v3_1"):
            # 1. Burnt -> Stain consistency rule
            if final_damage_labels["Burnt"] == 1:
                final_damage_labels["Stain"] = 1

            # 2. New Normal rule: Normal = 1 iff Burnt == 0 AND Torn == 0
            is_normal = 1 if (final_damage_labels["Burnt"] == 0 and final_damage_labels["Torn"] == 0) else 0
            final_damage_labels["Normal"] = is_normal

            predicted_labels = [c for c in DAMAGE_CATEGORIES if final_damage_labels[c] == 1]
            if is_normal == 1:
                predicted_labels.append("Normal")
        else:
            # Legacy V3 rule: Normal = 1 iff all 4 damages are 0
            any_damage = sum(raw_damage_labels[cat] for cat in DAMAGE_CATEGORIES) > 0
            if any_damage:
                final_damage_labels["Normal"] = 0
                predicted_labels = [c for c in DAMAGE_CATEGORIES if raw_damage_labels[c] == 1]
            else:
                final_damage_labels["Normal"] = 1
                predicted_labels = ["Normal"]

        return {
            "damage_labels": final_damage_labels,
            "raw_damage_labels": raw_damage_labels,
            "final_damage_labels": final_damage_labels,
            "predicted_labels": predicted_labels,
            "dynamic_thresholds": dynamic_thresholds,
            "threshold_adjustments": threshold_adjustments,
            "modes": modes,
            "normal_rule": "NOT(Burnt OR Torn)" if version in ("v3.1", "v3_1") else "NOT(Torn OR Folded OR Burnt OR Stain)",
            "burnt_implies_stain": True if version in ("v3.1", "v3_1") else False,
        }


# Module-level convenience functions
_DEFAULT_CALCULATOR: Optional[DynamicThresholdCalculator] = None


def get_default_calculator() -> DynamicThresholdCalculator:
    global _DEFAULT_CALCULATOR
    if _DEFAULT_CALCULATOR is None:
        _DEFAULT_CALCULATOR = DynamicThresholdCalculator()
    return _DEFAULT_CALCULATOR


def calculate_dynamic_threshold(
    category: str,
    evidence_scores: Dict[str, float],
    context_features: Dict[str, float],
) -> float:
    t_dyn, _, _ = get_default_calculator().calculate_dynamic_threshold(
        category, evidence_scores, context_features
    )
    return t_dyn


def calculate_all_dynamic_thresholds(
    evidence_scores: Dict[str, float],
    context_features: Dict[str, float],
) -> Dict[str, float]:
    dyn_info = get_default_calculator().calculate_all_dynamic_thresholds(
        evidence_scores, context_features
    )
    return {cat: info["dynamic_threshold"] for cat, info in dyn_info.items()}


def apply_dynamic_thresholds(
    evidence_scores: Dict[str, float],
    context_features: Dict[str, float],
    version: str = "v3",
) -> Dict[str, Any]:
    return get_default_calculator().apply_dynamic_thresholds(
        evidence_scores, context_features, version=version
    )
