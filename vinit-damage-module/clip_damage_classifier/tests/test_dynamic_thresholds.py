"""
test_dynamic_thresholds.py
Tests the V3 Dynamic Threshold Calculator, mathematical constraints,
safety clipping bounds, and safe V2 fallbacks.
"""

import unittest
from clip_damage_classifier.calibration.dynamic_thresholds import (
    DynamicThresholdCalculator,
    DEFAULT_V2_BASE_THRESHOLDS,
    DEFAULT_MIN_THRESHOLD,
    DEFAULT_MAX_THRESHOLD,
)


class TestDynamicThresholds(unittest.TestCase):

    def setUp(self):
        self.calc = DynamicThresholdCalculator()

    def test_base_threshold_anchoring(self):
        """Verifies default base thresholds match V2 baseline exactly."""
        self.assertEqual(self.calc.base_thresholds["Torn"], -0.0005)
        self.assertEqual(self.calc.base_thresholds["Folded"], 0.0000)
        self.assertEqual(self.calc.base_thresholds["Burnt"], 0.0038)
        self.assertEqual(self.calc.base_thresholds["Stain"], 0.0000)

    def test_safety_bounds_clipping(self):
        """Verifies thresholds are clipped to [-0.008, +0.008]."""
        # Test extreme context that would blow up unclipped threshold
        extreme_context = {
            "edge_density": 10.0,
            "contrast": -10.0,
            "area_ratio": 0.0,
            "dark_ratio": 10.0,
            "brightness": -10.0,
        }
        ev_scores = {"Torn": 0.0, "Folded": 0.0, "Burnt": 0.0, "Stain": 0.0}

        dyn_info = self.calc.calculate_all_dynamic_thresholds(ev_scores, extreme_context)
        for cat, info in dyn_info.items():
            t = info["dynamic_threshold"]
            self.assertGreaterEqual(t, DEFAULT_MIN_THRESHOLD)
            self.assertLessEqual(t, DEFAULT_MAX_THRESHOLD)

    def test_positive_evidence_coefficient_constraint(self):
        """Verifies that non-positive evidence coefficients trigger safe fallback."""
        # Inject mock model with negative w_evidence
        mock_calc = DynamicThresholdCalculator()
        mock_calc.models["Torn"] = {
            "w_evidence": -0.05,
            "intercept": 1.0,
            "feature_names": ["edge_density"],
            "w_context": [0.1],
            "scaler_means": {"edge_density": 0.0},
            "scaler_stds": {"edge_density": 1.0},
            "status": "fallback_nonpositive_evidence",
        }

        t_dyn, adj, mode = mock_calc.calculate_dynamic_threshold(
            "Torn", {"Torn": 0.0}, {"edge_density": 0.5}
        )
        self.assertEqual(t_dyn, DEFAULT_V2_BASE_THRESHOLDS["Torn"])
        self.assertEqual(adj, 0.0)
        self.assertIn("fallback", mode)

    def test_normal_exclusivity_invariant(self):
        """Verifies Normal = 1 iff Torn=Folded=Burnt=Stain=0."""
        # Case 1: All clean -> Normal = 1
        clean_ev = {"Torn": -0.01, "Folded": -0.01, "Burnt": -0.01, "Stain": -0.01}
        clean_ctx = {"edge_density": 0.05, "contrast": 0.3, "area_ratio": 0.8, "dark_ratio": 0.0, "brightness": 0.5}
        res_clean = self.calc.apply_dynamic_thresholds(clean_ev, clean_ctx)
        self.assertEqual(res_clean["damage_labels"]["Normal"], 1)
        self.assertEqual(res_clean["predicted_labels"], ["Normal"])

        # Case 2: One damage active -> Normal = 0
        damage_ev = {"Torn": 0.01, "Folded": -0.01, "Burnt": -0.01, "Stain": -0.01}
        res_damage = self.calc.apply_dynamic_thresholds(damage_ev, clean_ctx)
        self.assertEqual(res_damage["damage_labels"]["Normal"], 0)
        self.assertEqual(res_damage["damage_labels"]["Torn"], 1)
        self.assertIn("Torn", res_damage["predicted_labels"])
        self.assertNotIn("Normal", res_damage["predicted_labels"])

    def test_multilabel_cooccurrence(self):
        """Verifies multi-label co-occurrence is preserved."""
        multi_ev = {"Torn": 0.01, "Folded": 0.01, "Burnt": -0.01, "Stain": 0.01}
        clean_ctx = {"edge_density": 0.05, "contrast": 0.3, "area_ratio": 0.8, "dark_ratio": 0.0, "brightness": 0.5}
        res = self.calc.apply_dynamic_thresholds(multi_ev, clean_ctx)
        self.assertEqual(res["damage_labels"]["Torn"], 1)
        self.assertEqual(res["damage_labels"]["Folded"], 1)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["damage_labels"]["Burnt"], 0)
        self.assertEqual(res["damage_labels"]["Normal"], 0)
        self.assertEqual(set(res["predicted_labels"]), {"Torn", "Folded", "Stain"})


if __name__ == "__main__":
    unittest.main()
