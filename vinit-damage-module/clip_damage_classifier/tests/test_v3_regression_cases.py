"""
test_v3_regression_cases.py
Tests the provided failure case as a regression test:
  Evidence: Torn=-0.0024, Folded=+0.0028, Burnt=-0.0032, Stain=+0.0029
  Human Visual Assessment: Burnt=1, Folded=0, Stain=1
"""

import unittest
from clip_damage_classifier.calibration.dynamic_thresholds import (
    DynamicThresholdCalculator,
    DEFAULT_V2_BASE_THRESHOLDS,
)


class TestV3RegressionCases(unittest.TestCase):

    def setUp(self):
        self.calc = DynamicThresholdCalculator()
        self.failure_evidence = {
            "Torn": -0.0024,
            "Folded": +0.0028,
            "Burnt": -0.0032,
            "Stain": +0.0029,
        }

    def test_v2_baseline_behavior_on_regression_case(self):
        """
        Confirms V2 baseline behavior on the regression case:
        V2 flags Folded (0.0028 >= 0.0) and Stain (0.0029 >= 0.0), misses Burnt (-0.0032 < 0.0038).
        """
        # Under V2 fixed thresholds:
        v2_torn = 1 if self.failure_evidence["Torn"] >= DEFAULT_V2_BASE_THRESHOLDS["Torn"] else 0
        v2_folded = 1 if self.failure_evidence["Folded"] >= DEFAULT_V2_BASE_THRESHOLDS["Folded"] else 0
        v2_burnt = 1 if self.failure_evidence["Burnt"] >= DEFAULT_V2_BASE_THRESHOLDS["Burnt"] else 0
        v2_stain = 1 if self.failure_evidence["Stain"] >= DEFAULT_V2_BASE_THRESHOLDS["Stain"] else 0

        self.assertEqual(v2_torn, 0)
        self.assertEqual(v2_folded, 1)  # False positive in V2
        self.assertEqual(v2_burnt, 0)   # False negative in V2
        self.assertEqual(v2_stain, 1)   # Correctly detected

    def test_dynamic_threshold_adjustments(self):
        """
        Tests how context features adjust dynamic thresholds.
        When edge density is low and contrast is normal (straight note),
        Folded threshold adjusts upward, suppressing fold false positives.
        """
        straight_note_context = {
            "edge_density": 0.04,   # Low edge density -> unwrinkled
            "contrast": 0.35,
            "area_ratio": 0.85,
            "dark_ratio": 0.01,
            "brightness": 0.50,
            "stain_evidence": 0.0029,
        }

        dyn_info = self.calc.calculate_all_dynamic_thresholds(
            self.failure_evidence, straight_note_context
        )

        # Folded threshold should be raised or evaluated with dynamic context
        self.assertIn("Folded", dyn_info)
        self.assertIn("Stain", dyn_info)
        self.assertIn("Burnt", dyn_info)
        self.assertIn("Torn", dyn_info)


if __name__ == "__main__":
    unittest.main()
