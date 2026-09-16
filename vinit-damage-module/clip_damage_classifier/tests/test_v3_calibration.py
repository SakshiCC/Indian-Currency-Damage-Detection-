"""
test_v3_calibration.py
Tests the calibration artifact JSON schema, parameter values,
and backward compatibility of DamageClassifier interface with V2 and V3.
"""

import json
import unittest
from pathlib import Path
from clip_damage_classifier import DamageClassifier
from clip_damage_classifier.calibration.dynamic_thresholds import DEFAULT_V2_BASE_THRESHOLDS


class TestV3Calibration(unittest.TestCase):

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.calib_file = (
            self.project_root
            / "clip_damage_classifier"
            / "outputs"
            / "v3"
            / "calibrated_dynamic_thresholds.json"
        )

    def test_calibration_json_schema(self):
        """Verifies the exported JSON artifact exists and contains valid structure."""
        self.assertTrue(self.calib_file.exists(), "calibrated_dynamic_thresholds.json not found")
        with open(self.calib_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("version"), "v3")
        self.assertIn("classes", data)
        for cat in ["Torn", "Folded", "Burnt", "Stain"]:
            self.assertIn(cat, data["classes"])
            cat_info = data["classes"][cat]
            self.assertIn("status", cat_info)
            self.assertIn("w_evidence", cat_info)
            self.assertIn("intercept", cat_info)
            self.assertIn("feature_names", cat_info)
            self.assertIn("w_context", cat_info)

    def test_v2_backward_compatibility(self):
        """Verifies calibration_version='v2' operates with fixed thresholds."""
        classifier_v2 = DamageClassifier(
            device="cpu", use_preprocessing=False, calibration_version="v2"
        )
        self.assertEqual(classifier_v2.calibration_version, "v2")
        self.assertEqual(classifier_v2.thresholds["Burnt"], DEFAULT_V2_BASE_THRESHOLDS["Burnt"])

    def test_v3_default_mode(self):
        """Verifies DamageClassifier defaults to V3.1 mode and supports explicit V3."""
        classifier_v31 = DamageClassifier(device="cpu", use_preprocessing=False)
        self.assertEqual(classifier_v31.calibration_version, "v3.1")

        classifier_v3 = DamageClassifier(
            device="cpu", use_preprocessing=False, calibration_version="v3"
        )
        self.assertEqual(classifier_v3.calibration_version, "v3")


if __name__ == "__main__":
    unittest.main()
