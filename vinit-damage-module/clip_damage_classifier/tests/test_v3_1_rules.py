"""
test_v3_1_rules.py
============================================================
Comprehensive test suite for V3.1 rules:
  1. 8 Canonical Normal Decision Cases (Section 17)
  2. 4 Burnt -> Stain Consistency Cases (Section 18)
  3. Regression Image Behavior (Section 19)
  4. V3.1 Interface metadata & backward compatibility
============================================================
"""

import unittest
from pathlib import Path
import cv2
import numpy as np

from clip_damage_classifier.calibration.dynamic_thresholds import DynamicThresholdCalculator
from clip_damage_classifier.interface import DamageClassifier


class TestV31Rules(unittest.TestCase):

    def setUp(self):
        self.calc_v31 = DynamicThresholdCalculator(version="v3.1")
        self.calc_v3 = DynamicThresholdCalculator(version="v3")
        self.dummy_context = {
            "edge_density": 0.05,
            "contrast": 0.3,
            "area_ratio": 0.8,
            "dark_ratio": 0.0,
            "brightness": 0.5,
            "stain_evidence": 0.0,
        }

    # ----------------------------------------------------------------------
    # 1. 8 Canonical Normal Decision Cases (Section 17)
    # ----------------------------------------------------------------------
    def test_case_a_all_clean(self):
        """Case A: Torn=0, Folded=0, Burnt=0, Stain=0 -> Normal=1."""
        ev = {"Torn": -0.01, "Folded": -0.01, "Burnt": -0.01, "Stain": -0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Torn"], 0)
        self.assertEqual(res["damage_labels"]["Folded"], 0)
        self.assertEqual(res["damage_labels"]["Burnt"], 0)
        self.assertEqual(res["damage_labels"]["Stain"], 0)
        self.assertEqual(res["damage_labels"]["Normal"], 1)
        self.assertEqual(res["predicted_labels"], ["Normal"])

    def test_case_b_folded_only(self):
        """Case B: Torn=0, Folded=1, Burnt=0, Stain=0 -> Normal=1 (NEW V3.1 rule!)."""
        ev = {"Torn": -0.01, "Folded": 0.01, "Burnt": -0.01, "Stain": -0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Torn"], 0)
        self.assertEqual(res["damage_labels"]["Folded"], 1)
        self.assertEqual(res["damage_labels"]["Burnt"], 0)
        self.assertEqual(res["damage_labels"]["Stain"], 0)
        self.assertEqual(res["damage_labels"]["Normal"], 1)
        self.assertIn("Normal", res["predicted_labels"])
        self.assertIn("Folded", res["predicted_labels"])

    def test_case_c_stain_only(self):
        """Case C: Torn=0, Folded=0, Burnt=0, Stain=1 -> Normal=1 (NEW V3.1 rule!)."""
        ev = {"Torn": -0.01, "Folded": -0.01, "Burnt": -0.01, "Stain": 0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Torn"], 0)
        self.assertEqual(res["damage_labels"]["Folded"], 0)
        self.assertEqual(res["damage_labels"]["Burnt"], 0)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["damage_labels"]["Normal"], 1)
        self.assertIn("Normal", res["predicted_labels"])
        self.assertIn("Stain", res["predicted_labels"])

    def test_case_d_folded_and_stain(self):
        """Case D: Torn=0, Folded=1, Burnt=0, Stain=1 -> Normal=1 (NEW V3.1 rule!)."""
        ev = {"Torn": -0.01, "Folded": 0.01, "Burnt": -0.01, "Stain": 0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Torn"], 0)
        self.assertEqual(res["damage_labels"]["Folded"], 1)
        self.assertEqual(res["damage_labels"]["Burnt"], 0)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["damage_labels"]["Normal"], 1)
        self.assertIn("Normal", res["predicted_labels"])
        self.assertIn("Folded", res["predicted_labels"])
        self.assertIn("Stain", res["predicted_labels"])

    def test_case_e_torn_only(self):
        """Case E: Torn=1, Folded=0, Burnt=0, Stain=0 -> Normal=0."""
        ev = {"Torn": 0.01, "Folded": -0.01, "Burnt": -0.01, "Stain": -0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Torn"], 1)
        self.assertEqual(res["damage_labels"]["Folded"], 0)
        self.assertEqual(res["damage_labels"]["Burnt"], 0)
        self.assertEqual(res["damage_labels"]["Stain"], 0)
        self.assertEqual(res["damage_labels"]["Normal"], 0)
        self.assertEqual(res["predicted_labels"], ["Torn"])

    def test_case_f_burnt_and_stain(self):
        """Case F: Torn=0, Folded=0, Burnt=1, Stain=1 -> Normal=0."""
        ev = {"Torn": -0.01, "Folded": -0.01, "Burnt": 0.01, "Stain": 0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Torn"], 0)
        self.assertEqual(res["damage_labels"]["Folded"], 0)
        self.assertEqual(res["damage_labels"]["Burnt"], 1)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["damage_labels"]["Normal"], 0)
        self.assertEqual(set(res["predicted_labels"]), {"Burnt", "Stain"})

    def test_case_g_all_damages_active(self):
        """Case G: Torn=1, Folded=1, Burnt=1, Stain=1 -> Normal=0."""
        ev = {"Torn": 0.01, "Folded": 0.01, "Burnt": 0.01, "Stain": 0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Torn"], 1)
        self.assertEqual(res["damage_labels"]["Folded"], 1)
        self.assertEqual(res["damage_labels"]["Burnt"], 1)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["damage_labels"]["Normal"], 0)
        self.assertEqual(set(res["predicted_labels"]), {"Torn", "Folded", "Burnt", "Stain"})

    def test_case_h_folded_burnt_stain(self):
        """Case H: Torn=0, Folded=1, Burnt=1, Stain=1 -> Normal=0."""
        ev = {"Torn": -0.01, "Folded": 0.01, "Burnt": 0.01, "Stain": 0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Torn"], 0)
        self.assertEqual(res["damage_labels"]["Folded"], 1)
        self.assertEqual(res["damage_labels"]["Burnt"], 1)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["damage_labels"]["Normal"], 0)
        self.assertEqual(set(res["predicted_labels"]), {"Folded", "Burnt", "Stain"})

    # ----------------------------------------------------------------------
    # 2. 4 Burnt -> Stain Consistency Cases (Section 18)
    # ----------------------------------------------------------------------
    def test_burnt_stain_consistency_case_1(self):
        """Case 1: Burnt=1, raw_Stain=0 -> final_Stain=1."""
        ev = {"Torn": -0.01, "Folded": -0.01, "Burnt": 0.01, "Stain": -0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Burnt"], 1)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["raw_damage_labels"]["Stain"], 0)
        self.assertEqual(res["damage_labels"]["Normal"], 0)

    def test_burnt_stain_consistency_case_2(self):
        """Case 2: Burnt=1, raw_Stain=1 -> final_Stain=1."""
        ev = {"Torn": -0.01, "Folded": -0.01, "Burnt": 0.01, "Stain": 0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Burnt"], 1)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["raw_damage_labels"]["Stain"], 1)

    def test_burnt_stain_consistency_case_3(self):
        """Case 3: Burnt=0, raw_Stain=1 -> final_Stain=1."""
        ev = {"Torn": -0.01, "Folded": -0.01, "Burnt": -0.01, "Stain": 0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Burnt"], 0)
        self.assertEqual(res["damage_labels"]["Stain"], 1)
        self.assertEqual(res["raw_damage_labels"]["Stain"], 1)

    def test_burnt_stain_consistency_case_4(self):
        """Case 4: Burnt=0, raw_Stain=0 -> final_Stain=0."""
        ev = {"Torn": -0.01, "Folded": -0.01, "Burnt": -0.01, "Stain": -0.01}
        res = self.calc_v31.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Burnt"], 0)
        self.assertEqual(res["damage_labels"]["Stain"], 0)
        self.assertEqual(res["raw_damage_labels"]["Stain"], 0)

    # ----------------------------------------------------------------------
    # 3. Backward Compatibility with V3
    # ----------------------------------------------------------------------
    def test_v3_backward_compatibility(self):
        """Verifies that version='v3' maintains strict exclusivity (Folded=1 -> Normal=0)."""
        ev = {"Torn": -0.01, "Folded": 0.01, "Burnt": -0.01, "Stain": -0.01}
        res = self.calc_v3.apply_dynamic_thresholds(ev, self.dummy_context)
        self.assertEqual(res["damage_labels"]["Folded"], 1)
        self.assertEqual(res["damage_labels"]["Normal"], 0)
        self.assertNotIn("Normal", res["predicted_labels"])

    # ----------------------------------------------------------------------
    # 4. Interface Metadata Assertions
    # ----------------------------------------------------------------------
    def test_interface_metadata(self):
        """Verifies DamageClassifier exposes V3.1 metadata fields."""
        classifier = DamageClassifier(device="cpu", use_preprocessing=False)
        self.assertEqual(classifier.calibration_version, "v3.1")
        img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        out = classifier.classify(img)
        self.assertIn("raw_damage_labels", out)
        self.assertIn("final_damage_labels", out)
        self.assertIn("normal_rule", out)
        self.assertEqual(out["normal_rule"], "NOT(Burnt OR Torn)")
        self.assertTrue(out.get("burnt_implies_stain", False))


if __name__ == "__main__":
    unittest.main()
