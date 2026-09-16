"""
Unit tests verifying independent multi-label decisions without argmax.
"""
import unittest
from clip_damage_classifier.thresholds import apply_evidence_thresholds

class TestMultiLabel(unittest.TestCase):
    def test_independent_multi_label(self):
        thresholds = {"Torn": 0.0005, "Folded": 0.0005, "Burnt": 0.0015, "Stain": 0.0005}
        scores = {
            "torn_evidence_score": -0.010,
            "folded_evidence_score": +0.003,
            "burnt_evidence_score": +0.006,
            "stain_evidence_score": -0.008
        }
        res = apply_evidence_thresholds(scores, thresholds=thresholds)
        self.assertEqual(res["Folded"], 1)
        self.assertEqual(res["Burnt"], 1)
        self.assertEqual(res["Torn"], 0)
        self.assertEqual(res["Stain"], 0)
        self.assertEqual(res["Normal"], 0)
        self.assertEqual(res["predicted_labels"], "Folded; Burnt")

if __name__ == "__main__":
    unittest.main()
