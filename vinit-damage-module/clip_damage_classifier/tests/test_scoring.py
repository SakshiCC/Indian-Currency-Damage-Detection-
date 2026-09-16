"""
Unit tests for contrastive scoring.
"""
import unittest
import torch
from clip_damage_classifier.scoring import compute_contrastive_scores

class TestScoring(unittest.TestCase):
    def test_compute_contrastive_scores(self):
        img_feats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        pos_embeddings = {
            "Torn": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "Folded": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "Burnt": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            "Stain": torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        }
        neg_embeddings = {
            "Torn": torch.tensor([[0.5, 0.5, 0.0, 0.0]]),
            "Folded": torch.tensor([[0.5, 0.5, 0.0, 0.0]]),
            "Burnt": torch.tensor([[0.5, 0.5, 0.0, 0.0]]),
            "Stain": torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        }
        scores = compute_contrastive_scores(img_feats, pos_embeddings, neg_embeddings)
        self.assertEqual(len(scores), 1)
        # Torn: pos = 1.0, neg = 0.5, evidence = 0.5
        self.assertAlmostEqual(scores[0]["clip_torn_positive_score"], 1.0, places=2)
        self.assertAlmostEqual(scores[0]["clip_torn_negative_score"], 0.5, places=2)
        self.assertAlmostEqual(scores[0]["torn_evidence_score"], 0.5, places=2)

if __name__ == "__main__":
    unittest.main()
