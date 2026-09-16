"""
scoring.py
============================================================
Contrastive scoring module.
Calculates positive similarity, negative similarity, and relative
evidence score (pos - neg) independently per damage category.
NEVER uses argmax or softmax for single-label decisions.
============================================================
"""

import torch
from typing import Dict, List
from . import config


def compute_contrastive_scores(
    image_features: torch.Tensor,
    positive_embeddings: Dict[str, torch.Tensor],
    negative_embeddings: Dict[str, torch.Tensor],
    aggregation_method: str = config.PROMPT_AGGREGATION_METHOD
) -> List[Dict[str, float]]:
    """
    Computes contrastive scores for each image in batch:
      clip_<cat>_positive_score
      clip_<cat>_negative_score
      <cat>_evidence_score = positive_score - negative_score
    """
    batch_size = image_features.shape[0]
    batch_scores: List[Dict[str, float]] = [{} for _ in range(batch_size)]

    for cat in config.DAMAGE_CATEGORIES:
        cat_lower = cat.lower()
        pos_feats = positive_embeddings[cat]
        neg_feats = negative_embeddings[cat]

        # Positive similarity matrix: (B, P_pos)
        sim_pos = torch.matmul(image_features, pos_feats.T)
        # Negative similarity matrix: (B, P_neg)
        sim_neg = torch.matmul(image_features, neg_feats.T)

        if aggregation_method == "max":
            pos_agg, _ = torch.max(sim_pos, dim=-1)
            neg_agg, _ = torch.max(sim_neg, dim=-1)
        else:  # default mean
            pos_agg = torch.mean(sim_pos, dim=-1)
            neg_agg = torch.mean(sim_neg, dim=-1)

        evidence = pos_agg - neg_agg

        pos_list = pos_agg.cpu().tolist()
        neg_list = neg_agg.cpu().tolist()
        ev_list = evidence.cpu().tolist()

        for i in range(batch_size):
            batch_scores[i][f"clip_{cat_lower}_positive_score"] = round(float(pos_list[i]), 4)
            batch_scores[i][f"clip_{cat_lower}_negative_score"] = round(float(neg_list[i]), 4)
            batch_scores[i][f"{cat_lower}_evidence_score"] = round(float(ev_list[i]), 4)

    return batch_scores
