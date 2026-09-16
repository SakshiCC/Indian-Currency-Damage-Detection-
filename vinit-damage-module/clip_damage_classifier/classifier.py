"""
classifier.py
============================================================
Main multi-label damage classification engine using
contrastive positive vs negative evidence scoring.
============================================================
"""

from PIL import Image, ImageEnhance
from typing import List, Dict, Any, Tuple
from .clip_model import get_clip_wrapper
from .scoring import compute_contrastive_scores
from .thresholds import apply_evidence_thresholds
from .metadata_parser import parse_banknote_metadata
from . import config


class DamageClassifier:
    def __init__(
        self,
        thresholds: Dict[str, float] = None,
        aggregation_method: str = config.PROMPT_AGGREGATION_METHOD,
        device: str = config.DEVICE_PREFERENCE,
        multi_view: bool = config.ENABLE_MULTI_VIEW
    ):
        self.thresholds = thresholds or config.INITIAL_EVIDENCE_THRESHOLDS
        self.aggregation_method = aggregation_method
        self.clip = get_clip_wrapper(device=device)
        self.multi_view = multi_view

    def process_batch(
        self,
        batch_items: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        valid_images = []
        valid_items = []
        failed_items = []

        for item in batch_items:
            img_path = item["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                valid_images.append(img)
                valid_items.append(item)
            except Exception as e:
                failed_items.append({
                    "image_path": img_path,
                    "relative_path": item["relative_path"],
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })

        if not valid_images:
            return [], failed_items

        # Encode images
        if self.multi_view:
            # Primary full view
            f1 = self.clip.encode_images(valid_images)
            # Secondary mildly contrast-enhanced full view (preserves all edges)
            enhanced = [ImageEnhance.Contrast(im).enhance(1.15) for im in valid_images]
            f2 = self.clip.encode_images(enhanced)
            image_feats = (f1 + f2) / 2.0
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        else:
            image_feats = self.clip.encode_images(valid_images)

        # Compute contrastive scores
        batch_scores = compute_contrastive_scores(
            image_feats,
            self.clip.positive_embeddings,
            self.clip.negative_embeddings,
            aggregation_method=self.aggregation_method
        )

        results = []
        for item, scores in zip(valid_items, batch_scores):
            decisions = apply_evidence_thresholds(
                scores,
                thresholds=self.thresholds,
                ambiguity_margin=config.AMBIGUITY_MARGIN
            )
            metadata = parse_banknote_metadata(item["relative_path"], item["filename"])

            record = {
                "image_path": item["image_path"],
                "relative_path": item["relative_path"],
                "filename": item["filename"],
                "denomination": metadata["denomination"],
                "note_generation": metadata["note_generation"],
                "Torn": decisions["Torn"],
                "Folded": decisions["Folded"],
                "Burnt": decisions["Burnt"],
                "Stain": decisions["Stain"],
                "Normal": decisions["Normal"],
                "predicted_labels": decisions["predicted_labels"],
                "clip_torn_positive_score": scores["clip_torn_positive_score"],
                "clip_torn_negative_score": scores["clip_torn_negative_score"],
                "torn_evidence_score": scores["torn_evidence_score"],
                "clip_folded_positive_score": scores["clip_folded_positive_score"],
                "clip_folded_negative_score": scores["clip_folded_negative_score"],
                "folded_evidence_score": scores["folded_evidence_score"],
                "clip_burnt_positive_score": scores["clip_burnt_positive_score"],
                "clip_burnt_negative_score": scores["clip_burnt_negative_score"],
                "burnt_evidence_score": scores["burnt_evidence_score"],
                "clip_stain_positive_score": scores["clip_stain_positive_score"],
                "clip_stain_negative_score": scores["clip_stain_negative_score"],
                "stain_evidence_score": scores["stain_evidence_score"],
                "classification_status": decisions["classification_status"]
            }
            results.append(record)

        return results, failed_items
