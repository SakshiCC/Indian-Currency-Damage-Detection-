"""
clip_model.py
============================================================
Loads Hugging Face CLIP (openai/clip-vit-base-patch32).
Pre-encodes both positive and negative prompt ensembles and caches them.
============================================================
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from typing import Dict, List
from . import config
from .prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS


class CLIPModelWrapper:
    _instance = None

    def __init__(self, model_name: str = config.CLIP_MODEL_NAME, device: str = config.DEVICE_PREFERENCE):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[CLIP] Initializing {model_name} on device: {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.positive_embeddings: Dict[str, torch.Tensor] = {}
        self.negative_embeddings: Dict[str, torch.Tensor] = {}
        self._encode_contrastive_ensembles()

    def _encode_contrastive_ensembles(self):
        """Encode positive and negative prompt ensembles and cache normalized embeddings."""
        print("[CLIP] Pre-encoding and caching contrastive (pos/neg) prompt embeddings...")
        with torch.no_grad():
            for cat in config.DAMAGE_CATEGORIES:
                # Positive
                inp_pos = self.processor(text=POSITIVE_PROMPTS[cat], return_tensors="pt", padding=True)
                inp_pos = {k: v.to(self.device) for k, v in inp_pos.items()}
                pos_feats = self.model.get_text_features(**inp_pos)
                if not isinstance(pos_feats, torch.Tensor):
                    pos_feats = getattr(pos_feats, "pooler_output", pos_feats[0])
                self.positive_embeddings[cat] = pos_feats / pos_feats.norm(dim=-1, keepdim=True)

                # Negative
                inp_neg = self.processor(text=NEGATIVE_PROMPTS[cat], return_tensors="pt", padding=True)
                inp_neg = {k: v.to(self.device) for k, v in inp_neg.items()}
                neg_feats = self.model.get_text_features(**inp_neg)
                if not isinstance(neg_feats, torch.Tensor):
                    neg_feats = getattr(neg_feats, "pooler_output", neg_feats[0])
                self.negative_embeddings[cat] = neg_feats / neg_feats.norm(dim=-1, keepdim=True)

        print("[CLIP] Contrastive prompt embeddings cached successfully.")

    def encode_images(self, images: List) -> torch.Tensor:
        """Encode a batch of PIL Images and return L2-normalized features."""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_feats = self.model.get_image_features(**inputs)
            if not isinstance(image_feats, torch.Tensor):
                image_feats = getattr(image_feats, "pooler_output", image_feats[0])
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        return image_feats


def get_clip_wrapper(model_name: str = config.CLIP_MODEL_NAME, device: str = config.DEVICE_PREFERENCE) -> CLIPModelWrapper:
    if CLIPModelWrapper._instance is None:
        CLIPModelWrapper._instance = CLIPModelWrapper(model_name=model_name, device=device)
    return CLIPModelWrapper._instance
