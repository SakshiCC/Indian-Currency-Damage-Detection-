"""
clip_damage_classifier
======================
Independent zero-shot multi-label damage classification pipeline for Indian banknotes.
Uses Hugging Face CLIP (openai/clip-vit-base-patch32) with prompt ensembles
and OpenCV note preprocessing.
"""

from clip_damage_classifier.interface import DamageClassifier

__version__ = "2.0.0"
__all__ = ["DamageClassifier"]
