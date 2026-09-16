"""
yash_adapter.py
============================================================
Read-Only, Failure-Safe Adapter for Yash's Denomination Predictor.

Yash's module (src/denomination_predictor.py) requires TensorFlow/Keras
and models/best_improved_currency_efficientnetb0.keras.

This adapter:
  1. Does NOT modify any code in src/ or models/.
  2. Safely probes for TensorFlow and model existence.
  3. Returns real predictions when TensorFlow is available.
  4. Returns a structured, non-crashing status when TensorFlow is absent.
============================================================
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger("integration.yash_adapter")

# Ensure project root is in sys.path so src can be imported read-only
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class YashDenominationAdapter:
    """
    Adapter wrapping Yash's denomination predictor without altering src/.
    """

    def __init__(self, confidence_threshold: float = 0.0):
        self.confidence_threshold = confidence_threshold
        self._predictor_func = None
        self._available = False
        self._load_error: Optional[str] = None
        self._initialize()

    def _initialize(self):
        """Probes and imports Yash's predictor if dependencies exist."""
        try:
            # Check model file exists
            model_path = PROJECT_ROOT / "models" / "best_improved_currency_efficientnetb0.keras"
            if not model_path.exists():
                self._available = False
                self._load_error = f"Model file not found at {model_path}"
                logger.warning(self._load_error)
                return

            # Check TensorFlow import
            import tensorflow as tf  # noqa: F401
            from src.denomination_predictor import predict_denomination

            self._predictor_func = predict_denomination
            self._available = True
            logger.info("Yash denomination predictor initialized successfully.")
        except ImportError as e:
            self._available = False
            self._load_error = (
                f"TensorFlow / Keras dependency not available in current Python environment: {e}"
            )
            logger.info(self._load_error)
        except Exception as e:
            self._available = False
            self._load_error = f"Failed to initialize Yash predictor: {e}"
            logger.warning(self._load_error)

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """
        Run denomination prediction safely.
        Adapts Yash's output schema into a clean, typed response.
        """
        if not self._available or self._predictor_func is None:
            return {
                "available": False,
                "reason": self._load_error or "Predictor not initialized",
                "denomination": "Unknown",
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "class_index": -1,
                "is_background": False,
                "accepted": False,
            }

        try:
            # Pass image input directly to Yash's function
            raw_res = self._predictor_func(
                image_input, confidence_threshold=self.confidence_threshold
            )
            return {
                "available": True,
                "denomination": raw_res.get("denomination", "Unknown"),
                "confidence": float(raw_res.get("confidence", 0.0)),
                "confidence_percent": float(raw_res.get("confidence_percent", 0.0)),
                "class_index": int(raw_res.get("class_index", -1)),
                "is_background": bool(raw_res.get("is_background", False)),
                "accepted": bool(raw_res.get("accepted", True)),
                "raw": raw_res,
            }
        except Exception as e:
            logger.error(f"Error during denomination prediction: {e}")
            return {
                "available": False,
                "reason": f"Execution error: {e}",
                "denomination": "Unknown",
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "class_index": -1,
                "is_background": False,
                "accepted": False,
            }


# Singleton adapter instance
_DEFAULT_YASH_ADAPTER: Optional[YashDenominationAdapter] = None


def get_yash_adapter() -> YashDenominationAdapter:
    global _DEFAULT_YASH_ADAPTER
    if _DEFAULT_YASH_ADAPTER is None:
        _DEFAULT_YASH_ADAPTER = YashDenominationAdapter()
    return _DEFAULT_YASH_ADAPTER


def predict_denomination_safe(
    image_input: Union[str, Path, np.ndarray, Image.Image]
) -> Dict[str, Any]:
    """Module-level helper to predict denomination safely."""
    return get_yash_adapter().predict(image_input)


def is_yash_module_available() -> bool:
    """Check if Yash's denomination model can be executed."""
    return get_yash_adapter().is_available
