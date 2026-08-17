# ============================================================
# denomination_predictor.py
# Reusable Currency Denomination Prediction Module
# ============================================================

import os
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "10",
    "100",
    "20",
    "200",
    "2000",
    "50",
    "500",
    "Background"
]


# ============================================================
# MODEL PATH
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

SRC_DIR = os.path.dirname(CURRENT_FILE)

DENOMINATION_DIR = os.path.dirname(SRC_DIR)

MODEL_PATH = os.path.join(
    DENOMINATION_DIR,
    "models",
    "best_improved_currency_efficientnetb0.keras"
)


# ============================================================
# LOAD MODEL
# ============================================================

_model = None


def load_model():
    """
    Load the trained EfficientNetB0 model.

    The model is loaded only once and then reused.
    This is important during final integration because we
    don't want to load the model every time we make a prediction.
    """

    global _model

    if _model is None:

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Trained model not found:\n{MODEL_PATH}"
            )

        print("Loading denomination recognition model...")

        _model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print(
            "Denomination recognition model loaded."
        )

    return _model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):
    """
    Prepare an image for EfficientNetB0.

    Parameters
    ----------
    image:
        Can be:
        - image file path
        - NumPy array
        - TensorFlow tensor

    Returns
    -------
    Tensor suitable for model prediction.
    """

    # --------------------------------------------------------
    # Case 1: image path
    # --------------------------------------------------------

    if isinstance(
        image,
        (str, os.PathLike)
    ):

        image = tf.keras.utils.load_img(
            image,
            target_size=IMAGE_SIZE
        )

        image = tf.keras.utils.img_to_array(
            image
        )

    # --------------------------------------------------------
    # Convert to TensorFlow tensor
    # --------------------------------------------------------

    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Ensure 3 colour channels
    # --------------------------------------------------------

    if len(image.shape) == 2:

        image = tf.stack(
            [image, image, image],
            axis=-1
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICT DENOMINATION
# ============================================================

def predict_denomination(
    image,
    confidence_threshold=0.0
):
    """
    Predict the denomination of an Indian currency note.

    Parameters
    ----------
    image:
        Image path, NumPy array, or TensorFlow tensor.

    confidence_threshold:
        Optional minimum confidence.

    Returns
    -------
    Dictionary containing:

        {
            "denomination": "200",
            "confidence": 0.98,
            "confidence_percent": 98.0,
            "class_index": 3,
            "is_background": False
        }
    """

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    processed_image = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Find highest probability
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    # --------------------------------------------------------
    # Background detection
    # --------------------------------------------------------

    is_background = (
        predicted_class == "Background"
    )

    # --------------------------------------------------------
    # Confidence threshold
    # --------------------------------------------------------

    accepted = (
        confidence >= confidence_threshold
    )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = {

        "denomination":
            predicted_class,

        "confidence":
            confidence,

        "confidence_percent":
            confidence * 100,

        "class_index":
            predicted_index,

        "is_background":
            is_background,

        "accepted":
            accepted
    }

    return result


# ============================================================
# TOP K PREDICTIONS
# ============================================================

def get_top_predictions(
    image,
    top_k=3
):
    """
    Return the top K predictions.

    Example result:

        [
            {
                "denomination": "200",
                "confidence": 0.98
            },
            {
                "denomination": "100",
                "confidence": 0.01
            }
        ]
    """

    model = load_model()

    processed_image = preprocess_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_k = min(
        top_k,
        len(CLASS_NAMES)
    )

    top_indices = np.argsort(
        predictions
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({

            "denomination":
                CLASS_NAMES[index],

            "confidence":
                float(predictions[index]),

            "confidence_percent":
                float(
                    predictions[index] * 100
                )
        })

    return results


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DENOMINATION PREDICTOR TEST")
    print("=" * 60)

    image_path = (
        "denomination/data/"
        "Indian currency dataset v1/"
        "test/200__1.jpg"
    )

    print("\nTesting image:")
    print(image_path)

    print("\nGenerating prediction...")

    result = predict_denomination(image_path)

    print("\nPrediction result:")
    print(result)

    print("\nPrediction:")
    print(result["denomination"])

    print(
        f"Confidence: "
        f"{result['confidence_percent']:.2f}%"
    )

    print(
        f"Background: "
        f"{result['is_background']}"
    )

    print("\nTop 3 predictions:")

    top_predictions = get_top_predictions(
        image_path,
        top_k=3
    )

    for i, prediction in enumerate(
        top_predictions,
        start=1
    ):

        print(
            f"{i}. "
            f"{prediction['denomination']} "
            f"→ "
            f"{prediction['confidence_percent']:.2f}%"
        )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)