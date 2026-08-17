import os
import sys
import json
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "best_improved_currency_efficientnetb0.keras"
)

CLASS_NAMES_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "improved_class_names.json"
)


# ============================================================
# CHECK INPUT
# ============================================================

if len(sys.argv) != 2:

    print("\nUsage:")
    print("python denomination/src/predict.py <image_path>")

    print("\nExample:")
    print(
        "python denomination/src/predict.py "
        "denomination/data/Indian\\ currency\\ dataset\\ v1/test/500__233.jpg"
    )

    sys.exit(1)


IMAGE_PATH = sys.argv[1]


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if os.path.exists(CLASS_NAMES_PATH):

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as file:

        class_names = json.load(file)

else:

    class_names = [
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
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMAGE_SIZE
)

image_array = tf.keras.utils.img_to_array(
    image
)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating prediction...")

predictions = model.predict(
    image_array,
    verbose=0
)[0]


predicted_index = int(
    np.argmax(predictions)
)

predicted_class = class_names[
    predicted_index
]

confidence = float(
    predictions[predicted_index]
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 60)
print("CURRENCY DENOMINATION PREDICTION")
print("=" * 60)

print(
    f"Image       : {os.path.basename(IMAGE_PATH)}"
)

print(
    f"Prediction  : {predicted_class}"
)

print(
    f"Confidence  : {confidence * 100:.2f}%"
)


if predicted_class == "Background":

    print(
        "\nResult: No currency note detected."
    )

else:

    print(
        f"\nResult: ₹{predicted_class} note detected."
    )


# ============================================================
# TOP 3 PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("TOP 3 PREDICTIONS")
print("=" * 60)


top_indices = np.argsort(
    predictions
)[::-1][:3]


for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank}. "
        f"{class_names[index]:<12} "
        f"{predictions[index] * 100:.2f}%"
    )


print("\n")
print("=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)