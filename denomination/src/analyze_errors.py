# ============================================================
# analyze_errors.py
# Error Analysis for Improved Currency Recognition Model
# ============================================================

import os
import json
import shutil
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "Indian currency dataset v1"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "best_improved_currency_efficientnetb0.keras"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

MISCLASSIFIED_DIR = os.path.join(
    RESULTS_DIR,
    "improved_misclassified"
)

os.makedirs(
    MISCLASSIFIED_DIR,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

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
NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: name
    for index, name in enumerate(CLASS_NAMES)
}


SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif"
)


# ============================================================
# 3. DISPLAY INFORMATION
# ============================================================

print("=" * 60)
print("IMPROVED MODEL ERROR ANALYSIS")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nTest directory:")
print(TEST_DIR)

print("\nClasses:")
for index, name in enumerate(CLASS_NAMES):
    print(f"{index}: {name}")


# ============================================================
# 4. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nImproved model not found:\n"
        + MODEL_PATH
    )


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("\n")
print("=" * 60)
print("LOADING IMPROVED MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 6. FIND TEST IMAGES
# ============================================================

print("\n")
print("=" * 60)
print("LOADING TEST IMAGES")
print("=" * 60)


test_image_paths = []
actual_labels = []


for filename in sorted(
    os.listdir(TEST_DIR)
):

    if not filename.lower().endswith(
        SUPPORTED_EXTENSIONS
    ):
        continue

    full_path = os.path.join(
        TEST_DIR,
        filename
    )

    if not os.path.isfile(full_path):
        continue

    # --------------------------------------------------------
    # Extract denomination from filename
    #
    # Example:
    #
    # 50__183.jpg
    #
    # denomination = 50
    # --------------------------------------------------------

    if "__" not in filename:

        print(
            "WARNING: Cannot determine class:",
            filename
        )

        continue

    class_name = filename.split(
        "__"
    )[0]


    if class_name not in CLASS_TO_INDEX:

        print(
            "WARNING: Unknown class:",
            class_name,
            filename
        )

        continue


    label = CLASS_TO_INDEX[
        class_name
    ]


    test_image_paths.append(
        full_path
    )

    actual_labels.append(
        label
    )


actual_labels = np.array(
    actual_labels,
    dtype=np.int32
)


print(
    "Test images found:",
    len(test_image_paths)
)


# ============================================================
# 7. LOAD IMAGE FUNCTION
# ============================================================

def load_image(path):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)


predicted_labels = []
prediction_confidences = []


for index, image_path in enumerate(
    test_image_paths
):

    image = load_image(
        image_path
    )

    image = tf.expand_dims(
        image,
        axis=0
    )


    prediction = model.predict(
        image,
        verbose=0
    )[0]


    predicted_class = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_class]
    )


    predicted_labels.append(
        predicted_class
    )

    prediction_confidences.append(
        confidence
    )


predicted_labels = np.array(
    predicted_labels,
    dtype=np.int32
)


prediction_confidences = np.array(
    prediction_confidences,
    dtype=np.float32
)


print(
    "Predictions generated:",
    len(predicted_labels)
)


# ============================================================
# 9. FIND ERRORS
# ============================================================

correct_mask = (
    predicted_labels == actual_labels
)

incorrect_mask = (
    predicted_labels != actual_labels
)


correct_count = int(
    np.sum(correct_mask)
)

incorrect_count = int(
    np.sum(incorrect_mask)
)


accuracy = (
    correct_count /
    len(actual_labels)
)


print("\n")
print("=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

print(
    "Total test images :",
    len(actual_labels)
)

print(
    "Correct           :",
    correct_count
)

print(
    "Incorrect         :",
    incorrect_count
)

print(
    "Accuracy          :",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 10. DISPLAY MISCLASSIFICATIONS
# ============================================================

print("\n")
print("=" * 60)
print("MISCLASSIFIED IMAGES")
print("=" * 60)


errors = []


error_number = 0


for i in range(
    len(test_image_paths)
):

    if predicted_labels[i] == actual_labels[i]:
        continue


    error_number += 1


    actual_class = INDEX_TO_CLASS[
        actual_labels[i]
    ]

    predicted_class = INDEX_TO_CLASS[
        predicted_labels[i]
    ]

    confidence = prediction_confidences[i]


    filename = os.path.basename(
        test_image_paths[i]
    )


    print("\nError", error_number)

    print(
        "Image     :",
        filename
    )

    print(
        "Actual    :",
        actual_class
    )

    print(
        "Predicted :",
        predicted_class
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )


    error_information = {

        "image": filename,

        "image_path":
            test_image_paths[i],

        "actual":
            actual_class,

        "predicted":
            predicted_class,

        "confidence":
            float(confidence),

        "confidence_percent":
            float(confidence * 100)
    }


    errors.append(
        error_information
    )


    # --------------------------------------------------------
    # Copy misclassified image into results folder
    # --------------------------------------------------------

    destination_filename = (
        f"error_{error_number:02d}_"
        f"actual_{actual_class}_"
        f"predicted_{predicted_class}_"
        f"{filename}"
    )


    destination_path = os.path.join(
        MISCLASSIFIED_DIR,
        destination_filename
    )


    shutil.copy2(
        test_image_paths[i],
        destination_path
    )


# ============================================================
# 11. ERROR SUMMARY BY ACTUAL CLASS
# ============================================================

print("\n")
print("=" * 60)
print("ERROR SUMMARY BY ACTUAL CLASS")
print("=" * 60)


errors_by_class = {}


for class_name in CLASS_NAMES:

    class_index = CLASS_TO_INDEX[
        class_name
    ]

    class_errors = np.sum(
        (
            actual_labels == class_index
        )
        &
        (
            predicted_labels != class_index
        )
    )

    errors_by_class[class_name] = int(
        class_errors
    )


    print(
        f"{class_name:<12}: "
        f"{class_errors} error(s)"
    )


# ============================================================
# 12. CONFUSION PAIRS
# ============================================================

print("\n")
print("=" * 60)
print("CONFUSION PAIRS")
print("=" * 60)


confusion_pairs = {}


for i in range(
    len(test_image_paths)
):

    if predicted_labels[i] == actual_labels[i]:
        continue


    actual_class = INDEX_TO_CLASS[
        actual_labels[i]
    ]

    predicted_class = INDEX_TO_CLASS[
        predicted_labels[i]
    ]


    pair = (
        f"{actual_class} → "
        f"{predicted_class}"
    )


    if pair not in confusion_pairs:

        confusion_pairs[pair] = 0


    confusion_pairs[pair] += 1


if len(confusion_pairs) == 0:

    print(
        "No misclassifications found!"
    )

else:

    for pair, count in confusion_pairs.items():

        print(
            f"{pair}: "
            f"{count} image(s)"
        )


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(
        range(NUM_CLASSES)
    )
)


print(cm)


# ============================================================
# 14. FIND LOW-CONFIDENCE CORRECT PREDICTIONS
# ============================================================
#
# These are images the model got correct but wasn't very
# confident about.
#
# They are useful for future dataset improvement.
#
# ============================================================

print("\n")
print("=" * 60)
print("LOW-CONFIDENCE CORRECT PREDICTIONS")
print("=" * 60)


LOW_CONFIDENCE_THRESHOLD = 0.70


low_confidence = []


for i in range(
    len(test_image_paths)
):

    if predicted_labels[i] != actual_labels[i]:
        continue


    confidence = prediction_confidences[i]


    if confidence < LOW_CONFIDENCE_THRESHOLD:

        actual_class = INDEX_TO_CLASS[
            actual_labels[i]
        ]

        predicted_class = INDEX_TO_CLASS[
            predicted_labels[i]
        ]


        low_confidence.append({

            "image":
                os.path.basename(
                    test_image_paths[i]
                ),

            "actual":
                actual_class,

            "predicted":
                predicted_class,

            "confidence":
                float(confidence),

            "confidence_percent":
                float(confidence * 100)
        })


        print(
            os.path.basename(
                test_image_paths[i]
            )
        )

        print(
            "Actual    :",
            actual_class
        )

        print(
            "Predicted :",
            predicted_class
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )

        print()


if len(low_confidence) == 0:

    print(
        "No correct predictions below "
        f"{LOW_CONFIDENCE_THRESHOLD * 100:.0f}% confidence."
    )


# ============================================================
# 15. SAVE ERROR ANALYSIS REPORT
# ============================================================

report = {

    "model":
        "best_improved_currency_efficientnetb0.keras",

    "total_test_images":
        int(len(actual_labels)),

    "correct_predictions":
        correct_count,

    "incorrect_predictions":
        incorrect_count,

    "accuracy":
        float(accuracy),

    "accuracy_percent":
        float(accuracy * 100),

    "errors":
        errors,

    "errors_by_actual_class":
        errors_by_class,

    "confusion_pairs":
        confusion_pairs,

    "confusion_matrix":
        cm.tolist(),

    "low_confidence_correct_predictions":
        low_confidence
}


REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "improved_error_analysis.json"
)


with open(
    REPORT_PATH,
    "w"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)

print(
    "Test accuracy:",
    f"{accuracy * 100:.2f}%"
)

print(
    "Total errors:",
    incorrect_count
)

print("\nMisclassified images saved to:")
print(MISCLASSIFIED_DIR)

print("\nError analysis report saved to:")
print(REPORT_PATH)

print("\nConfusion matrix:")
print(cm)

print("\n")
print("=" * 60)
print("NEXT STEP")
print("=" * 60)

if incorrect_count == 0:

    print(
        "Excellent! No test errors were found."
    )

else:

    print(
        "Inspect the images inside:"
    )

    print(
        MISCLASSIFIED_DIR
    )

    print(
        "\nThese are the images the improved "
        "model still struggles with."
    )

print("=" * 60)