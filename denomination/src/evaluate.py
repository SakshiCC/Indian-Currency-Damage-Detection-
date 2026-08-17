import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

from preprocess import test_dataset


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_currency_efficientnetb0.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

CLASSIFICATION_REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "evaluation_metrics.json"
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


NUM_CLASSES = len(class_names)


# ============================================================
# DISPLAY EVALUATION CONFIGURATION
# ============================================================

print("\nMODEL EVALUATION")
print("=" * 60)

print(f"Model path   : {MODEL_PATH}")
print(f"Test images  : 91")
print(f"Classes      : {NUM_CLASSES}")

print("\nClass mapping:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions on test dataset...")
print("=" * 60)

y_true = []
y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


# Convert to NumPy arrays

y_true = np.array(y_true)
y_pred = np.array(y_pred)


print(f"Number of actual labels     : {len(y_true)}")
print(f"Number of predicted labels  : {len(y_pred)}")


# ============================================================
# CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


# ============================================================
# CALCULATE PRECISION
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# CALCULATE RECALL
# ============================================================

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# CALCULATE F1 SCORE
# ============================================================

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# DISPLAY OVERALL METRICS
# ============================================================

print("\nOVERALL TEST RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    zero_division=0
)


print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    CLASSIFICATION_REPORT_PATH,
    "w"
) as file:

    file.write(
        "INDIAN CURRENCY DENOMINATION MODEL\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)


print("\nCONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "Indian Currency Denomination - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1),
    "test_images": int(len(y_true))
}

with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nEVALUATION COMPLETE")
print("=" * 60)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print("\nFiles generated:")

print(
    f"Confusion matrix:\n{CONFUSION_MATRIX_PATH}"
)

print(
    f"\nClassification report:\n{CLASSIFICATION_REPORT_PATH}"
)

print(
    f"\nEvaluation metrics:\n{METRICS_PATH}"
)