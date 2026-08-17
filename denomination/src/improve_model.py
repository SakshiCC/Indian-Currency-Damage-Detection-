# ============================================================
# improve_model.py
# Improved Indian Currency Note Recognition
#
# Model:
# EfficientNetB0 + Transfer Learning + Fine Tuning
#
# Dataset structure:
#
# data/
# └── Indian currency dataset v1/
#     ├── train/
#     │   ├── 10/
#     │   ├── 20/
#     │   ├── 50/
#     │   ├── 100/
#     │   ├── 200/
#     │   ├── 500/
#     │   ├── 2000/
#     │   └── Background/
#     │
#     ├── validation/
#     │   ├── 10/
#     │   ├── 20/
#     │   └── ...
#     │
#     └── test/
#         ├── 2000__8.jpg
#         ├── 50__183.jpg
#         ├── 20__324.jpg
#         └── ...
#
# Test labels are extracted from filenames.
# Example:
# 2000__8.jpg -> class = 2000
# 50__183.jpg -> class = 50
# ============================================================


import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

print("=" * 60)
print("IMPROVED CURRENCY RECOGNITION MODEL")
print("=" * 60)

# ------------------------------------------------------------
# Find project root automatically
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "Indian currency dataset v1"
)

TRAIN_DIR = os.path.join(DATASET_DIR, "training")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. CHECK DATASET PATHS
# ============================================================

print("\nDataset paths")
print("=" * 60)

print("Project directory:")
print(PROJECT_DIR)

print("\nDataset directory:")
print(DATASET_DIR)

print("\nTraining directory:")
print(TRAIN_DIR)

print("\nValidation directory:")
print(VAL_DIR)

print("\nTest directory:")
print(TEST_DIR)


if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"\nDataset directory does not exist:\n{DATASET_DIR}"
    )

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"\nTraining directory does not exist:\n{TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"\nValidation directory does not exist:\n{VAL_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"\nTest directory does not exist:\n{TEST_DIR}"
    )


# ============================================================
# 3. CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

NUM_CLASSES = 8

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

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: name
    for index, name in enumerate(CLASS_NAMES)
}


print("\nConfiguration")
print("=" * 60)

print("Image size       :", IMAGE_SIZE)
print("Batch size       :", BATCH_SIZE)
print("Number of classes:", NUM_CLASSES)
print("Classes          :", CLASS_NAMES)


# ============================================================
# 4. GPU / CPU INFORMATION
# ============================================================

print("\nHardware")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")

print("GPU devices:", gpus)
print("CPU devices:", cpus)

if gpus:
    print("GPU available.")
else:
    print("No GPU detected.")
    print("Training will use CPU.")


# ============================================================
# 5. LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# ============================================================
# 6. LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")
print("=" * 60)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 7. PRINT DATASET INFORMATION
# ============================================================

print("\nDataset information")
print("=" * 60)

print("Training images  :", len(train_dataset.file_paths))
print("Validation images:", len(validation_dataset.file_paths))


# ============================================================
# 8. BUILD TEST DATASET FROM FLAT DIRECTORY
# ============================================================
#
# IMPORTANT:
#
# image_dataset_from_directory() cannot be used here because
# test images are not inside class folders.
#
# Example:
#
# test/
#   2000__8.jpg
#   50__183.jpg
#   20__324.jpg
#
# The label is extracted from the filename.
#
# 2000__8.jpg -> 2000
# 50__183.jpg  -> 50
#
# ============================================================

print("\nLoading test dataset...")
print("=" * 60)


SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif"
)


test_image_paths = []
test_labels = []


for filename in sorted(os.listdir(TEST_DIR)):

    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        continue

    full_path = os.path.join(TEST_DIR, filename)

    if not os.path.isfile(full_path):
        continue

    # --------------------------------------------------------
    # Extract class from filename
    #
    # Example:
    # 2000__8.jpg
    #
    # split("__") gives:
    # ["2000", "8.jpg"]
    # --------------------------------------------------------

    if "__" not in filename:
        print(
            "WARNING: Could not determine class from:",
            filename
        )
        continue

    class_name = filename.split("__")[0]

    if class_name not in CLASS_TO_INDEX:
        print(
            "WARNING: Unknown class:",
            class_name,
            "from file:",
            filename
        )
        continue

    label = CLASS_TO_INDEX[class_name]

    test_image_paths.append(full_path)
    test_labels.append(label)


test_labels = np.array(test_labels, dtype=np.int32)


if len(test_image_paths) == 0:
    raise ValueError(
        f"\nNo valid test images found in:\n{TEST_DIR}\n"
        "\nExpected filenames such as:\n"
        "2000__8.jpg\n"
        "50__183.jpg\n"
        "20__324.jpg"
    )


print("Test images:", len(test_image_paths))


# ============================================================
# 9. TEST IMAGE LOADING FUNCTION
# ============================================================

def load_test_image(path, label):
    """
    Load and resize one test image.
    """

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# 10. CREATE TEST TF.DATA DATASET
# ============================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_image_paths,
        test_labels
    )
)

test_dataset = test_dataset.map(
    load_test_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 11. TEST DATASET PIPELINE
# ============================================================

print("\nTesting dataset pipeline...")
print("=" * 60)

for images, labels in test_dataset.take(1):

    print("Test image batch shape :", images.shape)
    print("Test label batch shape :", labels.shape)
    print("Image data type        :", images.dtype)
    print(
        "Pixel value range      :",
        float(tf.reduce_min(images)),
        "to",
        float(tf.reduce_max(images))
    )


# ============================================================
# 12. DATA AUGMENTATION
# ============================================================

print("\nCreating data augmentation...")
print("=" * 60)


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.15
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomContrast(
            0.15
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# 13. LOAD EFFICIENTNETB0
# ============================================================

print("\nLoading EfficientNetB0...")
print("=" * 60)


base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


# ============================================================
# 14. FREEZE BACKBONE FOR INITIAL TRAINING
# ============================================================

base_model.trainable = False


# ============================================================
# 15. BUILD IMPROVED MODEL
# ============================================================

print("\nBuilding improved model...")
print("=" * 60)


inputs = tf.keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="input_image"
)


x = data_augmentation(inputs)


# EfficientNetB0 expects image values in the normal
# 0-255 range in the Keras implementation.
x = base_model(
    x,
    training=False
)


x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


x = tf.keras.layers.BatchNormalization(
    name="head_batch_normalization"
)(x)


x = tf.keras.layers.Dropout(
    0.35,
    name="dropout_1"
)(x)


x = tf.keras.layers.Dense(
    256,
    activation="relu",
    name="dense_256"
)(x)


x = tf.keras.layers.Dropout(
    0.30,
    name="dropout_2"
)(x)


outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="classification"
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="improved_currency_efficientnetb0"
)


# ============================================================
# 16. COMPILE MODEL - STAGE 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 17. MODEL SUMMARY
# ============================================================

print("\nModel summary")
print("=" * 60)

model.summary()


# ============================================================
# 18. CLASS WEIGHTS
# ============================================================
#
# Class weights help when some denominations have fewer
# training images than others.
#
# ============================================================

print("\nCalculating class weights...")
print("=" * 60)


train_labels_for_weights = []


for _, labels in train_dataset.unbatch():

    train_labels_for_weights.append(
        int(labels.numpy())
    )


train_labels_for_weights = np.array(
    train_labels_for_weights
)


unique_classes = np.unique(
    train_labels_for_weights
)


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=train_labels_for_weights
)


class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        unique_classes,
        class_weights_array
    )
}


print("Class weights:")

for class_id in sorted(class_weights):

    print(
        f"{class_id}: "
        f"{INDEX_TO_CLASS[class_id]} -> "
        f"{class_weights[class_id]:.4f}"
    )


# ============================================================
# 19. CALLBACKS
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_improved_currency_efficientnetb0.keras"
)


CHECKPOINT_PATH = os.path.join(
    MODELS_DIR,
    "checkpoint_improved_currency.keras"
)


callbacks_stage1 = [

    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        os.path.join(
            RESULTS_DIR,
            "improved_training_log.csv"
        )
    )
]


# ============================================================
# 20. STAGE 1 TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 1 - TRANSFER LEARNING")
print("=" * 60)

print("EfficientNetB0 backbone: FROZEN")
print("Learning rate: 0.001")


history_stage1 = model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=15,

    class_weight=class_weights,

    callbacks=callbacks_stage1
)


# ============================================================
# 21. FINE TUNING
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 2 - FINE TUNING")
print("=" * 60)


# Make backbone trainable
base_model.trainable = True


# ------------------------------------------------------------
# Freeze most of EfficientNetB0.
#
# Only the final layers are fine-tuned.
# This reduces the risk of destroying the ImageNet features.
# ------------------------------------------------------------

FINE_TUNE_FROM = 180


for layer in base_model.layers:

    layer.trainable = False


for layer in base_model.layers[FINE_TUNE_FROM:]:

    # Keep BatchNormalization layers frozen.
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

    else:
        layer.trainable = True


trainable_backbone_layers = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)


print(
    "Trainable EfficientNet layers:",
    trainable_backbone_layers
)


# ============================================================
# 22. RECOMPILE WITH SMALL LEARNING RATE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


print("\nFine-tuning learning rate: 0.00001")


# ============================================================
# 23. FINE-TUNING CALLBACKS
# ============================================================

callbacks_stage2 = [

    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# 24. STAGE 2 TRAINING
# ============================================================

history_stage2 = model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=20,

    class_weight=class_weights,

    callbacks=callbacks_stage2
)


# ============================================================
# 25. COMBINE TRAINING HISTORY
# ============================================================

history = {}

for key in history_stage1.history:

    history[key] = (
        history_stage1.history[key]
        + history_stage2.history.get(key, [])
    )


# ============================================================
# 26. SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    RESULTS_DIR,
    "improved_training_history.json"
)


with open(
    history_path,
    "w"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )


# ============================================================
# 27. FIND BEST VALIDATION RESULT
# ============================================================

val_accuracy = history["val_accuracy"]
val_loss = history["val_loss"]


best_epoch = int(
    np.argmax(val_accuracy)
)


best_val_accuracy = float(
    val_accuracy[best_epoch]
)


best_val_loss = float(
    val_loss[best_epoch]
)


print("\n")
print("=" * 60)
print("BEST VALIDATION RESULT")
print("=" * 60)

print(
    "Best epoch         :",
    best_epoch + 1
)

print(
    "Validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print(
    "Validation loss    :",
    f"{best_val_loss:.4f}"
)


# ============================================================
# 28. LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")
print("=" * 60)


best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


print("Best model loaded successfully.")


# ============================================================
# 29. GENERATE TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")
print("=" * 60)


test_predictions = best_model.predict(
    test_dataset,
    verbose=1
)


predicted_labels = np.argmax(
    test_predictions,
    axis=1
)


actual_labels = test_labels


# ============================================================
# 30. TEST ACCURACY
# ============================================================

test_accuracy = np.mean(
    predicted_labels == actual_labels
)


print("\n")
print("=" * 60)
print("IMPROVED MODEL TEST RESULT")
print("=" * 60)

print(
    "Test images:",
    len(actual_labels)
)

print(
    "Test accuracy:",
    f"{test_accuracy:.4f}"
)

print(
    "Test accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 31. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


report = classification_report(
    actual_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "improved_classification_report.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(report)


# ============================================================
# 32. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 33. SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(cm)

plt.title(
    "Improved Model - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "improved_confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 34. TRAINING GRAPH
# ============================================================

epochs_range = range(
    1,
    len(history["accuracy"]) + 1
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs_range,
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Improved Model - Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


accuracy_graph_path = os.path.join(
    RESULTS_DIR,
    "improved_accuracy.png"
)


plt.savefig(
    accuracy_graph_path,
    dpi=200
)


plt.close()


# ============================================================
# 35. LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs_range,
    history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Improved Model - Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


loss_graph_path = os.path.join(
    RESULTS_DIR,
    "improved_loss.png"
)


plt.savefig(
    loss_graph_path,
    dpi=200
)


plt.close()


# ============================================================
# 36. SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    MODELS_DIR,
    "improved_class_names.json"
)


with open(
    class_names_path,
    "w"
) as file:

    json.dump(
        CLASS_NAMES,
        file,
        indent=4
    )


# ============================================================
# 37. SAVE METRICS
# ============================================================

metrics = {

    "model": "EfficientNetB0",

    "image_size": list(IMAGE_SIZE),

    "number_of_classes": NUM_CLASSES,

    "classes": CLASS_NAMES,

    "training_images": len(
        train_dataset.file_paths
    ),

    "validation_images": len(
        validation_dataset.file_paths
    ),

    "test_images": len(
        test_image_paths
    ),

    "best_epoch": best_epoch + 1,

    "best_validation_accuracy":
        best_val_accuracy,

    "best_validation_loss":
        best_val_loss,

    "test_accuracy":
        float(test_accuracy),

    "test_accuracy_percent":
        float(test_accuracy * 100),

    "transfer_learning": True,

    "fine_tuning": True,

    "fine_tune_from_layer":
        FINE_TUNE_FROM,

    "stage_1_learning_rate":
        0.001,

    "stage_2_learning_rate":
        0.00001
}


metrics_path = os.path.join(
    RESULTS_DIR,
    "improved_model_metrics.json"
)


with open(
    metrics_path,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# 38. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("IMPROVED MODEL TRAINING COMPLETE")
print("=" * 60)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Test accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print("\nModel saved to:")
print(BEST_MODEL_PATH)

print("\nTraining history saved to:")
print(history_path)

print("\nClassification report saved to:")
print(report_path)

print("\nConfusion matrix saved to:")
print(confusion_matrix_path)

print("\nAccuracy graph saved to:")
print(accuracy_graph_path)

print("\nLoss graph saved to:")
print(loss_graph_path)

print("\nMetrics saved to:")
print(metrics_path)

print("\nClass names saved to:")
print(class_names_path)

print("\n")
print("=" * 60)
print("NEXT STEP")
print("=" * 60)

print(
    "Compare the improved test accuracy with your original "
    "91.21% result."
)

print("=" * 60)