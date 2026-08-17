import os
import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

# Path to the dataset
DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Indian currency dataset v1"
)

# EfficientNetB0 expects 224 x 224 RGB images
IMG_SIZE = (224, 224)

# Number of images processed at once
BATCH_SIZE = 32

# Random seed for reproducibility
SEED = 42


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DIR = os.path.join(DATASET_DIR, "training")
VALIDATION_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")
print("=" * 60)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# GET CLASS INFORMATION
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes: {len(class_names)}")


# ============================================================
# CREATE CLASS-TO-INDEX MAPPING
# ============================================================

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

print("\nClass mapping:")
print(class_to_index)


# ============================================================
# LOAD TEST DATASET
# ============================================================
#
# IMPORTANT:
# The test folder is different from training/validation.
#
# Test images are directly inside the test folder and their
# denomination is stored in the filename:
#
# 10__276.jpg
# 20__324.jpg
# 200__1.jpg
# 2000__8.jpg
# Background__123.jpg
#
# Therefore, we manually extract the label from the filename.
# ============================================================

print("\nLoading test dataset...")
print("=" * 60)


# Get all image files from test directory
test_file_paths = []

allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

for filename in os.listdir(TEST_DIR):

    if filename.lower().endswith(allowed_extensions):

        full_path = os.path.join(TEST_DIR, filename)

        if os.path.isfile(full_path):
            test_file_paths.append(full_path)


# Sort files for reproducibility
test_file_paths.sort()


# Extract labels from filenames
test_labels = []

for file_path in test_file_paths:

    filename = os.path.basename(file_path)

    # Example:
    # "200__1.jpg" -> "200"
    # "2000__8.jpg" -> "2000"
    # "Background__5.jpg" -> "Background"

    class_name = filename.split("__")[0]

    if class_name not in class_to_index:
        raise ValueError(
            f"Unknown class '{class_name}' found in test file: {filename}"
        )

    test_labels.append(class_to_index[class_name])


# Convert labels to TensorFlow tensors
test_labels = tf.constant(test_labels, dtype=tf.int32)


# ============================================================
# TEST IMAGE LOADING FUNCTION
# ============================================================

def load_test_image(file_path, label):

    image = tf.io.read_file(file_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# Create TensorFlow test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_file_paths, test_labels)
)

test_dataset = test_dataset.map(
    load_test_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# ============================================================
# DATASET PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# TEST THE TRAINING DATA PIPELINE
# ============================================================

print("\nTesting training dataset pipeline...")
print("=" * 60)

for images, labels in train_dataset.take(1):

    print("Image batch shape :", images.shape)
    print("Label batch shape :", labels.shape)
    print("Image data type   :", images.dtype)

    print(
        "Pixel value range :",
        images.numpy().min(),
        "to",
        images.numpy().max()
    )


# ============================================================
# TEST THE TEST DATA PIPELINE
# ============================================================

print("\nTesting test dataset pipeline...")
print("=" * 60)

for images, labels in test_dataset.take(1):

    print("Image batch shape :", images.shape)
    print("Label batch shape :", labels.shape)
    print("Image data type   :", images.dtype)

    print(
        "Pixel value range :",
        images.numpy().min(),
        "to",
        images.numpy().max()
    )


# ============================================================
# SUMMARY
# ============================================================

print("\nPreprocessing pipeline ready!")
print("=" * 60)

print(f"Image size        : {IMG_SIZE}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Number of classes : {len(class_names)}")
print(f"Classes           : {class_names}")

print("\nDataset sizes:")
print(f"Training images   : 3566")
print(f"Validation images : 345")
print(f"Test images       : {len(test_file_paths)}")

print("\nDatasets:")
print("Training   : Ready")
print("Validation : Ready")
print("Test       : Ready")

print("\nData augmentation:")
print("Random rotation")
print("Random zoom")
print("Random translation")
print("Random contrast")

print("\nEfficientNetB0 input:")
print("224 × 224 × 3")

print("\nPreprocessing completed successfully!")