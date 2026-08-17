from pathlib import Path
from PIL import Image

# --------------------------------------------------
# CHANGE THIS PATH
# --------------------------------------------------

DATASET_ROOT = Path(
    "/Users/yash/Documents/Yash/Symbiosis university/Msc CA sem 3/IMAGE PROCESSING /Currency-note-recognition/denomination/data/Indian currency dataset v1"
)

TRAIN_DIR = DATASET_ROOT / "training"
VAL_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"

CLASSES = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "2000",
    "Background",
]


# --------------------------------------------------
# Count images in training / validation folders
# --------------------------------------------------

def inspect_class_folders(directory):
    print(f"\n{'=' * 50}")
    print(f"DATASET: {directory.name.upper()}")
    print(f"{'=' * 50}")

    total = 0

    for class_name in CLASSES:
        class_dir = directory / class_name

        if not class_dir.exists():
            print(f"{class_name:12} : FOLDER NOT FOUND")
            continue

        images = [
            file for file in class_dir.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"{class_name:12} : {len(images)} images")

        total += len(images)

    print(f"{'-' * 50}")
    print(f"TOTAL        : {total} images")


# --------------------------------------------------
# Inspect test folder
# Test images have labels in their filenames
# Example: 10__206.jpg
# --------------------------------------------------

def inspect_test_folder(directory):
    print(f"\n{'=' * 50}")
    print("DATASET: TEST")
    print(f"{'=' * 50}")

    counts = {class_name: 0 for class_name in CLASSES}
    unknown = 0

    image_files = [
        file for file in directory.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    for file in image_files:

        # Example:
        # 10__206.jpg -> 10
        label = file.stem.split("__")[0]

        if label in counts:
            counts[label] += 1
        else:
            unknown += 1

    for class_name in CLASSES:
        print(f"{class_name:12} : {counts[class_name]} images")

    print(f"{'-' * 50}")
    print(f"TOTAL        : {len(image_files)} images")
    print(f"UNKNOWN      : {unknown} images")


# --------------------------------------------------
# Check images for corruption
# --------------------------------------------------

def check_corrupted_images(directory):
    print(f"\nChecking images in: {directory}")

    corrupted = []

    for file in directory.rglob("*"):

        if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            with Image.open(file) as image:
                image.verify()

        except Exception:
            corrupted.append(file)

    if corrupted:
        print("\nCORRUPTED IMAGES:")
        for file in corrupted:
            print(file)

    else:
        print("No corrupted images found.")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("\nINDIAN CURRENCY DATASET INSPECTION")

    inspect_class_folders(TRAIN_DIR)
    inspect_class_folders(VAL_DIR)
    inspect_test_folder(TEST_DIR)

    print("\n")
    check_corrupted_images(TRAIN_DIR)
    check_corrupted_images(VAL_DIR)
    check_corrupted_images(TEST_DIR)

    print("\nInspection complete.")