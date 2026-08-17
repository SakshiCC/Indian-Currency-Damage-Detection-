from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

# --------------------------------------------------
# Dataset location
# --------------------------------------------------

DATASET_ROOT = Path(
    "/Users/yash/Documents/Yash/Symbiosis university/Msc CA sem 3/IMAGE PROCESSING /Currency-note-recognition/denomination/data/Indian currency dataset v1"
)

TRAIN_DIR = DATASET_ROOT / "training"

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------
# Get image files
# --------------------------------------------------

def get_images(class_dir):
    return [
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]


# --------------------------------------------------
# Display one sample from each class
# --------------------------------------------------

def display_samples():

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for ax, class_name in zip(axes.flat, CLASSES):

        class_dir = TRAIN_DIR / class_name
        images = get_images(class_dir)

        image_path = random.choice(images)

        image = Image.open(image_path)

        ax.imshow(image)
        ax.set_title(class_name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Analyze image dimensions
# --------------------------------------------------

def analyze_dimensions():

    print("\nIMAGE DIMENSION ANALYSIS")
    print("=" * 60)

    for class_name in CLASSES:

        class_dir = TRAIN_DIR / class_name
        images = get_images(class_dir)

        widths = []
        heights = []

        for image_path in images:

            with Image.open(image_path) as image:

                width, height = image.size

                widths.append(width)
                heights.append(height)

        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)

        print(f"\n{class_name}")

        print(f"Images       : {len(images)}")
        print(f"Min size     : {min(widths)} × {min(heights)}")
        print(f"Max size     : {max(widths)} × {max(heights)}")
        print(f"Average size : "
              f"{avg_width:.1f} × {avg_height:.1f}")


# --------------------------------------------------
# Class distribution graph
# --------------------------------------------------

def plot_class_distribution():

    counts = []

    for class_name in CLASSES:

        class_dir = TRAIN_DIR / class_name

        images = get_images(class_dir)

        counts.append(len(images))

    plt.figure(figsize=(10, 6))

    plt.bar(CLASSES, counts)

    plt.xlabel("Currency Class")
    plt.ylabel("Number of Images")
    plt.title("Training Dataset Class Distribution")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    display_samples()

    analyze_dimensions()

    plot_class_distribution()