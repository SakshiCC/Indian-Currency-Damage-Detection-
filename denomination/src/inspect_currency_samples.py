from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random


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
    "2000"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


fig, axes = plt.subplots(7, 3, figsize=(12, 24))


for row, class_name in enumerate(CLASSES):

    class_dir = TRAIN_DIR / class_name

    images = [
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    samples = random.sample(images, min(3, len(images)))

    for col, image_path in enumerate(samples):

        image = Image.open(image_path)

        axes[row, col].imshow(image)

        axes[row, col].set_title(
            f"₹{class_name} - {image_path.name}",
            fontsize=8
        )

        axes[row, col].axis("off")


plt.tight_layout()
plt.show()