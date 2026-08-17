from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random


DATASET_ROOT = Path(
    "/Users/yash/Documents/Yash/Symbiosis university/Msc CA sem 3/IMAGE PROCESSING /Currency-note-recognition/denomination/data/Indian currency dataset v1"
)

BACKGROUND_DIR = DATASET_ROOT / "training" / "Background"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


images = [
    file
    for file in BACKGROUND_DIR.iterdir()
    if file.suffix.lower() in IMAGE_EXTENSIONS
]


samples = random.sample(images, min(12, len(images)))


fig, axes = plt.subplots(3, 4, figsize=(16, 10))


for ax, image_path in zip(axes.flat, samples):

    image = Image.open(image_path)

    ax.imshow(image)

    ax.set_title(image_path.name, fontsize=8)

    ax.axis("off")


plt.tight_layout()
plt.show()