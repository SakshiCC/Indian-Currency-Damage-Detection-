import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from model import model
from preprocess import (
    train_dataset,
    validation_dataset,
    class_names
)


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 20

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_currency_efficientnetb0.keras"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)

PLOT_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.png"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DISPLAY TRAINING CONFIGURATION
# ============================================================

print("\nTRAINING CONFIGURATION")
print("=" * 60)

print(f"Epochs           : {EPOCHS}")
print(f"Training images  : 3566")
print(f"Validation images: 345")
print(f"Classes          : {len(class_names)}")
print(f"Model            : EfficientNetB0")
print(f"Backbone         : Frozen")
print(f"Learning rate    : 0.001")

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# CALLBACKS
# ============================================================

# Save the model whenever validation accuracy improves
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


# Stop training if validation accuracy stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# Reduce learning rate if validation loss stops improving
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# START TRAINING
# ============================================================

print("\nStarting model training...")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# TRAINING COMPLETED
# ============================================================

print("\nTraining completed!")
print("=" * 60)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(HISTORY_PATH, "w") as file:
    json.dump(
        history_data,
        file,
        indent=4
    )


print(f"Training history saved to:")
print(HISTORY_PATH)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "w") as file:
    json.dump(
        class_names,
        file,
        indent=4
    )


print("\nClass mapping saved to:")
print(CLASS_NAMES_PATH)


# ============================================================
# FIND BEST EPOCH
# ============================================================

best_epoch = (
    max(
        range(len(history.history["val_accuracy"])),
        key=lambda i: history.history["val_accuracy"][i]
    )
)

best_val_accuracy = history.history["val_accuracy"][best_epoch]
best_val_loss = history.history["val_loss"][best_epoch]

print("\nBEST VALIDATION RESULT")
print("=" * 60)

print(f"Best epoch        : {best_epoch + 1}")
print(f"Validation accuracy: {best_val_accuracy:.4f}")
print(f"Validation loss    : {best_val_loss:.4f}")


# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nTraining graph saved to:")
print(PLOT_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nTRAINING SUMMARY")
print("=" * 60)

print(f"Best validation accuracy : {best_val_accuracy:.4f}")
print(f"Best validation loss     : {best_val_loss:.4f}")

print("\nBest model saved to:")
print(BEST_MODEL_PATH)

print("\nTraining history saved to:")
print(HISTORY_PATH)

print("\nClass names saved to:")
print(CLASS_NAMES_PATH)

print("\nTraining graph saved to:")
print(PLOT_PATH)

print("\nNext step:")
print("Evaluate the trained model on the test dataset.")