import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# Import our preprocessing pipeline
from preprocess import (
    train_dataset,
    validation_dataset,
    class_names,
    IMG_SIZE,
    data_augmentation
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = len(class_names)

DROPOUT_RATE = 0.30


# ============================================================
# BUILD EFFICIENTNETB0 MODEL
# ============================================================

def build_model():

    # --------------------------------------------------------
    # Load EfficientNetB0 pretrained on ImageNet
    # --------------------------------------------------------

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )

    # --------------------------------------------------------
    # Freeze the pretrained backbone
    # --------------------------------------------------------

    base_model.trainable = False


    # --------------------------------------------------------
    # Define model input
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=IMG_SIZE + (3,),
        name="input_image"
    )


    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    x = data_augmentation(inputs)


    # --------------------------------------------------------
    # EfficientNetB0 feature extraction
    # --------------------------------------------------------

    x = base_model(
        x,
        training=False
    )


    # --------------------------------------------------------
    # Global Average Pooling
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)


    # --------------------------------------------------------
    # Dropout
    # --------------------------------------------------------

    x = layers.Dropout(
        DROPOUT_RATE,
        name="dropout"
    )(x)


    # --------------------------------------------------------
    # Classification layer
    # --------------------------------------------------------

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification"
    )(x)


    # --------------------------------------------------------
    # Create final model
    # --------------------------------------------------------

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="currency_efficientnetb0"
    )


    return model


# ============================================================
# CREATE MODEL
# ============================================================

print("\nBuilding EfficientNetB0 model...")
print("=" * 60)

model = build_model()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# DISPLAY MODEL SUMMARY
# ============================================================

print("\nModel summary:")
print("=" * 60)

model.summary()


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nModel configuration:")
print("=" * 60)

print(f"Model name       : {model.name}")
print(f"Input size       : {IMG_SIZE}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Classes          : {class_names}")

print("\nBase model:")
print("EfficientNetB0")

print("\nPretrained weights:")
print("ImageNet")

print("\nTransfer learning:")
print("Enabled")

print("\nEfficientNetB0 backbone:")
print("Frozen")

print("\nClassification head:")
print("GlobalAveragePooling2D")
print(f"Dropout ({DROPOUT_RATE})")
print(f"Dense ({NUM_CLASSES} classes, softmax)")

print("\nOptimizer:")
print("Adam")

print("\nLearning rate:")
print("0.001")

print("\nLoss:")
print("Sparse Categorical Crossentropy")


# ============================================================
# TRAINABLE / NON-TRAINABLE PARAMETERS
# ============================================================

trainable_parameters = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

non_trainable_parameters = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.non_trainable_weights
)

print("\nParameters:")
print("=" * 60)

print(
    f"Trainable parameters     : "
    f"{trainable_parameters:,}"
)

print(
    f"Non-trainable parameters : "
    f"{non_trainable_parameters:,}"
)


# ============================================================
# TEST MODEL WITH ONE BATCH
# ============================================================

print("\nTesting model with one training batch...")
print("=" * 60)

for images, labels in train_dataset.take(1):

    predictions = model(images, training=False)

    print("Input shape       :", images.shape)
    print("Labels shape      :", labels.shape)
    print("Predictions shape :", predictions.shape)

    print(
        "Expected output shape:",
        f"(batch_size, {NUM_CLASSES})"
    )

    break


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nEfficientNetB0 model created successfully!")
print("=" * 60)

print("The model is ready for training.")