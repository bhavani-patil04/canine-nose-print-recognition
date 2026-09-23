import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "dataset/pet_biometric_clean/finetune_train.csv"

IMAGE_FOLDER = "dataset/pet_biometric_clean/images"

MODEL_FOLDER = "models/dog_specific_resnet"

BEST_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "best_model.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "dog_specific_resnet.keras"
)

EMBEDDING_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "dog_embedding_model.keras"
)

LABEL_MAPPING_PATH = os.path.join(
    MODEL_FOLDER,
    "dog_label_mapping.csv"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS = 15

RANDOM_SEED = 42

VALIDATION_IMAGES_PER_DOG = 1


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========")

df = pd.read_csv(CSV_PATH)

print(
    "Total images:",
    len(df)
)

print(
    "Total dogs:",
    df["dog ID"].nunique()
)


# ============================================================
# CLEAN DATA TYPES
# ============================================================

df["dog ID"] = df["dog ID"].astype(str)

df["nose print image"] = (
    df["nose print image"]
    .astype(str)
)


# ============================================================
# CREATE DOG LABELS
# ============================================================

dog_ids = sorted(
    df["dog ID"].unique()
)

dog_to_label = {
    dog_id: index
    for index, dog_id in enumerate(dog_ids)
}

df["label"] = df["dog ID"].map(
    dog_to_label
)

NUM_CLASSES = len(dog_ids)

print(
    "Number of dog classes:",
    NUM_CLASSES
)


# ============================================================
# SAVE LABEL MAPPING
# ============================================================

label_mapping_df = pd.DataFrame({
    "label": list(range(NUM_CLASSES)),
    "dog_id": dog_ids
})

label_mapping_df.to_csv(
    LABEL_MAPPING_PATH,
    index=False
)

print(
    "Label mapping saved:",
    LABEL_MAPPING_PATH
)


# ============================================================
# VERIFY IMAGE FILES
# ============================================================

print("\n========== CHECKING IMAGES ==========")

missing_images = []

for image_name in df["nose print image"]:

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    if not os.path.exists(image_path):

        missing_images.append(
            image_name
        )


print(
    "Missing images:",
    len(missing_images)
)

if len(missing_images) > 0:

    print(
        "\nFirst missing images:"
    )

    for image in missing_images[:10]:
        print(image)

    raise FileNotFoundError(
        "Some images are missing."
    )


# ============================================================
# IMAGE-LEVEL TRAIN / VALIDATION SPLIT
#
# Each dog remains represented in TRAINING.
#
# Since every training dog has 2 or 3 images:
#
#   1 image -> validation
#   remaining images -> training
#
# This makes validation meaningful for the classifier.
# ============================================================

print(
    "\n========== CREATING VALIDATION SPLIT =========="
)

train_parts = []

val_parts = []


for dog_id, group in df.groupby(
    "dog ID",
    sort=False
):

    group = group.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Keep at least one image for training
    # --------------------------------------------------------

    if len(group) >= 2:

        val_part = group.iloc[
            :VALIDATION_IMAGES_PER_DOG
        ]

        train_part = group.iloc[
            VALIDATION_IMAGES_PER_DOG:
        ]

    else:

        train_part = group

        val_part = group.iloc[0:0]


    train_parts.append(
        train_part
    )

    val_parts.append(
        val_part
    )


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

val_df = pd.concat(
    val_parts,
    ignore_index=True
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

train_dogs = set(
    train_df["dog ID"]
)

val_dogs = set(
    val_df["dog ID"]
)

print(
    "Training images:",
    len(train_df)
)

print(
    "Validation images:",
    len(val_df)
)

print(
    "Training dogs:",
    len(train_dogs)
)

print(
    "Validation dogs:",
    len(val_dogs)
)


# ============================================================
# VERIFY EVERY DOG HAS TRAINING DATA
# ============================================================

missing_train_classes = (
    set(dog_ids) - train_dogs
)

if missing_train_classes:

    raise ValueError(
        "Some dog identities are missing from training."
    )


# ============================================================
# VERIFY VALIDATION DOGS ALSO EXIST IN TRAINING
# ============================================================

unseen_validation_dogs = (
    val_dogs - train_dogs
)

if unseen_validation_dogs:

    raise ValueError(
        "Validation contains unseen dog identities."
    )


print(
    "All validation dogs are represented in training."
)


# ============================================================
# CONVERT DATA TO NUMPY
# ============================================================

train_image_names = (
    train_df["nose print image"]
    .values
)

train_labels = (
    train_df["label"]
    .values
    .astype(np.int32)
)

val_image_names = (
    val_df["nose print image"]
    .values
)

val_labels = (
    val_df["label"]
    .values
    .astype(np.int32)
)


# ============================================================
# TENSORFLOW IMAGE LOADER
# ============================================================

def load_image(image_name, label):

    # image_name is a TensorFlow tensor here.
    #
    # Therefore we use tf.strings.join()
    # instead of os.path.join().

    image_path = tf.strings.join(
        [
            IMAGE_FOLDER,
            "/",
            image_name
        ]
    )

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # ResNet50 ImageNet preprocessing
    image = preprocess_input(
        image
    )

    return image, label


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

print(
    "\n========== CREATING DATASETS =========="
)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_image_names,
        train_labels
    )
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_image_names,
        val_labels
    )
)


# ============================================================
# SHUFFLE TRAINING DATA
# ============================================================

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_df),
    seed=RANDOM_SEED,
    reshuffle_each_iteration=True
)


# ============================================================
# LOAD / PREPROCESS IMAGES
# ============================================================

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# BATCH
# ============================================================

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)


# ============================================================
# PREFETCH
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD RESNET50
# ============================================================

print(
    "\n========== BUILDING RESNET50 =========="
)

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ============================================================
# FREEZE BACKBONE
#
# Stage 1:
# Train only the new dog-specific layers.
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = base_model(
    inputs,
    training=False
)

x = GlobalAveragePooling2D()(
    x
)


# ============================================================
# 512-D DOG EMBEDDING
# ============================================================

embedding = Dense(
    512,
    activation="relu",
    name="dog_embedding"
)(x)


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

x = Dropout(
    0.3
)(embedding)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax",
    name="dog_classifier"
)(x)


# ============================================================
# FINAL MODEL
# ============================================================

model = Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print(
    "\n========== MODEL =========="
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\n========== STARTING TRAINING =========="
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL_PATH
)

print(
    "\nFinal model saved:"
)

print(
    FINAL_MODEL_PATH
)

print(
    "\nBest model saved:"
)

print(
    BEST_MODEL_PATH
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print(
    "\n========== TRAINING SUMMARY =========="
)

history_dict = history.history

best_epoch = (
    np.argmax(
        history_dict["val_accuracy"]
    ) + 1
)

best_val_accuracy = max(
    history_dict["val_accuracy"]
)

best_train_accuracy = (
    history_dict["accuracy"][
        best_epoch - 1
    ]
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Training accuracy at best epoch:",
    round(
        best_train_accuracy,
        4
    )
)

print(
    "Validation accuracy at best epoch:",
    round(
        best_val_accuracy,
        4
    )
)


# ============================================================
# CREATE EMBEDDING MODEL
#
# This removes the classifier and keeps only
# the 512-dimensional dog-specific embedding.
# ============================================================

embedding_model = Model(
    inputs=model.input,
    outputs=model.get_layer(
        "dog_embedding"
    ).output
)


# ============================================================
# SAVE EMBEDDING MODEL
# ============================================================

embedding_model.save(
    EMBEDDING_MODEL_PATH
)

print(
    "\nEmbedding model saved:"
)

print(
    EMBEDDING_MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n========== TRAINING COMPLETE =========="
)