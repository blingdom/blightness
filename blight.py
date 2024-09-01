#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (
    unicode_literals,
    print_function
    )

import os
import json
import keras
import random
import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from time import strftime
from tensorflow import data as tf_data


"""IMPORT DATA IMAGES"""
TESTDIR = "images"

# FILTER-OUT CORRUPTED IMAGES
"""
num_skipped = 0
print(f"Deleted {num_skipped} images.")
"""

"""Generate a Dataset"""
image_size = (228, 228)
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    TESTDIR + "/train",
    # validation_split=0.2,
    # subset="both",
    seed=1337,
    # label_mode="binary",
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = keras.utils.image_dataset_from_directory(
    TESTDIR + "/val",
    # validation_split=0.2,
    # subset="both",
    seed=1337,
    # label_mode="binary",
    image_size=image_size,
    batch_size=batch_size,
)

print(train_ds.class_names, val_ds.class_names)


"""Visualize the data"""
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
# plt.show()

"""Visualize the Validation"""
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
# plt.show()

"""Data Augmentation - Needed?"""
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


"""Visualize Augmented data"""
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")

plt.tight_layout()
# plt.show()

# print("""PreProcess - Option 1: Part of Model (GPU)""")
# inputs = keras.Input(shape=input_shape)
# x = data_augmentation(inputs)
# x = layers.Rescaling(1./255)(x)

"""Option 2: Apply to Dataset (CPU)"""
# augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))


"""Configure the dataset for performance"""
# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


def make_model(input_shape, num_classes):
    # print("""Build a model""", input_shape, num_classes)
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


"""Build a model"""
model = make_model(input_shape=image_size + (3,), num_classes=2)
model.summary()

model.save('models/InitMyModelTF.keras')

# Saving weights of model
model.save_weights('models/InitBlight.weights.h5') # tf format

"""
keras.utils.plot_model(
    model,
    to_file="blightlr.png",
    show_shapes=True,
    show_trainable=True,
    rankdir="LR"  # "TB"
    )
"""

"""Train the model"""
# """
epochs = 2  # 10

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/save_at_{epoch}.keras",
        # "tomato_weights.keras",
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
        ),
    keras.callbacks.EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=20,
        verbose=1,
        mode='auto'
        ),
]

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

# print("Start-Model-Fit", strftime('%d %b %Y %H:%M'))

hist = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    verbose=1
    )
# print("End-Model-Fit", strftime('%d %b %Y %H:%M'))
# """

# Saving model architecture to JSON file
model_json = model.to_json()

# Saving to local directory
with open('models/blightmodel.json','w') as json_file:
  json_file.write(model_json)

# model.save('models/MyModelh5.h5')
model.save('models/MyModelTF.keras')

# Saving weights of model
model.save_weights('models/blight.weights.h5') # h5 format
# model.save_weights('models/blightWeights.keras') # tf format

"""VISUALIZE THE TRAINING/VALIDATION DATA"""
# """
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()
# """