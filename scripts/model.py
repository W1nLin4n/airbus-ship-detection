import tensorflow as tf
from tensorflow.keras import models, layers
from config import DEFAULT_BATCH, DEFAULT_HEIGHT, DEFAULT_WIDTH, N_CHANNELS


def build_model():
    # Default input option
    input = layers.Input(shape=[DEFAULT_HEIGHT, DEFAULT_WIDTH, N_CHANNELS])
    input = layers.BatchNormalization()(input)

    # U-Net layers

    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(input)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D()(c4)

    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(p4)
    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(c5)

    u6 = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(c7)

    u8 = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(c8)

    u9 = layers.Conv2DTranspose(16, 3, strides=2, padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(c9)

    output = layers.Conv2D(1, 1, activation="sigmoid")(c9)

    return models.Model(inputs=[input], outputs=[output])