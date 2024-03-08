import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from PIL import Image
from scripts.utils import find_invalid_images, masks_combine, rle_decode, rle_encode
from config import TRAIN_IMAGES_DIR, TRAIN_FILE, DEFAULT_BATCH, SAMPLING_SIZE

class Augment(layers.Layer):
    """
    This class is used to augment training data
    """

    # The same seed is used everywhere so augmentations on inputs and labels stay consistent
    def __init__(self, seed=42):
        super().__init__()

        # Defining inputs augmentations
        self.augment_inputs = keras.Sequential(
            layers=[
                layers.Rescaling(1./255),
                layers.RandomFlip("horizontal_and_vertical", seed=seed),
                layers.RandomRotation(0.1, seed=seed),
                layers.RandomZoom((-0.2, 0.1), (-0.2, 0.1), seed=seed)
            ],
            name="Input_augmentations"
        )

        # Defining labels augmentations
        self.augment_labels = keras.Sequential(
            layers=[
                layers.RandomFlip("horizontal_and_vertical", seed=seed),
                layers.RandomRotation(0.1, seed=seed),
                layers.RandomZoom((-0.2, 0.1), (-0.2, 0.1), seed=seed),
                layers.Lambda(lambda x: tf.where(x > 0.5, 1., 0.))
            ],
            name="Labels_augmentations"
        )

    def __call__(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

def sample_by_ships(df: pd.DataFrame, samples=SAMPLING_SIZE):
    """
    :param df: dataframe from where samples should be taken
    :param samples: amount of samples to take
    :return: samples taken
    """
    return df.sample(samples, replace=True)

def image_gen(df: pd.DataFrame, valid=False) -> (np.ndarray, np.ndarray):
    """
    Generates data for training/validation
    :param df: samples data
    :param valid: is validation data used, or training
    :param batch_size: size of one batch
    :return: batch of data samples
    """
    def gen():
        # In case of validation, sampling should be normal, but when training - over/undersampling is used to
        # avoid imbalanced data
        if valid:
            balanced_df = list(df.groupby("ImageId"))
        else:
            balanced_df = df.groupby("ships").apply(sample_by_ships)
            balanced_df = list(balanced_df.groupby("ImageId"))
        while True:
            # Iterating over selected samples
            for img_id, masks in balanced_df:
                rgb = np.asarray(Image.open(TRAIN_IMAGES_DIR + "/" + img_id))
                mask = np.expand_dims(rle_decode(masks["EncodedPixels"].values[0]), -1)
                yield rgb, mask
            if valid:
                break

    return gen

def read_train_data(train_images_folder = TRAIN_IMAGES_DIR, train_file = TRAIN_FILE) -> (pd.DataFrame, pd.DataFrame):
    """
    Reads train data from provided sources
    :param train_images_folder: folder with train images
    :param train_file: file with masks
    :return: dataset split into train and validation
    """
    # Reading masks data
    print("Reading masks data")
    masks = pd.read_csv(train_file)

    # Excluding corrupted images if they are present in dataset
    print("Looking for corrupt images")
    corrupted_images = find_invalid_images(train_images_folder)
    masks.drop(masks[masks["ImageId"].isin(corrupted_images)].index, inplace=True)

    # Counting ships on each image
    print("Counting ships on images")
    masks["ships"] = masks["EncodedPixels"].map(lambda enc: 1 if isinstance(enc, str) else 0)
    unique_img_ids = masks.groupby("ImageId").agg({"ships": "sum"}).reset_index()

    # Combining masks and dropping unnecessary data
    print("Remasking images by combining separate parts")
    masks.drop(["ships"], axis=1, inplace=True)
    masks = masks.groupby("ImageId").apply(lambda x: rle_encode(masks_combine(x["EncodedPixels"].values))).reset_index()
    masks.columns = ["ImageId", "EncodedPixels"]

    # Splitting dataset into training and validation with stratification over ship count
    print("Splitting dataset")
    train_ids, val_ids = train_test_split(unique_img_ids, test_size=0.1, stratify=unique_img_ids["ships"])
    train_df, val_df = pd.merge(masks, train_ids), pd.merge(masks, val_ids)

    return train_df, val_df