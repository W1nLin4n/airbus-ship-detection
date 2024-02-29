import os

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from PIL import Image
from config import MODEL_PATH, TEST_IMAGES_DIR, TEST_FILE, DEFAULT_BATCH
from scripts.utils import rle_encode_test
from scripts.model import build_model
import gc
gc.enable()

def main():
    # Loading test data
    print("Loading test data")
    test_images_filenames = os.listdir(TEST_IMAGES_DIR)

    # Initializing model
    print("Initializing model")
    model = build_model()
    model.load_weights(MODEL_PATH)

    # Generating predictions
    print("Generating predictions")
    predictions = []
    images = []
    for filename in test_images_filenames:
        image = np.asarray(Image.open(TEST_IMAGES_DIR + "/" + filename))
        images += [image]
        if len(images) >= DEFAULT_BATCH:
            batch_prediction = model.predict(np.asarray(images))
            batch_prediction = np.squeeze(batch_prediction)
            batch_prediction = [rle_encode_test(batch_prediction[i], threshold=0.5)
                                for i in range(batch_prediction.shape[0])]
            predictions += batch_prediction
            images = []
    if len(images) != 0:
        batch_prediction = model.predict(np.asarray(images))
        batch_prediction = np.squeeze(batch_prediction)
        batch_prediction = [rle_encode_test(batch_prediction[i], threshold=0.5)
                            for i in range(batch_prediction.shape[0])]
        predictions += batch_prediction

    # Writing predictions in system
    print("Saving predictions")
    data = pd.DataFrame({"ImageId": test_images_filenames, "EncodedPixels": predictions})
    data.to_csv(TEST_FILE, index=False)

if __name__ == "__main__":
    main()