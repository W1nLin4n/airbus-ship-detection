import os

import numpy as np
from PIL import Image
from config import DEFAULT_HEIGHT, DEFAULT_WIDTH, PROCESSES
import multiprocessing as mp

def rle_encode_test(img: np.ndarray, threshold: float) -> str:
    """
    :param img: numpy array, > - mask, 2 - background
    :return: run length encoded mask as string
    """
    pixels = img.T.flatten()
    pixels = np.pad(pixels, 1)
    pixels = pixels > threshold
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_encode(img: np.ndarray) -> str:
    """
    :param img: numpy array, 1 - mask, 2 - background
    :return: run length encoded mask as string
    """
    pixels = img.T.flatten()
    pixels = np.pad(pixels, 1)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask: str, shape=(DEFAULT_WIDTH, DEFAULT_HEIGHT)) -> np.ndarray:
    """
    :param mask: run length encoded mask as string
    :param shape: shape of target
    :return: numpy array, 1 - mask, 2 - background
    """
    s = mask.split()
    starts, lengths = (np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2]))
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for l, h in zip(starts, ends):
        img[l:h] = 1
    return img.reshape(shape).T

def masks_combine(masks_list: list, shape=(DEFAULT_WIDTH, DEFAULT_HEIGHT)) -> np.ndarray:
    masks_combined = np.zeros(shape, dtype=np.int16)
    for mask in masks_list:
        if isinstance(mask, str):
            masks_combined += rle_decode(mask, shape)
    return masks_combined

def is_image_corrupted(path: str) -> bool:
    """
    Finds if image is corrupted
    :param path: full path to image
    :return: True if image is corrupted, otherwise False
    """
    try:
        img = Image.open(path)
        img.verify()
        img = np.asarray(Image.open(path))
    except:
        return True
    return False

def find_invalid_images(path: str) -> list:
    """
    Finds corrupted images
    :param path: path to folder
    :return: list of corrupted images
    """
    corrupted_images = []
    filenames = os.listdir(path)
    with mp.Pool(PROCESSES) as p:
        filenames_mask = p.map(is_image_corrupted, [path + "/" + filename for filename in filenames])
    for i, filename in enumerate(filenames):
        if filenames_mask[i]:
            corrupted_images.append(filename)
    return corrupted_images