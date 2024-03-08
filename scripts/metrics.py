import keras.ops as O
from keras.losses import binary_crossentropy
from config import BCE_FACTOR

def dice_coeff(y_true, y_pred, eps=1):
    """
    Calculates dice coefficient
    :param y_true: true values
    :param y_pred: predictions
    :param eps: smoothing parameter
    :return: dice coefficient
    """
    intersection = O.sum(y_true * y_pred, axis=[1, 2, 3])
    union = O.sum(y_true, axis=[1, 2, 3]) + O.sum(y_pred, axis=[1, 2, 3])
    return O.mean((2. * intersection + eps) / (union + eps), axis=0)
def dice_bce_loss(y_true, y_pred):
    """
    Combines binary crossentropy with dice coefficient
    :param y_true: true values
    :param y_pred: predictions
    :return: combined loss
    """
    return BCE_FACTOR*binary_crossentropy(y_true, y_pred) - O.log(dice_coeff(y_true, y_pred))