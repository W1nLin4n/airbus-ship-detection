import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

def dice_coeff(y_true, y_pred, eps=1):
    """
    Calculates dice coefficient
    :param y_true: true values
    :param y_pred: predictions
    :param eps: smoothing parameter
    :return: dice coefficient
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + eps) / (union + eps), axis=0)
def dice_bce_loss(y_true, y_pred):
    """
    Combines binary crossentropy with dice coefficient
    :param y_true: true values
    :param y_pred: predictions
    :return: combined loss
    """
    return 0.02*binary_crossentropy(y_true, y_pred) - dice_coeff(y_true, y_pred)