from config import DEFAULT_HEIGHT, DEFAULT_WIDTH, N_CHANNELS, MODEL_PATH, N_EPOCHS, DEFAULT_BATCH, BATCHES_PER_EPOCH, BATCHES_PER_VALID
from scripts.preprocessing import read_train_data, image_gen
from scripts.preprocessing import Augment
from scripts.model import build_model
from scripts.metrics import dice_bce_loss, dice_coeff
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.data import AUTOTUNE
import gc
gc.enable()


def main():
    # Configuring gpu
    gpus = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

    # Reading data
    print("Loading data")
    train_df, val_df = read_train_data()

    # Creating train dataset
    print("Creating train samples generator")
    train_generator = image_gen(train_df)
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH, N_CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH, 1), dtype=tf.float32)
        )
    )
    train_dataset = train_dataset.map(Augment()).repeat().batch(DEFAULT_BATCH,
                                                                num_parallel_calls=AUTOTUNE,
                                                                deterministic=False).prefetch(AUTOTUNE)

    # Creating validation dataset
    print("Creating validation samples generator")
    val_generator = image_gen(val_df, valid=True)
    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH, N_CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH, 1), dtype=tf.float32)
        )
    )
    val_dataset = val_dataset.repeat().batch(DEFAULT_BATCH,
                                             num_parallel_calls=AUTOTUNE,
                                             deterministic=False).prefetch(AUTOTUNE)

    # Creating model
    print("Building model")
    model = build_model()
    model.summary()
    model.compile(optimizer=Adam(), loss=dice_bce_loss, metrics=[dice_coeff, "binary_accuracy"])

    # Setting up checkpoints system
    checkpoint = ModelCheckpoint(MODEL_PATH,
                                 monitor='val_dice_coeff',
                                 verbose=1,
                                 save_best_only=True,
                                 mode="max")

    # Setting up lr reduction
    lr_reduction = ReduceLROnPlateau(monitor='val_dice_coeff',
                                     factor=0.3,
                                     patience=2,
                                     verbose=1,
                                     mode="max",
                                     min_lr=1e-6)
    early_stop = EarlyStopping(monitor="val_dice_coeff",
                               mode="max",
                               verbose=1,
                               patience=5,
                               restore_best_weights=True)

    print("Starting training")
    model.fit(train_dataset,
              epochs=N_EPOCHS,
              steps_per_epoch=BATCHES_PER_EPOCH,
              verbose=1,
              callbacks=[checkpoint, lr_reduction, early_stop],
              validation_data=val_dataset,
              validation_steps=BATCHES_PER_VALID)


if __name__ == "__main__":
    main()