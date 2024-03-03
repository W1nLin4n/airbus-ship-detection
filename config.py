# Path to train images directory
TRAIN_IMAGES_DIR = "/data/images"

# Path to train file
TRAIN_FILE = "/data/masks.csv"

# Path to test images directory
TEST_IMAGES_DIR = "/data/images"

# Path to test file
TEST_FILE = "/data/masks.csv"

# Path to model
MODEL_PATH = "/data/model.keras"

# How many samples should we expect from each class(in our case, class is an amount of boats in image)
SAMPLING_SIZE = 2000

# Default batch size
DEFAULT_BATCH=4

# Default dimensions
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 768
N_CHANNELS = 3

# Default batches per epoch
BATCHES_PER_EPOCH = (SAMPLING_SIZE * 20) // DEFAULT_BATCH

# Default batches for validation
BATCHES_PER_VALID = 1600 // DEFAULT_BATCH

# Default epochs
N_EPOCHS = 100

# Amount of subprocesses to use in preprocessing
PROCESSES = 8