# airbus-ship-detection
This app can be used to train/test a UNet, built to recognize ships on
images.

## Environment
### .env
You should provide a .env configuration file with paths to necessary data.

You should provide following values:
- TRAIN_DIR - directory with training images
- TRAIN_FILE - csv file with train segmentation encodings
- TEST_DIR - directory with testing images
- TEST_FILE - csv file with test segmentation encodings
- PARAMETERS_FILE - file with model parameters

For correct behavior ensure that you have all these files/directories created on 
your local machine, otherwise nothing will work.
TEST_FILE and PARAMETERS_FILE can/will be edited, depending on which
setup you ran.

### config.py
Some configuration constants are provided in this file. Their purpose is described
in comments, so if you need to change some parameters, feel free to do so.

## Setup
### Commons
Docker container inside a project uses tensorflow image as it's base
and it is better to pull it before building project.

To pull this image use `docker pull tensorflow/tensorflow:nightly-gpu-jupyter`

### Jupyter
To start a jupyter server, use `docker compose -f compose.jupyter.yaml up --build` 
inside project directory. After server starts, you will see a link in console. Copy 
and paste into browser to see contents.

### Train
To start training, use `docker compose -f compose.train.yaml up --build` inside project directory.
Model weights will be automatically saved during and after training.

### Test
To start testing, use `docker compose -f compose.test.yaml up --build` inside project directory.
It will automatically generate a file with image segmentation encodings

## Solution
This app uses a simplified UNet architecture to find ships on images.

Solution pipeline looks like this:
- Reading masks data from csv file
- Validating images, with broken images being dropped from dataset
- Balancing training dataset around amount of ships on image
- Applying augmentations on training data
- Building UNet model and training it, using dice score and bce
- Saving weights of model to file

Result is quite good for such a small network:
- Private score(93% of test data): 0.76566
- Public score(7% of test data): 0.5209

From these we can understand, that the model generalized pretty well.