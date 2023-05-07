# Configuration file for emotion recognition training

# Model and dataset settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 7
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5

# Path settings
TRAIN_DIR = "../data/train/"
TEST_DIR = "../data/test/"

# Model saving settings
MODEL_DIR = "../models/"
MODEL_NAME = "emotion_recognition_effnetb0.h5"
