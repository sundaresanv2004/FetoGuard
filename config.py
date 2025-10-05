import torch

# --- Project Paths ---
ANNO_PATH = './FPUS23_Dataset/Dataset/boxes/annotation'
IMG_PATH = './FPUS23_Dataset/Dataset/four_poses'
MODEL_SAVE_PATH = "fetus_unet_model.pth"

# --- Data & Dataloader Settings ---
BATCH_SIZE = 10
NUM_WORKERS = 0
IMAGE_SIZE = 256
TRAIN_SPLIT_RATIO = 0.8

# --- Model Settings ---
IN_CHANNELS = 3
NUM_CLASSES = 5

# --- Training Settings ---
LEARNING_RATE = 1e-4
EPOCHS = 25
DEVICE = "auto"
RANDOM_SEED = 42

# --- Resume Training Settings ---
LOAD_MODEL = True