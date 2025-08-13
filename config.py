# config.py
import os

# -----------------------
# Google Drive save directory
# -----------------------
DRIVE_SAVE_DIR = "/content/drive/MyDrive/tr_ocr/saved_models"

# -----------------------
# Dataset paths (relative to cloned repo in Colab)
# -----------------------
DATASET_DIR = "./modi_dataset"
ORG_IMAGES_DIR = os.path.join(DATASET_DIR, "org_images")
SYNTHETIC_IMAGES_DIR = os.path.join(DATASET_DIR, "sys_images")
IMAGE_LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# -----------------------
# Model paths
# -----------------------
MODEL_NAME = "microsoft/trocr-small-handwritten"
FREEZE_CHECKPOINT = "/content/drive/MyDrive/tr_ocr/saved_models"

# -----------------------
# Training hyperparameters
# -----------------------
BATCH_SIZE = 4
NUM_EPOCHS_PHASE1 = 5
NUM_EPOCHS_PHASE2 = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
LOGGING_STEPS = 50

# -----------------------
# Image preprocessing
# -----------------------
IMAGE_SIZE = (384, 384)  # TrOCR default
