import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

# Ensure cache directories are writable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
METRICS_DIR = BASE_DIR / "metrics"

MODEL_PATH = MODEL_DIR / "sign_language_model.keras"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
HAND_LANDMARKER_MODEL = BASE_DIR / "hand_landmarker.task"

# Full 26 English Alphabets (A-Z)
CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

# MediaPipe Configuration
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.70
MIN_TRACKING_CONFIDENCE = 0.50

# Feature Extraction
# 21 hand landmarks, each has (x, y, z) = 63 features
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3

# Data Collection Parameters
SAMPLES_PER_CLASS = 150
COUNTDOWN_SECONDS = 3

# Model Training Hyperparameters
EPOCHS = 75
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Real-time Inference & UI Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.65
PREDICTION_SMOOTHING_WINDOW = 5   # Majority voting buffer to eliminate jitter
SENTENCE_DEBOUNCE_FRAMES = 12     # Hold stable sign for 12 frames (~0.4s) to commit letter
TOP_K_PREDICTIONS = 3             # Show top-3 probable signs on HUD
