import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib and Keras config dirs are writable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import config
from model.train import load_dataset, plot_confusion_matrix

def evaluate_model():
    print("==================================================")
    print("       SIGN LANGUAGE MODEL EVALUATION SUITE       ")
    print("==================================================")

    if not Path(config.MODEL_PATH).exists():
        print(f"[ERROR] Model file not found at '{config.MODEL_PATH}'. Please train the model first.")
        return

    if not Path(config.LABEL_MAP_PATH).exists():
        print(f"[ERROR] Label map not found at '{config.LABEL_MAP_PATH}'.")
        return

    with open(config.LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    inv_label_map = {int(v): k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in range(len(label_map))]

    # Load dataset
    X, y, _, _ = load_dataset()
    num_classes = len(label_map)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    print(f"[INFO] Loading trained model from {config.MODEL_PATH}...")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    loss, acc = model.evaluate(X, y_cat, verbose=0)
    print(f"\n[RESULTS] Evaluation on full dataset ({len(X)} samples):")
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(f"  Loss:     {loss:.4f}")

    y_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    print("\nDetailed Classification Report:")
    report = classification_report(y, y_pred, target_names=class_names)
    print(report)

    cm = confusion_matrix(y, y_pred)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    eval_cm_path = config.METRICS_DIR / "eval_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, eval_cm_path)
    print(f"[SAVED] Confusion matrix plot saved to: {eval_cm_path}")

if __name__ == "__main__":
    evaluate_model()
