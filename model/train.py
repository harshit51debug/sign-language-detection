import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib and Keras config dirs are writable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

import config
from model.architecture import build_sign_classifier

def load_dataset():
    """
    Loads all .npy feature vectors from config.DATA_DIR subfolders.
    Returns:
        X: np.ndarray of shape (N, 63)
        y: np.ndarray of shape (N,) integer class indices
        label_map: dict mapping class name to integer index
    """
    data_dir = Path(config.DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please collect data or run generate_sample_data.py first.")

    features_list = []
    labels_list = []
    
    # Sort classes for deterministic mapping
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in '{data_dir}'.")

    label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
    inv_label_map = {idx: cls_name for cls_name, idx in label_map.items()}

    for cls_name in classes:
        cls_folder = data_dir / cls_name
        sample_files = list(cls_folder.glob("*.npy"))
        for sf in sample_files:
            try:
                feat = np.load(sf)
                if feat.shape == (config.FEATURE_DIM,):
                    features_list.append(feat)
                    labels_list.append(label_map[cls_name])
            except Exception as e:
                print(f"[WARN] Error reading {sf}: {e}")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    
    print(f"[DATASET] Loaded {len(X)} samples across {len(classes)} classes: {classes}")
    return X, y, label_map, inv_label_map

def plot_training_history(history, save_path):
    """Saves loss and accuracy curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", lw=2, linestyle="--")
    axes[0].set_title("Model Accuracy across Epochs", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Loss Plot
    axes[1].plot(history.history["loss"], label="Train Loss", lw=2, color="orange")
    axes[1].plot(history.history["val_loss"], label="Val Loss", lw=2, linestyle="--", color="red")
    axes[1].set_title("Model Loss across Epochs", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved training history plot to {save_path}")

def plot_confusion_matrix(cm, class_names, save_path):
    """Saves heatmap of confusion matrix."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 8}
    )
    plt.title("Confusion Matrix - 26 ASL Alphabet Classifier", fontsize=15, pad=15)
    plt.xlabel("Predicted Class", fontsize=13)
    plt.ylabel("True Class", fontsize=13)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved confusion matrix heatmap to {save_path}")

def train():
    # 1. Load Data
    X, y, label_map, inv_label_map = load_dataset()
    num_classes = len(label_map)

    # Save label map
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=4)
    print(f"[INFO] Saved label map to {config.LABEL_MAP_PATH}")

    # 2. Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SPLIT,
        stratify=y,
        random_state=config.RANDOM_STATE
    )

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Build & Compile Model
    model = build_sign_classifier(
        input_dim=config.FEATURE_DIM,
        num_classes=num_classes
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 4. Callbacks
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Train
    print("\n[TRAINING] Starting neural network training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Ensure best model is saved
    model.save(config.MODEL_PATH)
    print(f"[SUCCESS] Model saved to {config.MODEL_PATH}")

    # 6. Evaluation & Metrics
    print("\n[EVALUATION] Generating evaluation metrics on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  -> Test Loss: {test_loss:.4f}")
    print(f"  -> Test Accuracy: {test_acc * 100:.2f}%")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = [inv_label_map[i] for i in range(num_classes)]
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:\n", report)

    # Save classification report
    report_path = config.METRICS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\nTest Loss: {test_loss:.4f}\n\n")
        f.write(report)

    # Save plots
    plot_training_history(history, config.METRICS_DIR / "training_curves.png")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, config.METRICS_DIR / "confusion_matrix.png")

    print(f"\n[DONE] All evaluation artifacts saved in '{config.METRICS_DIR}'.")

if __name__ == "__main__":
    train()
