import os
import sys
import time
import json
import subprocess
import platform
from collections import deque, Counter
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib and Keras config dirs are writable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")

import cv2
import numpy as np
import tensorflow as tf

import config
from utils.landmark_extractor import HandLandmarkExtractor
from utils.visualization import (
    draw_header,
    draw_fps,
    draw_prediction_card,
    draw_sentence_panel,
    draw_hand_bounding_box
)

def speak_text(text: str):
    """Speaks text using macOS native 'say' command or safe fallback."""
    if not text.strip():
        return
    if platform.system() == "Darwin":
        try:
            subprocess.Popen(["say", text.strip()])
        except Exception as e:
            print(f"[SPEECH ERROR] {e}")

def run_detector():
    print("==================================================")
    print("    REAL-TIME SIGN LANGUAGE DETECTOR (AI/ML)      ")
    print("==================================================")

    # 1. Check prerequisites
    if not Path(config.MODEL_PATH).exists():
        print(f"[ERROR] Model file not found at '{config.MODEL_PATH}'.")
        print("Please train your model first using: python model/train.py")
        return

    if not Path(config.LABEL_MAP_PATH).exists():
        print(f"[ERROR] Label map not found at '{config.LABEL_MAP_PATH}'.")
        return

    # 2. Load Model and Label Map
    print(f"[INFO] Loading trained Keras model from {config.MODEL_PATH}...")
    model = tf.keras.models.load_model(config.MODEL_PATH)
    
    with open(config.LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    # Map index -> label
    inv_label_map = {int(v): k for k, v in label_map.items()}
    print(f"[INFO] Classes recognized ({len(label_map)}): {list(label_map.keys())}")

    # 3. Initialize Webcam & Extractor
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {config.CAMERA_INDEX}.")
        return

    extractor = HandLandmarkExtractor()

    # Smoothing and Debouncing state
    prediction_window = deque(maxlen=config.PREDICTION_SMOOTHING_WINDOW)
    sentence_buffer = []
    last_confirmed_sign = None
    sign_hold_counter = 0

    fps = 0.0
    prev_time = time.time()

    print("\nControls:")
    print("  [SPACE]     Add space to sentence")
    print("  [BACKSPACE] Delete last character/word")
    print("  [S]         Speak aloud current sentence (macOS)")
    print("  [C]         Clear sentence")
    print("  [Q]         Quit application\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Camera frame drop.")
            break

        # Mirror horizontally for natural webcam interaction
        frame = cv2.flip(frame, 1)

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        # Extract landmarks
        results = extractor.process_frame(frame)
        features, landmarks, handedness = extractor.extract_landmarks(results)

        predicted_sign = "None"
        confidence = 0.0
        top_k_list = []
        hold_ratio = 0.0

        if features is not None:
            # Draw hand skeleton
            extractor.draw_landmarks(frame, landmarks)

            # Reshape feature vector for model: (1, 63)
            feat_input = np.expand_dims(features, axis=0)

            # Model inference
            preds = model.predict(feat_input, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            raw_confidence = float(preds[top_idx])

            # Get Top-K candidate probabilities
            top_k_indices = np.argsort(preds)[::-1][:getattr(config, "TOP_K_PREDICTIONS", 3)]
            top_k_list = [(inv_label_map.get(int(idx), "?"), float(preds[idx])) for idx in top_k_indices]

            if raw_confidence >= config.CONFIDENCE_THRESHOLD:
                raw_sign = inv_label_map.get(top_idx, "Unknown")
                prediction_window.append(raw_sign)
                confidence = raw_confidence
            else:
                prediction_window.append("...")

            # Majority voting across smoothing window
            if len(prediction_window) == config.PREDICTION_SMOOTHING_WINDOW:
                sign_counts = Counter(prediction_window)
                most_common_sign, count = sign_counts.most_common(1)[0]
                if count >= (config.PREDICTION_SMOOTHING_WINDOW // 2 + 1) and most_common_sign != "...":
                    predicted_sign = most_common_sign

                    # Sentence debouncing logic
                    if predicted_sign == last_confirmed_sign:
                        sign_hold_counter += 1
                        hold_ratio = min(1.0, sign_hold_counter / max(1, config.SENTENCE_DEBOUNCE_FRAMES))
                        if sign_hold_counter == config.SENTENCE_DEBOUNCE_FRAMES:
                            # Append to sentence
                            if len(predicted_sign) == 1:
                                sentence_buffer.append(predicted_sign)
                            else:
                                sentence_buffer.append(" " + predicted_sign + " ")
                    else:
                        last_confirmed_sign = predicted_sign
                        sign_hold_counter = 0
                        hold_ratio = 0.0
            
            # Draw target reticle box around hand with label
            draw_hand_bounding_box(frame, landmarks, label=f"{predicted_sign} ({int(confidence * 100)}%)")
        else:
            prediction_window.clear()
            sign_hold_counter = 0
            last_confirmed_sign = None
            hold_ratio = 0.0

        # Render Modern Glassmorphic UI Elements
        draw_header(frame, "Sign Language AI Recognizer (A-Z)", fps=fps)
        draw_prediction_card(
            frame,
            predicted_label=predicted_sign,
            confidence=confidence,
            handedness=handedness,
            top_k_list=top_k_list,
            hold_ratio=hold_ratio
        )
        current_sentence = "".join(sentence_buffer)
        show_cursor = (int(time.time() * 2) % 2 == 0)
        draw_sentence_panel(frame, current_sentence, show_cursor=show_cursor)

        cv2.imshow("Sign Language AI Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            sentence_buffer.append(" ")
        elif key == 8 or key == ord('d'):  # Backspace or 'd'
            if sentence_buffer:
                sentence_buffer.pop()
        elif key == ord('c'):
            sentence_buffer.clear()
        elif key == ord('s'):
            speak_text(current_sentence)

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("[INFO] Application terminated cleanly.")

if __name__ == "__main__":
    run_detector()
