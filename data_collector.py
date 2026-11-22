import os
import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure cache directories are writable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")

import cv2
import numpy as np

import config
from utils.landmark_extractor import HandLandmarkExtractor

def collect_data(classes=None, samples_per_class=None):
    classes = classes or config.CLASSES
    samples_per_class = samples_per_class or config.SAMPLES_PER_CLASS
    
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {config.CAMERA_INDEX}.")
        return

    print("==================================================")
    print("       SIGN LANGUAGE DATA COLLECTION TOOL         ")
    print("==================================================")
    print(f"Total Classes to Record: {len(classes)} -> {classes}")
    print(f"Target Samples per Class: {samples_per_class}")
    print("\nControls:")
    print("  [SPACE] Start recording current sign")
    print("  [N]     Skip to next sign")
    print("  [R]     Reset / re-record current sign")
    print("  [Q]     Quit collection")
    print("==================================================\n")

    for class_idx, class_name in enumerate(classes):
        class_folder = data_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        
        # Check existing samples
        existing_files = list(class_folder.glob("*.npy"))
        samples_collected = len(existing_files)

        is_recording = False
        print(f"\n[{class_idx + 1}/{len(classes)}] Class: '{class_name}' ({samples_collected}/{samples_per_class} existing)")

        while samples_collected < samples_per_class:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # Mirror view
            h, w, _ = frame.shape

            # Process landmarks
            results = extractor.process_frame(frame)
            features, landmarks, handedness = extractor.extract_landmarks(results)

            # Draw visual guides
            if landmarks:
                extractor.draw_landmarks(frame, landmarks)

            # Overlay HUD
            hud_color = (0, 0, 255) if is_recording else (0, 200, 255)
            status_text = "RECORDING" if is_recording else "PAUSED - Press [SPACE] to Record"
            
            # Status Banner
            cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 24), -1)
            cv2.putText(
                frame,
                f"Class: {class_name} [{class_idx + 1}/{len(classes)}]",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 230, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"Status: {status_text}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 120) if is_recording else (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            # Progress Bar
            progress_ratio = samples_collected / samples_per_class
            bar_w = 300
            bar_x = w - bar_w - 30
            cv2.rectangle(frame, (bar_x, 25), (bar_x + bar_w, 45), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, 25), (bar_x + int(bar_w * progress_ratio), 45), (50, 220, 100), -1)
            cv2.putText(
                frame,
                f"{samples_collected}/{samples_per_class} ({int(progress_ratio * 100)}%)",
                (bar_x, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            # Footer
            cv2.rectangle(frame, (0, h - 35), (w, h), (15, 15, 18), -1)
            cv2.putText(
                frame,
                "[SPACE] Toggle Record  |  [N] Next  |  [R] Reset  |  [Q] Quit",
                (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (170, 170, 170),
                1,
                cv2.LINE_AA
            )

            # If recording and hand is detected, save sample
            if is_recording and features is not None:
                sample_file = class_folder / f"sample_{samples_collected:04d}.npy"
                np.save(sample_file, features)
                samples_collected += 1
                # Small throttle to ensure slightly distinct frames
                time.sleep(0.04)

            cv2.imshow("Sign Language Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Quitting data collection.")
                cap.release()
                cv2.destroyAllWindows()
                extractor.close()
                return
            elif key == ord(' '):
                is_recording = not is_recording
            elif key == ord('n'):
                print(f"[INFO] Skipping '{class_name}'.")
                break
            elif key == ord('r'):
                print(f"[INFO] Resetting samples for '{class_name}'.")
                for f in class_folder.glob("*.npy"):
                    f.unlink()
                samples_collected = 0
                is_recording = False

        print(f"[SUCCESS] Finished class: '{class_name}' with {samples_collected} samples.")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("\n[COMPLETE] All data collection completed! You can now run model training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hand gesture landmark dataset.")
    parser.add_argument("--samples", type=int, default=config.SAMPLES_PER_CLASS, help="Samples per sign class")
    args = parser.parse_args()
    collect_data(samples_per_class=args.samples)
