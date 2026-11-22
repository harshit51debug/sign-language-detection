import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import config

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm webbing
    (5, 9), (9, 13), (13, 17), (0, 9), (0, 13)
]

class HandLandmarkExtractor:
    """
    Extracts and normalizes 3D hand landmarks using modern MediaPipe Tasks API.
    Performs wrist-relative translation normalization, scale normalization,
    and optional handedness mirroring for robust, position-invariant classification.
    """

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        mirror_left_hand: bool = True
    ):
        self.mirror_left_hand = mirror_left_hand
        self.model_path = Path(config.HAND_LANDMARKER_MODEL)

        # Download model asset if missing
        if not self.model_path.exists():
            print(f"[INFO] Downloading hand landmarker model to {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, str(self.model_path))

        base_options = mp_tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Converts BGR image to RGB and detects hand landmarks via MediaPipe Tasks.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)
        return results

    def extract_landmarks(self, results):
        """
        Extracts and normalizes hand landmarks from MediaPipe results.
        
        Returns:
            features: 1D np.ndarray of shape (63,) or None if no hand detected.
            hand_landmarks: List of landmark points for drawing.
            handedness_label: "Left" or "Right" (or "Unknown").
        """
        if not results.hand_landmarks or len(results.hand_landmarks) == 0:
            return None, None, None

        # Take the primary hand detected
        hand_landmarks = results.hand_landmarks[0]
        handedness_label = "Right"
        if results.handedness and len(results.handedness) > 0 and len(results.handedness[0]) > 0:
            handedness_label = results.handedness[0][0].category_name

        raw_coords = []
        for lm in hand_landmarks:
            raw_coords.append([lm.x, lm.y, lm.z])
        raw_coords = np.array(raw_coords, dtype=np.float32)  # Shape: (21, 3)

        # 1. Translation Normalization: Translate relative to wrist (landmark 0)
        wrist = raw_coords[0, :]
        relative_coords = raw_coords - wrist

        # 2. Handedness Normalization: Mirror left hand along x-axis to match right hand
        if self.mirror_left_hand and handedness_label == "Left":
            relative_coords[:, 0] = -relative_coords[:, 0]

        # 3. Scale Normalization: Normalize by maximum Euclidean distance from wrist
        distances = np.linalg.norm(relative_coords, axis=1)
        max_dist = np.max(distances)
        if max_dist > 1e-6:
            normalized_coords = relative_coords / max_dist
        else:
            normalized_coords = relative_coords

        # Flatten (21, 3) -> (63,)
        features = normalized_coords.flatten()
        return features, hand_landmarks, handedness_label

    def draw_landmarks(self, frame_bgr: np.ndarray, hand_landmarks):
        """
        Draws the hand skeleton and landmarks onto the frame using modern neon HUD styling.
        """
        if not hand_landmarks:
            return frame_bgr

        h, w, _ = frame_bgr.shape
        points = []
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))

        # Palm connections (Webbing)
        palm_edges = [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (0, 9), (0, 13)]
        for p1, p2 in palm_edges:
            cv2.line(frame_bgr, points[p1], points[p2], (65, 80, 110), 2, cv2.LINE_AA)

        # Finger bone connections (Cyan glow)
        finger_edges = [
            (1, 2), (2, 3), (3, 4),        # Thumb
            (5, 6), (6, 7), (7, 8),        # Index
            (9, 10), (10, 11), (11, 12),   # Middle
            (13, 14), (14, 15), (15, 16),  # Ring
            (17, 18), (18, 19), (19, 20)   # Pinky
        ]
        for p1, p2 in finger_edges:
            cv2.line(frame_bgr, points[p1], points[p2], (0, 230, 255), 2, cv2.LINE_AA)

        # Draw joints / landmark nodes
        for idx, (cx, cy) in enumerate(points):
            if idx in [4, 8, 12, 16, 20]:  # Fingertips (glowing magenta/white)
                cv2.circle(frame_bgr, (cx, cy), 8, (255, 60, 180), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (cx, cy), 11, (255, 120, 210), 1, cv2.LINE_AA)
            elif idx == 0:  # Wrist joint
                cv2.circle(frame_bgr, (cx, cy), 8, (0, 240, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (cx, cy), 12, (255, 255, 255), 1, cv2.LINE_AA)
            else:  # Finger knuckles
                cv2.circle(frame_bgr, (cx, cy), 4, (50, 240, 120), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (cx, cy), 6, (20, 60, 40), 1, cv2.LINE_AA)

        return frame_bgr

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass
