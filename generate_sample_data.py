import os
import sys
import shutil
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure cache directories are writable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")

import numpy as np
import config

def get_canonical_hand_pose(sign_name: str) -> np.ndarray:
    """
    Returns canonical (21, 3) 3D hand landmarks for all 26 ASL alphabet signs (A-Z).
    Wrist is positioned at origin (0, 0, 0).
    Landmarks indices:
      0: Wrist
      1-4: Thumb (CMC, MCP, IP, Tip)
      5-8: Index (MCP, PIP, DIP, Tip)
      9-12: Middle (MCP, PIP, DIP, Tip)
      13-16: Ring (MCP, PIP, DIP, Tip)
      17-20: Pinky (MCP, PIP, DIP, Tip)
    """
    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [0.0, 0.0, 0.0]

    # Knuckle base positions (MCP joints) relative to wrist
    mcp_x = [-0.18, -0.09, 0.0, 0.09, 0.17]
    mcp_y = [-0.15, -0.38, -0.40, -0.37, -0.33]

    coords[1] = [mcp_x[0], mcp_y[0], 0.0]
    coords[5] = [mcp_x[1], mcp_y[1], 0.0]
    coords[9] = [mcp_x[2], mcp_y[2], 0.0]
    coords[13] = [mcp_x[3], mcp_y[3], 0.0]
    coords[17] = [mcp_x[4], mcp_y[4], 0.0]

    # Finger extensions: [Thumb, Index, Middle, Ring, Pinky]
    # 1.0 = fully extended upright, 0.0 = curled into fist
    sign_key = sign_name.upper().strip()

    finger_extensions = {
        "A": [0.35, 0.05, 0.05, 0.05, 0.05],     # Fist, thumb upright beside index
        "B": [0.10, 1.00, 1.00, 1.00, 1.00],     # 4 fingers straight up, thumb tucked
        "C": [0.45, 0.45, 0.45, 0.45, 0.45],     # Curved arc forming 'C'
        "D": [0.30, 1.00, 0.20, 0.20, 0.20],     # Index up, others curl touching thumb
        "E": [0.15, 0.15, 0.15, 0.15, 0.15],     # Tight curled claw, thumb tucked under
        "F": [0.35, 0.25, 1.00, 1.00, 1.00],     # Index & thumb touching ("OK"), 3 up
        "G": [0.70, 0.90, 0.05, 0.05, 0.05],     # Index pointing sideways/forward with thumb
        "H": [0.20, 0.95, 0.95, 0.05, 0.05],     # Index & middle together pointing forward
        "I": [0.15, 0.05, 0.05, 0.05, 1.00],     # Pinky up, others in fist
        "J": [0.20, 0.05, 0.05, 0.05, 0.95],     # Pinky up with curve tilt
        "K": [0.40, 1.00, 0.70, 0.05, 0.05],     # Index up, middle angled, thumb between
        "L": [0.95, 1.00, 0.05, 0.05, 0.05],     # Thumb & index at 90 deg ("L")
        "M": [0.30, 0.10, 0.10, 0.10, 0.05],     # 3 fingers over thumb
        "N": [0.30, 0.10, 0.10, 0.05, 0.05],     # 2 fingers over thumb
        "O": [0.35, 0.35, 0.35, 0.35, 0.35],     # All fingers curve to meet thumb tip
        "P": [0.35, 0.85, 0.65, 0.05, 0.05],     # 'K' pointing downward
        "Q": [0.75, 0.80, 0.05, 0.05, 0.05],     # 'G' pointing downward
        "R": [0.15, 0.98, 0.98, 0.05, 0.05],     # Index & middle crossed
        "S": [0.25, 0.05, 0.05, 0.05, 0.05],     # Fist, thumb over front
        "T": [0.35, 0.10, 0.05, 0.05, 0.05],     # Thumb between index & middle
        "U": [0.15, 1.00, 1.00, 0.05, 0.05],     # Index & middle upright together
        "V": [0.15, 1.00, 1.00, 0.05, 0.05],     # Index & middle spread (Peace sign)
        "W": [0.20, 1.00, 1.00, 1.00, 0.05],     # Index, middle, ring spread ("W")
        "X": [0.15, 0.50, 0.05, 0.05, 0.05],     # Index hooked/bent, others folded
        "Y": [0.95, 0.05, 0.05, 0.05, 0.95],     # Thumb & pinky out ("Hang loose")
        "Z": [0.15, 0.95, 0.05, 0.05, 0.05]      # Index pointing
    }

    ext = finger_extensions.get(sign_key, [0.5, 0.5, 0.5, 0.5, 0.5])

    # Finger joint definitions
    finger_indices = [
        [1, 2, 3, 4],       # Thumb
        [5, 6, 7, 8],       # Index
        [9, 10, 11, 12],    # Middle
        [13, 14, 15, 16],   # Ring
        [17, 18, 19, 20]    # Pinky
    ]

    for f_idx, indices in enumerate(finger_indices):
        f_ext = ext[f_idx]
        for step, lm_idx in enumerate(indices[1:], start=1):
            if f_idx == 0:  # Thumb
                if sign_key in ["L", "Y", "G"]:
                    dx = -0.09 * (f_ext + 0.2)
                    dy = -0.02
                    dz = 0.01
                elif sign_key in ["A", "S"]:
                    dx = -0.04
                    dy = -0.08 * f_ext
                    dz = 0.04
                else:
                    dx = -0.06 * f_ext
                    dy = -0.07 * f_ext
                    dz = 0.05 * (1.0 - f_ext)
            else:  # Fingers (Index, Middle, Ring, Pinky)
                # Abduction / spread
                spread = (f_idx - 2) * 0.035
                if sign_key == "V" and f_idx in [1, 2]:
                    spread = -0.045 if f_idx == 1 else 0.045  # V spread
                elif sign_key in ["U", "H"] and f_idx in [1, 2]:
                    spread = -0.01 if f_idx == 1 else 0.01    # Close parallel
                elif sign_key == "R" and f_idx in [1, 2]:
                    spread = 0.02 if f_idx == 1 else -0.02     # Crossed

                if f_ext > 0.4:
                    dx = spread * f_ext
                    dy = -0.11 * f_ext
                    dz = 0.02 if sign_key not in ["C", "O"] else 0.07 * (1.0 - f_ext)
                else:
                    dx = spread * 0.2
                    dy = 0.04 * (1.0 - f_ext)  # Curled downward
                    dz = 0.08 * (1.0 - f_ext)  # Curled forward in z

            coords[lm_idx] = coords[indices[step - 1]] + np.array([dx, dy, dz], dtype=np.float32)

    # Scale normalization
    distances = np.linalg.norm(coords, axis=1)
    max_d = np.max(distances)
    if max_d > 0:
        coords /= max_d

    return coords

def generate_synthetic_dataset(samples_per_class: int = config.SAMPLES_PER_CLASS):
    """
    Generates synthetic landmark dataset for all 26 ASL alphabet signs (A-Z)
    with realistic anatomical jitter, 3D rotations, and scale perturbations.
    """
    data_dir = Path(config.DATA_DIR)
    
    # Clean previous dataset to ensure only 26 alphabets are present
    if data_dir.exists():
        print(f"[CLEANUP] Resetting {data_dir} for 26-class dataset...")
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("   GENERATING 26-ALPHABET SYNTHETIC DATASET (A-Z) ")
    print("==================================================")
    print(f"Total Classes: {len(config.CLASSES)} -> {config.CLASSES}")
    print(f"Samples per Class: {samples_per_class}")
    print(f"Total Target Dataset Size: {len(config.CLASSES) * samples_per_class} samples\n")

    for class_idx, class_name in enumerate(config.CLASSES, 1):
        class_folder = data_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        canonical = get_canonical_hand_pose(class_name)

        for i in range(samples_per_class):
            # 1. 3D Rotation jitter (pitch, roll, yaw jitter up to ~12 degrees)
            angles = np.random.uniform(-0.12, 0.12, size=3)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])]
            ])
            Ry = np.array([
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])]
            ])
            Rz = np.array([
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1]
            ])
            R = Rz @ Ry @ Rx

            rotated = canonical @ R.T

            # 2. Gaussian landmark measurement noise
            noise = np.random.normal(0, 0.015, size=canonical.shape)
            augmented = rotated + noise

            # 3. Wrist origin translation & scale normalization
            wrist = augmented[0, :]
            augmented = augmented - wrist
            max_dist = np.max(np.linalg.norm(augmented, axis=1))
            if max_dist > 0:
                augmented /= max_dist

            features = augmented.flatten().astype(np.float32)
            np.save(class_folder / f"sample_{i:04d}.npy", features)

        print(f"  [{class_idx:02d}/26] [OK] Generated {samples_per_class} samples for '{class_name}'.")

    print(f"\n[COMPLETE] 26-alphabet dataset saved at: {data_dir.resolve()}")

if __name__ == "__main__":
    generate_synthetic_dataset()
