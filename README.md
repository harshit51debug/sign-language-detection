# Real-Time 26-Alphabet Sign Language Recognition System using OpenCV, MediaPipe & TensorFlow/Keras

An end-to-end Computer Vision & Deep Learning project capable of recognizing all **26 American Sign Language (ASL) alphabets (A through Z)** in real-time, translating finger spelling into words and synthesized audible speech.

---

## 📌 Project Overview & Highlights

- **Full 26-Alphabet Recognition**: Classifies the entire English alphabet (`A` through `Z`) in real time.
- **Computer Vision Pipeline**: Ultra-lightweight 21 3D hand landmark tracking using **Google MediaPipe Tasks API** running with hardware acceleration on Apple Silicon Metal GPU / standard CPUs.
- **Geometric Normalization**: 
  - **Wrist Translation Invariance**: Centers keypoints relative to the wrist $(x_0, y_0, z_0)$.
  - **Scale Invariance**: Normalizes by maximum Euclidean hand radius to maintain accuracy regardless of camera distance.
  - **Handedness Invariance**: Automatically mirrors left-hand coordinates, allowing users to sign with either hand.
- **Deep Learning Classifier**: Regularized Deep Multi-Layer Perceptron (Input 63 $\rightarrow$ Dense 256 $\rightarrow$ Dense 128 $\rightarrow$ Dense 64 $\rightarrow$ Softmax 26) with Batch Normalization, L2 regularization, and Dropout.
- **Glassmorphic Cyber-HUD UI**:
  - Live FPS and camera feed indicator.
  - Primary detected letter with dynamic confidence gradient bar.
  - **Top-3 Alternate Candidate Probabilities**: Displays the top 3 potential matches with probability bars.
  - **Stabilize-to-Commit Progress Bar**: Real-time visual timer showing gesture hold duration before appending to the sentence buffer.
  - Dual-color neon skeleton: Deep cobalt palm mesh with electric cyan finger bones, emerald knuckles, and glowing magenta fingertip nodes.
  - Target reticle corner-bracket bounding box.
  - Modern bottom dock displaying accumulated sentence, letter/word counts, blinking cursor, and keyboard hotkeys.
  - **Text-to-Speech (TTS)**: Instant speech synthesis of formed sentences.

---

## 🏗️ System Architecture

```
   ┌──────────────────────┐
   │  Webcam Video Stream │ -> 1280x720 RGB @ 45-60 FPS
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │   MediaPipe Tasks    │ -> Palm Detection & 21 3D Hand Landmarks
   │   Hand Landmarker    │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Coordinate Transform │ -> Wrist Translation: P_i - P_wrist
   │   & Normalization    │ -> Handedness Inversion: Left hand mirrored
   │                      │ -> Scale Normalization: P_i / max_dist
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Keras Neural Net    │ -> Dense(256) + BatchNorm + Dropout(0.3)
   │     Classifier       │ -> Dense(128) + BatchNorm + Dropout(0.3)
   │                      │ -> Dense(64)  + BatchNorm + Dropout(0.2)
   │                      │ -> Softmax Output Layer (26 Classes: A-Z)
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Glassmorphic HUD &  │ -> Sliding Window Majority Voting (Buffer=5)
   │   Sentence Engine    │ -> Stabilize-to-Commit Debounce Timer (12 frames)
   │                      │ -> Top-3 Alternate Candidate Breakdown
   │                      │ -> Text-to-Speech (TTS)
   └──────────────────────┘
```

---

## 🗂️ Project Directory Structure

```text
sign lang/
├── config.py                 # Central configurations, 26 classes & hyperparams
├── requirements.txt          # Python library dependencies
├── hand_landmarker.task      # MediaPipe Tasks vision model asset (7.5 MB)
├── data_collector.py         # Interactive webcam tool to record custom gesture datasets
├── generate_sample_data.py   # Canonical seed dataset generator for all 26 alphabets (A-Z)
├── realtime_detector.py      # Real-time webcam inference application with Cyber-HUD
├── evaluate.py               # Standalone model evaluation & 26x26 confusion matrix
├── data/                     # Stored gesture landmark vectors (.npy) for A-Z
├── model/
│   ├── architecture.py       # Keras Deep Neural Network definition (256-128-64-26)
│   └── train.py              # Model training, validation curves & metrics export
├── utils/
│   ├── landmark_extractor.py # MediaPipe extraction, 3D normalization & neon skeleton
│   └── visualization.py      # Glassmorphic HUD, top-3 candidates, reticle & dock
├── saved_models/             # Exported Keras model & 26-class label map
└── metrics/                  # Training history plots & 26x26 confusion matrix heatmaps
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Generate / Prepare 26-Alphabet Dataset

You can quickly generate all 26 ASL alphabet signs (A to Z) with geometric variations and 3D rotations:

```bash
python generate_sample_data.py
```
This prepares 150 augmented samples per class $\times$ 26 letters = **3,900 total samples** in `data/`.

*(Optional)* To record your own live webcam samples:
```bash
python data_collector.py
```

---

### 3. Train the 26-Class TensorFlow/Keras Classifier

```bash
python model/train.py
```
- Trains the deep neural network across all 26 classes.
- Saves the trained model to `saved_models/sign_language_model.keras` and `saved_models/label_map.json`.
- Exports training curves (`training_curves.png`) and confusion matrix (`confusion_matrix.png`) to `metrics/`.

---

### 4. Evaluate the Model

```bash
python evaluate.py
```
Evaluates accuracy, precision, recall, and generates the high-resolution $26 \times 26$ confusion matrix heatmap.

---

### 5. Launch the Real-Time Webcam Detector

```bash
python realtime_detector.py
```

#### Real-Time Interactive Controls:
| Key | Action |
| :--- | :--- |
| **Hold Sign Steady** | Hold gesture for ~0.4s to automatically commit the letter |
| **`[SPACE]`** | Insert a space between words |
| **`[BACKSPACE]` / `[D]`** | Delete the last committed character |
| **`[S]`** | Speak aloud the accumulated sentence via Text-to-Speech |
| **`[C]`** | Clear the sentence buffer |
| **`[Q]`** | Exit and close the application |

---

## 📊 Experimental Results (26 Classes)

- **Overall Dataset Accuracy**: **`95.26%`** across 3,900 samples
- **Independent Test Accuracy**: **`92.18%`** across 780 test samples
- **Inference Latency**: **$<1.5$ ms** per frame
- **Camera Pipeline FPS**: **45–60 FPS** on Apple Silicon M-series

---

## 📜 License
This project is open-source and intended for academic and educational purposes.
