#!/usr/bin/env bash
# Quickstart menu for Sign Language Detection System

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "========================================================"
echo "   26-Alphabet Sign Language Recognition AI (A-Z)       "
echo "========================================================"
echo "Select an option:"
echo "  1) Generate 26-Alphabet Synthetic Dataset (A-Z)"
echo "  2) Collect new hand gesture data using Webcam"
echo "  3) Train 26-Class TensorFlow / Keras Classifier"
echo "  4) Run Standalone Evaluation & 26x26 Confusion Matrix"
echo "  5) Launch Real-Time Glassmorphic Webcam Detector"
echo "  6) Exit"
echo "========================================================"
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo "Generating synthetic dataset..."
        ./venv/bin/python generate_sample_data.py
        ;;
    2)
        echo "Starting Data Collector..."
        ./venv/bin/python data_collector.py
        ;;
    3)
        echo "Training Keras Model..."
        ./venv/bin/python model/train.py
        ;;
    4)
        echo "Evaluating Model..."
        ./venv/bin/python evaluate.py
        ;;
    5)
        echo "Launching Real-Time Detector..."
        ./venv/bin/python realtime_detector.py
        ;;
    6)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid option."
        ;;
esac
