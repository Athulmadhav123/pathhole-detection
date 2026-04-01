# pathhole-detection
📌 Overview

This project is a Deep Learning-based Pothole Detection System that classifies road images as Pothole or Normal Road using a trained CNN model.
It also includes a Gradio web interface and live webcam detection for real-time predictions.

🚀 Features
📷 Image-based pothole detection
🎥 Live webcam detection
🤖 Deep Learning model (CNN)
🌐 Interactive UI using Gradio
⚡ Fast and simple deployment

Model Details
Model Type: Convolutional Neural Network (CNN)
Input Size: 224 × 224 × 3
Output: Binary Classification (Pothole / Normal)
Framework: TensorFlow / Keras

Project Structure
pothole_project/
│
├── app.py                # Gradio UI for image detection
├── live_app.py           # Live webcam detection
├── train_model.py        # Model training script
├── pothole_model.h5      # Trained model
│
├── yes/                  # Pothole images
├── no/                   # Normal road images
└── flagged/              # Misclassified or test images
