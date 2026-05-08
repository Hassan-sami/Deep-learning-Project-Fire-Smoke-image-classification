# 🔥 Forest Fire & Smoke Detection Web Application

An advanced AI-powered web application for detecting forest fires and smoke in images, videos, and real-time webcam feeds. Built with Flask, TensorFlow, and OpenCV.

![Forest Fire Detection](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Detection Capabilities
- **🖼️ Single Image Detection**: Upload and analyze individual images with confidence scores
- **📁 Batch Processing**: Process multiple images simultaneously with comprehensive results
- **🎥 Video Analysis**: Analyze video footage frame-by-frame with temporal detection
- **📹 Real-time Webcam**: Live fire and smoke detection using your webcam

### Detection Features
- Multi-class classification (Non-Fire, Fire, Smoke)
- Confidence threshold configuration
- Test-time augmentation for improved accuracy
- Real-time probability visualization
- Alert system for high-confidence fire/smoke detections

### Reporting & Analytics
- Interactive HTML reports with charts and statistics
- JSON export for programmatic access
- Detection history tracking
- Performance metrics (inference time, confidence distribution)
- Class distribution analysis

### Model Management
- Easy model loading and switching
- Support for .h5 and .keras model formats
- JSON-based label configuration
- Auto-detection of available models
- Model upload functionality

### User Interface
- Modern, responsive web design
- Real-time results visualization
- Interactive charts and graphs
- Progress indicators for batch processing
- Mobile-friendly interface

### Additional Features
- Email alerts for fire/smoke detections
- CPU/GPU device selection
- Configurable frame skip for video processing
- Detection logging and history
- Dark/light theme support

## 🚀 Installation

### Prerequisites

- Python 3.11.9 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Clone the Repository

```bash
git clone https://github.com/Hassan-sami/Deep-learning-Project-Fire-Smoke-image-classification.git
cd Deep-learning-Project-Fire-Smoke-image-classification/image-classification-app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
<----optional--->
mkdir models
mkdir uploads\images
mkdir uploads\videos
mkdir uploads\temp
mkdir logs
mkdir outputs
<----optional--->
<----important--->
models/
├── forest_fire_model.(h5/keras)    # Your trained model file
└── forest_fire_model.json  # Labels configuration file
<----important--->
python app.py
