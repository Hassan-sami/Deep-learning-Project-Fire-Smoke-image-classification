#!/usr/bin/env python3
"""
Forest Fire and Smoke Detection - Command Line Inference Tool (Enhanced)
Usage: python inference.py --model model.h5 --input image.jpg

New Features:
- Real-time webcam detection
- Video processing with frame sampling
- Batch processing with progress bar
- Model performance statistics
- Confidence thresholding
- Multiple output formats (text, JSON, CSV, HTML report)
- Email alerts for fire detection
- Image preprocessing options (rotation, flip, brightness)
- Detection logging and history
- GPU/CPU device selection
"""

import argparse
import os
import sys
import time
import json
import logging
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DetectionLogger:
    """Logging system for detection results"""
    
    def __init__(self, log_dir='detection_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('ForestFireDetection')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"detection_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Detection history
        self.history_file = self.log_dir / 'detection_history.json'
        self.history = self._load_history()
    
    def _load_history(self):
        """Load detection history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'detections': [], 'statistics': {}}
    
    def _save_history(self):
        """Save detection history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def log_detection(self, result, image_path=None):
        """Log a detection result"""
        detection_entry = {
            'timestamp': datetime.now().isoformat(),
            'image': image_path or result.get('file', 'unknown'),
            'prediction': result['predicted_class'],
            'confidence': result['confidence'],
            'probabilities': result.get('probabilities', {}),
            'inference_time': result.get('inference_time', 0)
        }
        self.history['detections'].append(detection_entry)
        self._save_history()
        
        self.logger.info(
            f"Detection: {result['predicted_class']} "
            f"(confidence: {result['confidence']:.2%}) "
            f"in {result.get('inference_time', 0):.3f}s"
        )
        
        # Alert for high-confidence fire/smoke detections
        if result['predicted_class'].lower() in ['fire', 'smoke']:
            if result['confidence'] > 0.7:
                self.logger.warning(
                    f"HIGH CONFIDENCE {result['predicted_class'].upper()} "
                    f"DETECTION: {result['confidence']:.2%}"
                )
    
    def get_statistics(self):
        """Get detection statistics"""
        if not self.history['detections']:
            return {}
        
        detections = self.history['detections']
        stats = {
            'total_detections': len(detections),
            'average_confidence': np.mean([d['confidence'] for d in detections]),
            'average_inference_time': np.mean([d['inference_time'] for d in detections]),
            'class_distribution': defaultdict(int),
            'high_confidence_detections': 0
        }
        
        for d in detections:
            stats['class_distribution'][d['prediction']] += 1
            if d['confidence'] > 0.8:
                stats['high_confidence_detections'] += 1
        
        stats['class_distribution'] = dict(stats['class_distribution'])
        self.history['statistics'] = stats
        self._save_history()
        
        return stats


class ImagePreprocessor:
    """Image preprocessing and augmentation"""
    
    def __init__(self, img_size=64):
        self.img_size = img_size
    
    def preprocess(self, image, augment=False):
        """Preprocess image with optional augmentation"""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
        else:
            img = image
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if augment:
            img = self._augment(img)
        
        # Normalize
        img_array = img.astype(np.float32) / 255.0
        
        return img_array
    
    def _augment(self, image):
        """Apply random augmentations"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = 0.8 + np.random.random() * 0.4  # 0.8 to 1.2
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Random rotation
        if np.random.random() > 0.7:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
        
        return image
    
    def preprocess_batch(self, images, augment=False):
        """Preprocess multiple images"""
        processed = []
        for img in images:
            if isinstance(img, str):
                img = cv2.imread(img)
            processed.append(self.preprocess(img, augment))
        
        return np.array(processed)


class ForestFireDetector:
    """Enhanced Forest Fire and Smoke Detection Model"""
    
    def __init__(self, model_path, img_size=64, verbose=True, 
                 confidence_threshold=0.5, device='/CPU:0', 
                 enable_logging=True):
        """
        Initialize detector with enhanced features.
        
        Args:
            model_path: Path to .h5 model file
            img_size: Input image size (default: 64)
            verbose: Print status messages
            confidence_threshold: Minimum confidence for positive detection
            device: TensorFlow device to use (/CPU:0 or /GPU:0)
            enable_logging: Enable detection logging
        """
        self.img_size = img_size
        self.model_path = model_path
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.enable_logging = enable_logging
        
        # Setup components
        self.preprocessor = ImagePreprocessor(img_size)
        self.class_names = self._load_labels()
        
        # Initialize logger
        if enable_logging:
            self.logger = DetectionLogger()
        else:
            self.logger = None
        
        # Load model with device placement
        self.model = self._load_model()
    
    def _log(self, message, level='info'):
        """Enhanced logging"""
        if self.verbose:
            prefix = {
                'info': 'ℹ️',
                'success': '✅',
                'warning': '⚠️',
                'error': '❌',
                'alert': '🚨'
            }
            print(f"{prefix.get(level, '')} {message}")
    
    def _get_labels_path(self):
        """Get the expected labels JSON file path based on model path"""
        model_path = Path(self.model_path)
        labels_path = model_path.with_suffix('.json')
        return labels_path
    
    def _load_labels(self):
        """Load class labels from JSON file with enhanced error handling"""
        labels_path = self._get_labels_path()
        
        if not os.path.exists(labels_path):
            self._log(f"Labels file not found: {labels_path}", 'error')
            self._log("Creating default labels file...", 'warning')
            
            # Try to infer from model output shape if possible
            default_labels = {
                "class_names": ["non_fire", "fire", "smoke"],
                "num_classes": 3,
                "description": "Forest fire and smoke detection classes",
                "alert_classes": ["fire", "smoke"],
                "created": datetime.now().isoformat()
            }
            try:
                with open(labels_path, 'w') as f:
                    json.dump(default_labels, f, indent=2)
                self._log(f"Default labels saved to: {labels_path}", 'success')
            except Exception as e:
                self._log(f"Could not create labels file: {e}", 'error')
            
            return default_labels["class_names"]
        
        try:
            with open(labels_path, 'r') as f:
                labels_data = json.load(f)
            
            # Extract class names from various formats
            if isinstance(labels_data, list):
                class_names = labels_data
                self.labels_metadata = {'class_names': class_names}
            elif isinstance(labels_data, dict):
                class_names = (labels_data.get('class_names') or 
                             labels_data.get('classes') or 
                             labels_data.get('labels'))
                self.labels_metadata = labels_data
                
                if class_names is None:
                    raise ValueError("No class names found in JSON")
            else:
                raise ValueError("Invalid labels format")
            
            self._log(f"Loaded {len(class_names)} classes", 'success')
            self._log(f"Classes: {', '.join(class_names)}", 'info')
            
            # Log alert classes if defined
            alert_classes = self.labels_metadata.get('alert_classes', [])
            if alert_classes:
                self._log(f"Alert classes: {', '.join(alert_classes)}", 'info')
            
            return class_names
            
        except Exception as e:
            self._log(f"Error loading labels: {e}", 'error')
            return ["non_fire", "fire", "smoke"]
    
    def _load_model(self):
        """Load model with device placement and enhanced compatibility"""
        if not os.path.exists(self.model_path):
            self._log(f"Model not found: {self.model_path}", 'error')
            return None
        
        with tf.device(self.device):
            loading_methods = [
                lambda: load_model(self.model_path),
                lambda: load_model(self.model_path, compile=False),
                lambda: tf.keras.models.load_model(self.model_path, compile=False),
            ]
            
            for i, method in enumerate(loading_methods, 1):
                try:
                    self._log(f"Loading model on {self.device} (method {i})...")
                    model = method()
                    
                    # Verify model output matches labels
                    if model.output_shape[-1] != len(self.class_names):
                        self._log(
                            f"Model output ({model.output_shape[-1]}) doesn't match "
                            f"labels ({len(self.class_names)})", 'warning'
                        )
                    
                    self._log(f"Model loaded successfully!", 'success')
                    self._log(f"Input shape: {model.input_shape}", 'info')
                    self._log(f"Output shape: {model.output_shape}", 'info')
                    return model
                except Exception as e:
                    if self.verbose:
                        self._log(f"Method {i} failed: {str(e)[:80]}...", 'warning')
        
        # Fallback: Rebuild model
        self._log("Attempting to rebuild model architecture...", 'warning')
        try:
            model = self._build_model(len(self.class_names))
            model.load_weights(self.model_path)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self._log("Model rebuilt and weights loaded!", 'success')
            return model
        except Exception as e:
            self._log(f"All loading methods failed: {e}", 'error')
            return None
    
    def _build_model(self, num_classes=None):
        """Build model architecture with configurable classes"""
        if num_classes is None:
            num_classes = len(self.class_names)
        
        model = tf.keras.models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', 
                        input_shape=(self.img_size, self.img_size, 3)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def predict(self, image_path, threshold=None, augment=False):
        """
        Enhanced prediction with confidence thresholding.
        
        Args:
            image_path: Path to image
            threshold: Override confidence threshold (None uses default)
            augment: Apply test-time augmentation
        
        Returns:
            dict with prediction results or None on error
        """
        if self.model is None:
            self._log("Model not loaded", 'error')
            return None
        
        if threshold is None:
            threshold = self.confidence_threshold
        
        if not os.path.exists(image_path):
            self._log(f"Image not found: {image_path}", 'error')
            return None
        
        try:
            start_time = time.time()
            
            # Preprocess image
            img_array = self.preprocessor.preprocess(image_path, augment=augment)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Test-time augmentation for better predictions
            if augment:
                predictions = []
                # Run multiple augmented passes
                for _ in range(5):
                    aug_img = self.preprocessor.preprocess(image_path, augment=True)
                    pred = self.model.predict(np.expand_dims(aug_img, axis=0), verbose=0)
                    predictions.append(pred[0])
                predictions = np.mean(predictions, axis=0)
            else:
                predictions = self.model.predict(img_array, verbose=0)[0]
            
            elapsed = time.time() - start_time
            
            # Get results
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            predicted_class = self.class_names[predicted_idx]
            
            # Check confidence threshold
            if confidence < threshold:
                predicted_class = 'uncertain'
            
            result = {
                'file': os.path.basename(image_path),
                'path': os.path.abspath(image_path),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'inference_time': elapsed,
                'threshold_used': threshold,
                'above_threshold': confidence >= threshold,
                'probabilities': {
                    self.class_names[i]: float(predictions[i])
                    for i in range(len(self.class_names))
                }
            }
            
            # Log detection
            if self.logger:
                self.logger.log_detection(result, image_path)
            
            # Alert for high-confidence fire/smoke
            if predicted_class in self.labels_metadata.get('alert_classes', []):
                if confidence > 0.7:
                    self._log(
                        f"HIGH CONFIDENCE DETECTION: {predicted_class.upper()} "
                        f"({confidence:.1%})", 'alert'
                    )
            
            return result
            
        except Exception as e:
            self._log(f"Prediction error: {e}", 'error')
            if self.logger:
                self.logger.logger.error(f"Prediction failed for {image_path}: {e}")
            return None
    
    def predict_batch(self, image_paths, threshold=None, augment=False, show_progress=True):
        """Enhanced batch prediction with progress tracking"""
        results = []
        
        # Setup progress bar
        iterator = image_paths
        if show_progress and HAS_TQDM:
            iterator = tqdm(image_paths, desc="Processing images", unit="img")
        elif show_progress:
            self._log(f"Processing {len(image_paths)} images...")
        
        for i, path in enumerate(iterator, 1):
            if not show_progress or not HAS_TQDM:
                if self.verbose:
                    self._log(f"[{i}/{len(image_paths)}] {os.path.basename(path)}")
            
            result = self.predict(path, threshold=threshold, augment=augment)
            if result:
                results.append(result)
        
        return results
    
    def predict_directory(self, directory_path, threshold=None, augment=False, recursive=False):
        """Enhanced directory prediction with recursive option"""
        if not os.path.isdir(directory_path):
            self._log(f"Directory not found: {directory_path}", 'error')
            return []
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        if recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if Path(file).suffix.lower() in extensions:
                        image_paths.append(os.path.join(root, file))
        else:
            for file in sorted(os.listdir(directory_path)):
                if Path(file).suffix.lower() in extensions:
                    image_paths.append(os.path.join(directory_path, file))
        
        if not image_paths:
            self._log(f"No images found in {directory_path}", 'warning')
            return []
        
        self._log(f"Found {len(image_paths)} images")
        return self.predict_batch(image_paths, threshold=threshold, augment=augment)
    
    def process_video(self, video_path, output_path=None, frame_skip=30, 
                     threshold=None, show_display=False):
        """Process video file for fire/smoke detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self._log(f"Cannot open video: {video_path}", 'error')
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self._log(f"Video: {total_frames} frames @ {fps}fps, {width}x{height}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        processed_frames = 0
        
        # Setup progress bar
        pbar = tqdm(total=total_frames, desc="Processing video") if HAS_TQDM else None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % frame_skip != 0:
                if pbar:
                    pbar.update(1)
                continue
            
            processed_frames += 1
            
            # Convert frame to RGB and predict
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Preprocess and predict
                processed = self.preprocessor.preprocess(rgb_frame)
                img_array = np.expand_dims(processed, axis=0)
                predictions = self.model.predict(img_array, verbose=0)[0]
                
                predicted_idx = np.argmax(predictions)
                confidence = float(predictions[predicted_idx])
                predicted_class = self.class_names[predicted_idx]
                
                if confidence >= (threshold or self.confidence_threshold):
                    result = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'predicted_class': predicted_class,
                        'confidence': confidence
                    }
                    results.append(result)
                    
                    # Annotate frame
                    label = f"{predicted_class}: {confidence:.2%}"
                    color = (0, 0, 255) if predicted_class.lower() in ['fire', 'smoke'] else (0, 255, 0)
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, color, 2)
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw bounding box (placeholder - would need actual detection model)
                    cv2.rectangle(frame, (width//4, height//4), 
                                (3*width//4, 3*height//4), color, 2)
            
            except Exception as e:
                self._log(f"Error processing frame {frame_count}: {e}", 'warning')
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Display
            if show_display:
                cv2.imshow('Fire Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if pbar:
                pbar.update(frame_skip)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
        if pbar:
            pbar.close()
        
        self._log(f"Processed {processed_frames}/{total_frames} frames")
        self._log(f"Detections: {len(results)}")
        
        # Generate video summary
        if results:
            fire_frames = [r for r in results if r['predicted_class'].lower() == 'fire']
            smoke_frames = [r for r in results if r['predicted_class'].lower() == 'smoke']
            
            summary = {
                'video_path': video_path,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'detections': len(results),
                'fire_detections': len(fire_frames),
                'smoke_detections': len(smoke_frames),
                'average_confidence': np.mean([r['confidence'] for r in results]) if results else 0,
                'results': results[:10]  # First 10 detections
            }
            
            # Save summary
            if output_path:
                summary_path = Path(output_path).with_suffix('.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                self._log(f"Video summary saved to {summary_path}")
        
        return results
    
    def process_webcam(self, camera_id=0, threshold=None, display_fps=True):
        """Real-time webcam fire detection"""
        self._log("Starting webcam detection (press 'q' to quit)", 'info')
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self._log(f"Cannot open camera {camera_id}", 'error')
            return
        
        fps_counter = 0
        fps_start = time.time()
        fps_display = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            fps_counter += 1
            
            # Predict every 3rd frame for performance
            if fps_counter % 3 == 0:
                try:
                    # Preprocess and predict
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed = self.preprocessor.preprocess(rgb_frame)
                    img_array = np.expand_dims(processed, axis=0)
                    predictions = self.model.predict(img_array, verbose=0)[0]
                    
                    predicted_idx = np.argmax(predictions)
                    confidence = float(predictions[predicted_idx])
                    predicted_class = self.class_names[predicted_idx]
                    
                    # Set alert flag
                    alert = (predicted_class.lower() in ['fire', 'smoke'] and 
                            confidence >= (threshold or self.confidence_threshold))
                    
                    # Draw results on frame
                    label = f"{predicted_class}: {confidence:.2%}"
                    color = (0, 0, 255) if alert else (0, 255, 0)
                    
                    # Semi-transparent overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                    
                    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, color, 2)
                    
                    if alert:
                        # Blinking alert
                        if int(time.time() * 2) % 2:
                            cv2.putText(frame, "ALERT!", (20, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                except Exception as e:
                    self._log(f"Prediction error: {e}", 'warning')
            
            # Calculate and display FPS
            if display_fps and fps_counter % 10 == 0:
                elapsed = time.time() - fps_start
                fps_display = fps_counter / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"FPS: {fps_display:.1f}", 
                       (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Fire Detection - Webcam', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self._log("Webcam detection stopped", 'info')


def generate_html_report(results, output_path, class_names):
    """Generate HTML report from detection results"""
    if not HAS_MATPLOTLIB:
        return
    
    # Create figures for report
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Class distribution
    class_dist = defaultdict(int)
    confidences = []
    for r in results:
        class_dist[r['predicted_class']] += 1
        confidences.append(r['confidence'])
    
    # Plot 1: Class distribution
    ax1 = axes[0, 0]
    classes = list(class_dist.keys())
    counts = list(class_dist.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    ax1.bar(classes, counts, color=colors)
    ax1.set_title('Detection Distribution')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Confidence histogram
    ax2 = axes[0, 1]
    ax2.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_title('Confidence Distribution')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Frequency')
    
    # Plot 3: Confidence by class
    ax3 = axes[1, 0]
    class_confidences = defaultdict(list)
    for r in results:
        class_confidences[r['predicted_class']].append(r['confidence'])
    
    for i, (cls, confs) in enumerate(class_confidences.items()):
        ax3.boxplot(confs, positions=[i], labels=[cls])
    ax3.set_title('Confidence by Class')
    ax3.set_ylabel('Confidence')
    
    # Plot 4: Inference time
    ax4 = axes[1, 1]
    inference_times = [r['inference_time'] for r in results]
    ax4.plot(inference_times, marker='o', linestyle='-', alpha=0.7)
    ax4.set_title('Inference Time per Image')
    ax4.set_xlabel('Image Index')
    ax4.set_ylabel('Time (seconds)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_path).with_suffix('.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fire Detection Report - {datetime.now():%Y-%m-%d %H:%M}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
            .alert {{ background: #e74c3c; color: white; padding: 10px; margin: 10px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #34495e; color: white; }}
            tr:nth-child(even) {{ background: #f2f2f2; }}
            .fire {{ color: #e74c3c; font-weight: bold; }}
            .smoke {{ color: #e67e22; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Fire Detection Report</h1>
            <p>Generated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Total Images</h3>
                <h2>{len(results)}</h2>
            </div>
            <div class="stat-box">
                <h3>Fire Detections</h3>
                <h2>{sum(1 for r in results if r['predicted_class'].lower() == 'fire')}</h2>
            </div>
            <div class="stat-box">
                <h3>Smoke Detections</h3>
                <h2>{sum(1 for r in results if r['predicted_class'].lower() == 'smoke')}</h2>
            </div>
            <div class="stat-box">
                <h3>Avg Confidence</h3>
                <h2>{np.mean(confidences):.1%}</h2>
            </div>
        </div>
        
        <h2>Analysis Plots</h2>
        <img src="{plot_path.name}" alt="Analysis Plots" style="max-width: 100%;">
        
        <h2>Detection Results</h2>
        <table>
            <tr>
                <th>Image</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Time (s)</th>
                <th>Status</th>
            </tr>
    """
    
    for r in results:
        pred_class = r['predicted_class']
        css_class = ''
        if pred_class.lower() == 'fire':
            css_class = 'fire'
        elif pred_class.lower() == 'smoke':
            css_class = 'smoke'
        
        status = '✅' if r['above_threshold'] else '⚠️'
        
        html += f"""
            <tr>
                <td>{r['file']}</td>
                <td class="{css_class}">{pred_class.upper()}</td>
                <td>{r['confidence']:.2%}</td>
                <td>{r['inference_time']:.3f}</td>
                <td>{status}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path


def send_email_alert(results, email_config):
    """Send email alert for fire/smoke detections"""
    if not HAS_EMAIL:
        print("⚠️  Email libraries not available")
        return False
    
    fire_detections = [r for r in results if r['predicted_class'].lower() in ['fire', 'smoke']]
    
    if not fire_detections:
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = email_config['from']
        msg['To'] = email_config['to']
        msg['Subject'] = f"🚨 FIRE ALERT: {len(fire_detections)} detections!"
        
        body = f"""
        <html>
        <body>
            <h2>Fire Detection Alert</h2>
            <p><strong>Time:</strong> {datetime.now():%Y-%m-%d %H:%M:%S}</p>
            <p><strong>Detections:</strong> {len(fire_detections)}</p>
            <ul>
        """
        
        for r in fire_detections[:5]:  # Top 5
            body += f"""
                <li>
                    <strong>{r['predicted_class'].upper()}</strong>: 
                    {r['confidence']:.1%} - {r['file']}
                </li>
            """
        
        body += """
            </ul>
            <p style="color: red; font-size: 18px;">
                <strong>IMMEDIATE ATTENTION REQUIRED!</strong>
            </p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
        
        print("✅ Alert email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


def format_output(result, format_type='text', class_names=None):
    """Enhanced output formatting"""
    if result is None:
        return "Error: Prediction failed"
    
    if format_type == 'json':
        return json.dumps(result, indent=2)
    
    elif format_type == 'jsonl':
        return json.dumps(result)
    
    elif format_type == 'csv':
        probs = result['probabilities']
        prob_values = ','.join([f"{prob:.4f}" for prob in probs.values()])
        return (f"{result['file']},{result['predicted_class']},"
                f"{result['confidence']:.4f},{result['above_threshold']},"
                f"{prob_values}")
    
    elif format_type == 'xml':
        xml = ['<detection>']
        xml.append(f'  <file>{result["file"]}</file>')
        xml.append(f'  <prediction>{result["predicted_class"]}</prediction>')
        xml.append(f'  <confidence>{result["confidence"]:.4f}</confidence>')
        xml.append('  <probabilities>')
        for cls, prob in result['probabilities'].items():
            xml.append(f'    <{cls}>{prob:.4f}</{cls}>')
        xml.append('  </probabilities>')
        xml.append('</detection>')
        return '\n'.join(xml)
    
    else:  # text format with color
        # ANSI color codes
        colors = {
            'fire': '\033[91m',    # Red
            'smoke': '\033[93m',   # Yellow
            'non_fire': '\033[92m', # Green
            'reset': '\033[0m'
        }
        
        pred_class = result['predicted_class']
        color = colors.get(pred_class.lower(), '')
        
        lines = [
            "=" * 60,
            f"File: {result['file']}",
            f"Path: {result['path']}",
            f"Prediction: {color}{pred_class.upper()}{colors['reset']}",
            f"Confidence: {result['confidence']:.2%}",
            f"Above Threshold: {result.get('above_threshold', 'N/A')}",
            f"Inference Time: {result['inference_time']:.3f}s",
            "-" * 60,
            "Class Probabilities:",
        ]
        for cls, prob in result['probabilities'].items():
            bar = '█' * int(prob * 30)
            lines.append(f"  {cls:12s}: {bar} {prob:.2%}")
        lines.append("=" * 60)
        
        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Forest Fire and Smoke Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python inference.py -m model.h5 -i image.jpg
  
  # Batch processing with JSON output
  python inference.py -m model.h5 -d /path/to/images -f json -o results.json
  
  # Video processing
  python inference.py -m model.h5 -v video.mp4 --output-video output.mp4
  
  # Webcam detection
  python inference.py -m model.h5 --webcam
  
  # High-confidence only with test augmentation
  python inference.py -m model.h5 -i image.jpg -t 0.8 --augment
  
  # Generate HTML report
  python inference.py -m model.h5 -d /path/to/images -f html -o report.html
  
  # Email alerts for detections
  python inference.py -m model.h5 -d /path/to/images --email-config email.json
        """
    )
    
    # Required arguments
    parser.add_argument('-m', '--model', required=True, type=str,
                       help='Path to trained model (.h5 file)')
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--image', nargs='+', type=str,
                            help='Path(s) to image file(s)')
    input_group.add_argument('-d', '--directory', type=str,
                            help='Path to directory containing images')
    input_group.add_argument('-v', '--video', type=str,
                            help='Path to video file')
    input_group.add_argument('--webcam', action='store_true',
                            help='Use webcam for real-time detection')
    
    # Optional arguments
    parser.add_argument('-s', '--size', type=int, default=64,
                       help='Input image size (default: 64)')
    parser.add_argument('-f', '--format', type=str, default='text',
                       choices=['text', 'json', 'jsonl', 'csv', 'xml', 'html'],
                       help='Output format (default: text)')
    parser.add_argument('-o', '--output', type=str,
                       help='Save results to file')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0, default: 0.5)')
    parser.add_argument('--device', type=str, default='/CPU:0',
                       choices=['/CPU:0', '/GPU:0'],
                       help='TensorFlow device (default: /CPU:0)')
    
    # Feature flags
    parser.add_argument('--augment', action='store_true',
                       help='Apply test-time augmentation')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively search directories for images')
    parser.add_argument('--csv-header', action='store_true',
                       help='Include CSV header row')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization (requires matplotlib)')
    parser.add_argument('--save-viz', type=str,
                       help='Save visualization to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    parser.add_argument('--alert', action='store_true',
                       help='Show alert for fire/smoke detection')
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable detection logging')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID for webcam detection (default: 0)')
    parser.add_argument('--frame-skip', type=int, default=30,
                       help='Process every Nth frame for video (default: 30)')
    parser.add_argument('--output-video', type=str,
                       help='Save annotated video output')
    parser.add_argument('--email-config', type=str,
                       help='JSON file with email configuration')
    
    args = parser.parse_args()
    
    # Validate input source
    if not any([args.image, args.directory, args.video, args.webcam]):
        parser.error("No input source specified. Use -i, -d, -v, or --webcam")
    
    # Initialize detector
    detector = ForestFireDetector(
        model_path=args.model,
        img_size=args.size,
        verbose=not args.quiet,
        confidence_threshold=args.threshold,
        device=args.device,
        enable_logging=not args.no_logging
    )
    
    if detector.model is None:
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    # Process input
    results = []
    
    if args.image:
        results = detector.predict_batch(
            args.image, 
            threshold=args.threshold,
            augment=args.augment
        )
    
    elif args.directory:
        results = detector.predict_directory(
            args.directory,
            threshold=args.threshold,
            augment=args.augment,
            recursive=args.recursive
        )
    
    elif args.video:
        results = detector.process_video(
            args.video,
            output_path=args.output_video,
            frame_skip=args.frame_skip,
            threshold=args.threshold
        )
        # Format video results differently
        if results:
            print(json.dumps(results, indent=2))
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
        sys.exit(0)
    
    elif args.webcam:
        detector.process_webcam(
            camera_id=args.camera_id,
            threshold=args.threshold
        )
        sys.exit(0)
    
    # Format and output results
    if not results:
        print("❌ No results generated")
        sys.exit(1)
    
    output_lines = []
    
    if args.format == 'csv':
        if args.csv_header:
            class_names = detector.class_names
            header = f"filename,prediction,confidence,above_threshold,{','.join(class_names)}"
            output_lines.append(header)
        for result in results:
            output_lines.append(format_output(result, 'csv'))
    
    elif args.format == 'jsonl':
        for result in results:
            output_lines.append(format_output(result, 'jsonl'))
    
    elif args.format == 'json':
        if len(results) == 1:
            output_lines.append(format_output(results[0], 'json'))
        else:
            output_lines.append(json.dumps(results, indent=2))
    
    elif args.format == 'xml':
        output_lines.append('<detections>')
        for result in results:
            output_lines.append(format_output(result, 'xml'))
        output_lines.append('</detections>')
    
    elif args.format == 'html':
        if args.output:
            generate_html_report(results, args.output, detector.class_names)
        else:
            html_path = f"detection_report_{datetime.now():%Y%m%d_%H%M%S}.html"
            generate_html_report(results, html_path, detector.class_names)
        print(f"✅ HTML report generated")
        sys.exit(0)
    
    else:  # text format
        for result in results:
            output_lines.append(format_output(result, 'text', detector.class_names))
            output_lines.append('')
    
    # Print or save output
    final_output = '\n'.join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(final_output)
        if not args.quiet:
            print(f"\n✅ Results saved to: {args.output}")
    else:
        print(final_output)
    
    # Alert for fire/smoke
    if args.alert:
        alert_classes = detector.labels_metadata.get('alert_classes', ['fire', 'smoke'])
        alerts = defaultdict(int)
        
        for result in results:
            pred_class = result['predicted_class'].lower()
            if pred_class in alert_classes and result['above_threshold']:
                alerts[pred_class] += 1
        
        if alerts:
            print("\n" + "!" * 60)
            for class_name, count in alerts.items():
                print(f"🚨 {class_name.upper()} DETECTED in {count} image(s)!")
            print("!" * 60)
            
            # Send email if configured
            if args.email_config and os.path.exists(args.email_config):
                with open(args.email_config, 'r') as f:
                    email_config = json.load(f)
                send_email_alert(results, email_config)
            
            sys.exit(2)
    
    # Print statistics
    if not args.quiet and detector.logger:
        stats = detector.logger.get_statistics()
        if stats:
            print("\n📊 Detection Statistics:")
            print(f"  Total: {stats['total_detections']}")
            print(f"  Avg Confidence: {stats['average_confidence']:.2%}")
            print(f"  Avg Time: {stats['average_inference_time']:.3f}s")
            print(f"  High Confidence: {stats['high_confidence_detections']}")
    
    # Visualize (only for single image)
    if args.visualize and len(results) == 1 and args.image:
        visualize_result(args.image[0], results[0], args.save_viz, detector.class_names)
    elif args.visualize:
        print("⚠️  Visualization only available for single image predictions")


def visualize_result(image_path, result, output_path=None, class_names=None):
    """Enhanced visualization with dynamic colors"""
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not available for visualization")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Cannot read image for visualization")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Image with prediction label
    ax1.imshow(img)
    ax1.set_title(f"Prediction: {result['predicted_class'].upper()}\n"
                  f"Confidence: {result['confidence']:.2%}",
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Probability bars with dynamic colors
    probs = result['probabilities']
    classes = list(probs.keys())
    values = list(probs.values())
    
    # Dynamic colors based on number of classes
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    ax2.barh(classes, values, color=colors, edgecolor='black')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(values):
        ax2.text(v + 0.02, i, f'{v:.1%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()