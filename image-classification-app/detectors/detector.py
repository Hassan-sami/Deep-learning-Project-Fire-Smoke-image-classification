import os
import time
import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
from tensorflow.keras import layers

from .preprocessor import ImagePreprocessor
from .utils import load_labels, get_image_size_from_labels, generate_detection_id, get_timestamp

class ForestFireDetector:
    """Enhanced Forest Fire and Smoke Detection Model"""
    
    def __init__(self, model_path, img_size=None, confidence_threshold=0.5, 
                 device='/CPU:0', enable_logging=True):
        """
        Initialize detector.
        
        Args:
            model_path: Path to model file (.h5 or .keras)
            img_size: Input image size (if None, will try to read from labels JSON)
            confidence_threshold: Minimum confidence for positive detection
            device: TensorFlow device
            enable_logging: Enable detection logging
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.enable_logging = enable_logging
        self.model_id = generate_detection_id()
        
        # Load class names and metadata
        self.class_names, self.labels_metadata = load_labels(model_path)
        
        # Determine image size
        if img_size is None:
            # Try to get image size from labels JSON
            detected_size = get_image_size_from_labels(model_path)
            if detected_size is not None:
                self.img_size = detected_size
                print(f"Using image size from labels JSON: {self.img_size}x{self.img_size}")
            else:
                # Fallback to default
                self.img_size = 64
                print(f"No image size found in labels JSON, using default: {self.img_size}x{self.img_size}")
        else:
            self.img_size = img_size
            print(f"Using specified image size: {self.img_size}x{self.img_size}")
        
        # Setup components
        self.preprocessor = ImagePreprocessor(self.img_size)
        
        # Load model
        self.model = self._load_model()
        
        # Detection history
        self.detection_history = []
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'total_time': 0,
            'average_time': 0,
            'class_counts': defaultdict(int),
            'high_confidence_count': 0,
            'image_size': self.img_size
        }
    
    def _load_model(self):
        """Load model with device placement"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with tf.device(self.device):
            try:
                model = load_model(self.model_path)
            except:
                model = load_model(self.model_path, compile=False)
            
            # Get model's expected input shape
            expected_input_shape = model.input_shape
            
            # Extract expected size from model
            if len(expected_input_shape) == 4:  # (batch, height, width, channels)
                model_height = expected_input_shape[1]
                model_width = expected_input_shape[2]
                
                # Warn if image size doesn't match model input
                if model_height is not None and model_width is not None:
                    if model_height != self.img_size or model_width != self.img_size:
                        print(f"Warning: Configured image size ({self.img_size}x{self.img_size}) "
                              f"doesn't match model input shape ({model_height}x{model_width})")
                        print(f"Adjusting image size to match model: {model_height}x{model_width}")
                        self.img_size = model_height
                        self.preprocessor = ImagePreprocessor(self.img_size)
                        self.metrics['image_size'] = self.img_size
            
            # Verify model output matches labels
            if model.output_shape[-1] != len(self.class_names):
                print(f"Warning: Model output ({model.output_shape[-1]}) "
                      f"doesn't match labels ({len(self.class_names)})")
            
            return model
    
    # ... rest of the detector code remains the same ...
    
    def predict(self, image_path, threshold=None, augment=False):
        """Predict single image"""
        if threshold is None:
            threshold = self.confidence_threshold
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            start_time = time.time()
            
            # Preprocess image
            img_array = self.preprocessor.preprocess(image_path, augment=augment)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Test-time augmentation
            if augment:
                predictions = []
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
                'id': generate_detection_id(),
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
                },
                'timestamp': datetime.now().isoformat(),
                'model_id': self.model_id
            }
            
            # Update metrics
            self._update_metrics(result)
            
            return result
            
        except Exception as e:
            raise Exception(f"Prediction error: {e}")
    
    def predict_batch(self, image_paths, threshold=None, augment=False):
        """Predict multiple images"""
        results = []
        errors = []
        
        for i, path in enumerate(image_paths):
            try:
                result = self.predict(path, threshold=threshold, augment=augment)
                if result:
                    results.append(result)
            except Exception as e:
                errors.append({'file': os.path.basename(path), 'error': str(e)})
        
        return {
            'results': results,
            'errors': errors,
            'total': len(image_paths),
            'successful': len(results),
            'failed': len(errors)
        }
    
    def predict_directory(self, directory_path, threshold=None, augment=False, recursive=False):
        """Predict all images in directory"""
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
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
        
        return self.predict_batch(image_paths, threshold=threshold, augment=augment)
    
    def process_video(self, video_path, output_path=None, frame_skip=30, threshold=None):
        """Process video for fire/smoke detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames
            if frame_count % frame_skip != 0:
                continue
            
            processed_frames += 1
            
            try:
                # Predict frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    cv2.rectangle(frame, (width//4, height//4), 
                                (3*width//4, 3*height//4), color, 2)
            
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
            
            if writer:
                writer.write(frame)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        # Generate summary
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
            'timeline': results
        }
        
        return summary
    
    def process_webcam_frame(self, frame, threshold=None):
        """Process single webcam frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = self.preprocessor.preprocess(rgb_frame)
            img_array = np.expand_dims(processed, axis=0)
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            predicted_class = self.class_names[predicted_idx]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    self.class_names[i]: float(predictions[i])
                    for i in range(len(self.class_names))
                }
            }
        except Exception as e:
            return None
    
    def _update_metrics(self, result):
        """Update detection metrics"""
        self.metrics['total_predictions'] += 1
        self.metrics['total_time'] += result['inference_time']
        self.metrics['average_time'] = (self.metrics['total_time'] / 
                                        self.metrics['total_predictions'])
        self.metrics['class_counts'][result['predicted_class']] += 1
        
        if result['confidence'] > 0.8:
            self.metrics['high_confidence_count'] += 1
        
        self.detection_history.append(result)
    
    def get_metrics(self):
        """Get detection metrics"""
        return {
            **self.metrics,
            'class_counts': dict(self.metrics['class_counts']),
            'total_history': len(self.detection_history)
        }
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        if not self.detection_history:
            return {}
        
        stats = {
            'total_detections': len(self.detection_history),
            'average_confidence': np.mean([d['confidence'] for d in self.detection_history]),
            'average_inference_time': np.mean([d['inference_time'] for d in self.detection_history]),
            'class_distribution': dict(self.metrics['class_counts']),
            'high_confidence_detections': self.metrics['high_confidence_count'],
            'alert_detections': sum(1 for d in self.detection_history 
                                   if d['predicted_class'].lower() in ['fire', 'smoke'] 
                                   and d['confidence'] > 0.7)
        }
        
        return stats