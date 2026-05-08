#!/usr/bin/env python3
"""
Forest Fire and Smoke Detection - Command Line Inference Tool
Usage: python inference.py --model model.h5 --input image.jpg
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ForestFireDetector:
    """Forest Fire and Smoke Detection Model"""
    
    def __init__(self, model_path, img_size=64, verbose=True):
        """
        Initialize detector.
        
        Args:
            model_path: Path to .h5 model file
            img_size: Input image size (default: 64)
            verbose: Print status messages
        """
        self.img_size = img_size
        self.model_path = model_path
        self.verbose = verbose
        
        # Load labels from corresponding JSON file
        self.class_names = self._load_labels()
        
        # Load model
        self.model = self._load_model()
        
    def _log(self, message, level='info'):
        """Logging helper"""
        if self.verbose:
            prefix = {
                'info': 'ℹ️',
                'success': '✅',
                'warning': '⚠️',
                'error': '❌'
            }
            print(f"{prefix.get(level, '')} {message}")
    
    def _get_labels_path(self):
        """Get the expected labels JSON file path based on model path"""
        model_path = Path(self.model_path)
        # Replace .h5 extension with .json
        labels_path = model_path.with_suffix('.json')
        return labels_path
    
    def _load_labels(self):
        """Load class labels from JSON file"""
        labels_path = self._get_labels_path()
        
        if not os.path.exists(labels_path):
            self._log(f"Labels file not found: {labels_path}", 'error')
            self._log("Creating default labels file...", 'warning')
            # Create default labels file
            default_labels = {
                "class_names": ["non_fire", "fire", "smoke"],
                "num_classes": 3,
                "description": "Forest fire and smoke detection classes"
            }
            try:
                with open(labels_path, 'w') as f:
                    json.dump(default_labels, f, indent=2)
                self._log(f"Default labels saved to: {labels_path}", 'success')
                return default_labels["class_names"]
            except Exception as e:
                self._log(f"Could not create labels file: {e}", 'error')
                self._log("Using hardcoded default labels", 'warning')
                return ["non_fire", "fire", "smoke"]
        
        try:
            with open(labels_path, 'r') as f:
                labels_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(labels_data, list):
                class_names = labels_data
            elif isinstance(labels_data, dict):
                # Try common keys for class names
                class_names = labels_data.get('class_names') or \
                             labels_data.get('classes') or \
                             labels_data.get('labels') or \
                             labels_data.get('categories')
                
                if class_names is None:
                    self._log("JSON file found but no class names detected", 'error')
                    self._log("Expected keys: 'class_names', 'classes', 'labels', or 'categories'", 'warning')
                    return ["non_fire", "fire", "smoke"]
            else:
                self._log("Invalid labels format in JSON file", 'error')
                return ["non_fire", "fire", "smoke"]
            
            if not class_names:
                self._log("Empty class names in labels file", 'error')
                return ["non_fire", "fire", "smoke"]
            
            self._log(f"Loaded {len(class_names)} classes from {labels_path}", 'success')
            self._log(f"Classes: {class_names}", 'info')
            return class_names
            
        except json.JSONDecodeError as e:
            self._log(f"Invalid JSON in labels file: {e}", 'error')
            return ["non_fire", "fire", "smoke"]
        except Exception as e:
            self._log(f"Error loading labels: {e}", 'error')
            return ["non_fire", "fire", "smoke"]
    
    def _load_model(self):
        """Load model with version compatibility"""
        if not os.path.exists(self.model_path):
            self._log(f"Model not found: {self.model_path}", 'error')
            return None
        
        loading_methods = [
            # Method 1: Standard loading
            lambda: load_model(self.model_path),
            
            # Method 2: Load without compilation
            lambda: load_model(self.model_path, compile=False),
            
            # Method 3: Load with custom objects
            lambda: tf.keras.models.load_model(self.model_path, compile=False),
        ]
        
        for i, method in enumerate(loading_methods, 1):
            try:
                self._log(f"Loading model (method {i})...")
                model = method()
                self._log(f"Model loaded successfully!", 'success')
                return model
            except Exception as e:
                if self.verbose:
                    self._log(f"Method {i} failed: {str(e)[:80]}...", 'warning')
        
        # Fallback: Rebuild model and load weights
        self._log("Attempting to rebuild model architecture...", 'warning')
        try:
            num_classes = len(self.class_names)
            model = self._build_model(num_classes)
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
        """Build model architecture matching training"""
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        
        if num_classes is None:
            num_classes = len(self.class_names)
        
        model = tf.keras.models.Sequential([
            Conv2D(16, (3, 3), activation='relu', 
                   input_shape=(self.img_size, self.img_size, 3)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=3, strides=2),
            Conv2D(16, (3, 3), activation='relu'),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=3, strides=2),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    def predict(self, image_path):
        """
        Predict on a single image.
        
        Returns:
            dict with prediction results or None on error
        """
        if self.model is None:
            self._log("Model not loaded", 'error')
            return None
        
        if not os.path.exists(image_path):
            self._log(f"Image not found: {image_path}", 'error')
            return None
        
        try:
            start_time = time.time()
            
            # Load and preprocess
            img = load_img(image_path, target_size=(self.img_size, self.img_size))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)[0]
            elapsed = time.time() - start_time
            
            # Get results
            predicted_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(predictions[predicted_idx])
            
            result = {
                'file': os.path.basename(image_path),
                'path': os.path.abspath(image_path),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'inference_time': elapsed,
                'probabilities': {
                    self.class_names[i]: float(predictions[i])
                    for i in range(len(self.class_names))
                }
            }
            
            return result
            
        except Exception as e:
            self._log(f"Prediction error: {e}", 'error')
            return None
    
    def predict_batch(self, image_paths):
        """Predict on multiple images"""
        results = []
        
        for i, path in enumerate(image_paths, 1):
            self._log(f"[{i}/{len(image_paths)}] Processing: {os.path.basename(path)}")
            result = self.predict(path)
            if result:
                results.append(result)
        
        return results
    
    def predict_directory(self, directory_path):
        """Predict on all images in directory"""
        if not os.path.isdir(directory_path):
            self._log(f"Directory not found: {directory_path}", 'error')
            return []
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for file in sorted(os.listdir(directory_path)):
            if Path(file).suffix.lower() in extensions:
                image_paths.append(os.path.join(directory_path, file))
        
        if not image_paths:
            self._log(f"No images found in {directory_path}", 'warning')
            return []
        
        self._log(f"Found {len(image_paths)} images")
        return self.predict_batch(image_paths)


def format_output(result, format_type='text'):
    """Format prediction result for output"""
    if result is None:
        return "Error: Prediction failed"
    
    if format_type == 'json':
        return json.dumps(result, indent=2)
    
    elif format_type == 'csv':
        probs = result['probabilities']
        prob_values = ','.join([f"{prob:.4f}" for prob in probs.values()])
        return (f"{result['file']},{result['predicted_class']},"
                f"{result['confidence']:.4f},{prob_values}")
    
    else:  # text format
        lines = [
            "=" * 60,
            f"File: {result['file']}",
            f"Path: {result['path']}",
            f"Prediction: {result['predicted_class'].upper()}",
            f"Confidence: {result['confidence']:.2%}",
            f"Inference Time: {result['inference_time']:.3f}s",
            "-" * 60,
            "Class Probabilities:",
        ]
        for cls, prob in result['probabilities'].items():
            bar = '█' * int(prob * 30)
            lines.append(f"  {cls:12s}: {bar} {prob:.2%}")
        lines.append("=" * 60)
        
        return '\n'.join(lines)


def visualize_result(image_path, result, output_path=None):
    """Visualize prediction result"""
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
    
    # Probability bars - dynamic colors based on number of classes
    probs = result['probabilities']
    classes = list(probs.keys())
    values = list(probs.values())
    
    # Generate colors dynamically
    import matplotlib.cm as cm
    colors = cm.Set3(np.linspace(0, 1, len(classes)))
    
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


def main():
    parser = argparse.ArgumentParser(
        description='Forest Fire and Smoke Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python inference.py -m model.h5 -i image.jpg

  # Batch prediction with JSON output
  python inference.py -m model.h5 -i image1.jpg image2.jpg -f json

  # Directory prediction with visualization
  python inference.py -m model.h5 -d /path/to/images -v

  # Save results to file
  python inference.py -m model.h5 -i image.jpg -o results.json -f json

  # CSV output for batch processing
  python inference.py -m model.h5 -d /path/to/images -f csv --csv-header

Note:
  Labels are automatically loaded from a JSON file with the same name as the model.
  Example: model.h5 -> model.json
  
  Expected JSON format:
  {
    "class_names": ["non_fire", "fire", "smoke"],
    "num_classes": 3,
    "description": "Optional description"
  }
  
  Or simply:
  ["non_fire", "fire", "smoke"]
        """
    )
    
    # Required arguments
    parser.add_argument('-m', '--model', required=True, type=str,
                       help='Path to trained model (.h5 file)')
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--image', nargs='+', type=str,
                            help='Path(s) to image file(s)')
    input_group.add_argument('-d', '--directory', type=str,
                            help='Path to directory containing images')
    input_group.add_argument('-v', '--video', type=str,
                            help='Path to video file (experimental)')
    
    # Optional arguments
    parser.add_argument('-s', '--size', type=int, default=64,
                       help='Input image size (default: 64)')
    parser.add_argument('-f', '--format', type=str, default='text',
                       choices=['text', 'json', 'csv'],
                       help='Output format (default: text)')
    parser.add_argument('-o', '--output', type=str,
                       help='Save results to file')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization (requires matplotlib)')
    parser.add_argument('--save-viz', type=str,
                       help='Save visualization to file')
    parser.add_argument('--csv-header', action='store_true',
                       help='Include CSV header row')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    parser.add_argument('--alert', action='store_true',
                       help='Show alert for fire/smoke detection')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ForestFireDetector(
        model_path=args.model,
        img_size=args.size,
        verbose=not args.quiet
    )
    
    if detector.model is None:
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    # Process input
    results = []
    
    if args.image:
        # Single or multiple images
        results = detector.predict_batch(args.image)
    
    elif args.directory:
        # Directory of images
        results = detector.predict_directory(args.directory)
    
    elif args.video:
        print("⚠️  Video processing not implemented in CLI version")
        print("   Use the Python API for video processing")
        sys.exit(0)
    
    # Format and output results
    if not results:
        print("❌ No results generated")
        sys.exit(1)
    
    # Prepare output
    output_lines = []
    
    if args.format == 'csv':
        if args.csv_header:
            # Dynamic CSV header based on class names
            class_names = detector.class_names
            header = f"filename,prediction,confidence,{','.join(class_names)}"
            output_lines.append(header)
        for result in results:
            output_lines.append(format_output(result, 'csv'))
    
    elif args.format == 'json':
        if len(results) == 1:
            output_lines.append(format_output(results[0], 'json'))
        else:
            output_lines.append(json.dumps(results, indent=2))
    
    else:  # text format
        for result in results:
            output_lines.append(format_output(result, 'text'))
            output_lines.append('')  # Blank line between results
    
    # Print or save output
    final_output = '\n'.join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(final_output)
        if not args.quiet:
            print(f"\n✅ Results saved to: {args.output}")
    else:
        print(final_output)
    
    # Dynamic alert for any critical classes
    if args.alert:
        # Check for classes that might indicate emergencies
        alert_classes = {'fire', 'smoke', 'flame', 'burning', 'wildfire'}
        alerts = {}
        
        for result in results:
            pred_class = result['predicted_class'].lower()
            if pred_class in alert_classes:
                alerts[pred_class] = alerts.get(pred_class, 0) + 1
        
        if alerts:
            print("\n" + "!" * 60)
            for class_name, count in alerts.items():
                print(f"⚠️  {class_name.upper()} DETECTED in {count} image(s)!")
            print("!" * 60)
            
            # Exit with error code for pipeline integration
            sys.exit(2)
    
    # Visualize (only for single image)
    if args.visualize and len(results) == 1 and args.image:
        visualize_result(args.image[0], results[0], args.save_viz)
    elif args.visualize:
        print("⚠️  Visualization only available for single image predictions")


if __name__ == '__main__':
    main()