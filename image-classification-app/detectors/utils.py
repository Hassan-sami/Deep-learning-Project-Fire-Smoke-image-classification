import json
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path

def load_labels(model_path):
    """Load class labels from JSON file with enhanced error handling"""
    model_path = Path(model_path)
    labels_path = model_path.with_suffix('.json')
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    # Extract class names from various formats
    if isinstance(labels_data, list):
        class_names = labels_data
        metadata = {'class_names': class_names}
    elif isinstance(labels_data, dict):
        class_names = (labels_data.get('class_names') or 
                      labels_data.get('classes') or 
                      labels_data.get('labels'))
        metadata = labels_data
        if class_names is None:
            raise ValueError("No class names found in JSON")
    else:
        raise ValueError("Invalid labels format")
    
    return class_names, metadata


def get_image_size_from_labels(model_path):
    """Extract image size from labels JSON file"""
    model_path = Path(model_path)
    labels_path = model_path.with_suffix('.json')
    
    if not labels_path.exists():
        return None
    
    try:
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        if isinstance(labels_data, dict):
            # Check for different possible key names
            img_size = (labels_data.get('img_size') or 
                       labels_data.get('image_size') or 
                       labels_data.get('input_size') or
                       labels_data.get('size'))
            
            if img_size is not None:
                return int(img_size)
            
            # Check for width/height format
            img_width = labels_data.get('img_width') or labels_data.get('image_width')
            img_height = labels_data.get('img_height') or labels_data.get('image_height')
            
            if img_width is not None and img_height is not None:
                # Return width (assuming square input, or return tuple)
                return int(img_width)
        
        return None
    except Exception as e:
        print(f"Warning: Could not read image size from labels: {e}")
        return None


def format_output(result, format_type='text'):
    """Format detection result for output"""
    if format_type == 'json':
        return json.dumps(result, indent=2)
    
    elif format_type == 'text':
        lines = [
            "=" * 60,
            f"File: {result['file']}",
            f"Prediction: {result['predicted_class'].upper()}",
            f"Confidence: {result['confidence']:.2%}",
            f"Inference Time: {result.get('inference_time', 0):.3f}s",
            "-" * 60,
            "Class Probabilities:"
        ]
        for cls, prob in result['probabilities'].items():
            bar = '█' * int(prob * 30)
            lines.append(f"  {cls:12s}: {bar} {prob:.2%}")
        lines.append("=" * 60)
        return '\n'.join(lines)
    
    return str(result)

def generate_detection_id():
    """Generate unique detection ID"""
    return uuid.uuid4().hex[:12]

def get_timestamp():
    """Get formatted timestamp"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')