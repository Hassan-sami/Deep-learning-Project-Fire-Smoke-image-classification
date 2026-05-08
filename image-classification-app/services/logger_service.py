import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

class DetectionLogger:
    """Logging system for detection results"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('ForestFireDetection')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"detection_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        
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
            'file': image_path or result.get('file', 'unknown'),
            'image': image_path or result.get('file', 'unknown'),  # Keep both for compatibility
            'predicted_class': result.get('predicted_class', 'unknown'),
            'prediction': result.get('predicted_class', 'unknown'),  # Keep both for compatibility
            'confidence': result.get('confidence', 0),
            'probabilities': result.get('probabilities', {}),
            'inference_time': result.get('inference_time', 0),
            'above_threshold': result.get('above_threshold', True)
        }
        self.history['detections'].append(detection_entry)
        self._save_history()
        
        self.logger.info(
            f"Detection: {detection_entry['predicted_class']} "
            f"(confidence: {detection_entry['confidence']:.2%})"
        )
        
        # Alert for high-confidence fire/smoke
        if detection_entry['predicted_class'].lower() in ['fire', 'smoke']:
            if detection_entry['confidence'] > 0.7:
                self.logger.warning(
                    f"HIGH CONFIDENCE {detection_entry['predicted_class'].upper()} DETECTION"
                )

    def get_statistics(self):
        """Get detection statistics"""
        if not self.history['detections']:
            return {}
        
        detections = self.history['detections']
        stats = {
            'total_detections': len(detections),
            'average_confidence': np.mean([d.get('confidence', 0) for d in detections]),
            'average_inference_time': np.mean([d.get('inference_time', 0) for d in detections]),
            'class_distribution': defaultdict(int),
            'high_confidence_detections': 0
        }
        
        for d in detections:
            pred = d.get('predicted_class', d.get('prediction', 'unknown'))
            stats['class_distribution'][pred] += 1
            if d.get('confidence', 0) > 0.8:
                stats['high_confidence_detections'] += 1
        
        stats['class_distribution'] = dict(stats['class_distribution'])
        self.history['statistics'] = stats
        self._save_history()
        
        return stats
    
    # def get_statistics(self):
    #     """Get detection statistics"""
    #     if not self.history['detections']:
    #         return {}
        
    #     detections = self.history['detections']
    #     stats = {
    #         'total_detections': len(detections),
    #         'average_confidence': np.mean([d['confidence'] for d in detections]),
    #         'average_inference_time': np.mean([d.get('inference_time', 0) for d in detections]),
    #         'class_distribution': defaultdict(int),
    #         'high_confidence_detections': 0
    #     }
        
    #     for d in detections:
    #         stats['class_distribution'][d['prediction']] += 1
    #         if d['confidence'] > 0.8:
    #             stats['high_confidence_detections'] += 1
        
    #     stats['class_distribution'] = dict(stats['class_distribution'])
    #     self.history['statistics'] = stats
    #     self._save_history()
        
    #     return stats
    
    def get_recent_detections(self, limit=10):
        """Get recent detections"""
        detections = self.history['detections']
        return detections[-limit:] if detections else []