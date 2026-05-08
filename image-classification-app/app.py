#!/usr/bin/env python3
"""
Forest Fire Detection Web Application
Main Flask application with all routes and API endpoints
"""

import os
import json
import base64
import threading
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from collections import defaultdict

from flask import (Flask, render_template, request, jsonify, 
                   send_file, redirect, url_for, flash, session)

from config import Config
from detectors import ForestFireDetector, format_output, get_timestamp
from services import DetectionLogger, ReportService, EmailService

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# Create directories
Config.init_directories()

# Initialize services
detector = None
logger = DetectionLogger(str(Config.LOGS_DIR))
report_service = ReportService()
email_service = EmailService(Config.EMAIL_CONFIG)

# Store available models
available_models = []

def get_available_models():
    """Scan models directory for available models"""
    global available_models
    available_models = []
    
    models_dir = Config.MODELS_DIR
    for ext in Config.ALLOWED_MODEL_EXTENSIONS:
        for model_file in models_dir.glob(f'*.{ext}'):
            labels_file = model_file.with_suffix('.json')
            if labels_file.exists():
                available_models.append({
                    'name': model_file.name,
                    'path': str(model_file),
                    'size': model_file.stat().st_size,
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                })
    
    return available_models

def init_detector(model_path):
    """Initialize or reinitialize the detector"""
    global detector
    try:
        detector = ForestFireDetector(
            model_path=model_path,
            img_size=Config.DEFAULT_IMAGE_SIZE,
            confidence_threshold=Config.DEFAULT_CONFIDENCE_THRESHOLD,
            device=Config.DEFAULT_DEVICE,
            enable_logging=True
        )
        return True, "Model loaded successfully"
    except Exception as e:
        return False, str(e)

# Context processor for template variables
@app.context_processor
def inject_globals():
    return {
        'available_models': available_models,
        'detector_loaded': detector is not None,
        'detector_info': {
            'model_path': detector.model_path if detector else None,
            'class_names': detector.class_names if detector else [],
            'confidence_threshold': detector.confidence_threshold if detector else Config.DEFAULT_CONFIDENCE_THRESHOLD,
            'metrics': detector.get_metrics() if detector else {}
        } if detector else None,
        'Config': Config,  # Add this line to make Config available in all templates
        'default_image_size': Config.DEFAULT_IMAGE_SIZE  # Add this for convenience
    }

# ==================== Routes ====================

@app.route('/')
def index():
    """Home page"""
    get_available_models()
    return render_template('index.html')

@app.route('/models')
def list_models():
    """List available models"""
    models = get_available_models()
    return render_template('models.html', models=models)

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a model"""
    model_name = request.form.get('model_name') or request.json.get('model_name')
    
    if not model_name:
        flash('No model specified', 'error')
        return redirect(url_for('index'))
    
    model_path = Config.MODELS_DIR / model_name
    
    if not model_path.exists():
        flash(f'Model not found: {model_name}', 'error')
        return redirect(url_for('index'))
    
    success, message = init_detector(str(model_path))
    
    if success:
        flash(message, 'success')
    else:
        flash(f'Failed to load model: {message}', 'error')
    
    return redirect(url_for('index'))

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Upload a new model file"""
    if 'model_file' not in request.files:
        flash('No model file provided', 'error')
        return redirect(url_for('list_models'))
    
    model_file = request.files['model_file']
    labels_file = request.files.get('labels_file')
    
    if model_file.filename == '':
        flash('No model file selected', 'error')
        return redirect(url_for('list_models'))
    
    # Check model extension
    model_ext = model_file.filename.rsplit('.', 1)[1].lower()
    if model_ext not in Config.ALLOWED_MODEL_EXTENSIONS:
        flash(f'Invalid model format. Allowed: {Config.ALLOWED_MODEL_EXTENSIONS}', 'error')
        return redirect(url_for('list_models'))
    
    # Save model
    model_filename = secure_filename(model_file.filename)
    model_path = Config.MODELS_DIR / model_filename
    model_file.save(str(model_path))
    
    # Handle labels file
    if labels_file and labels_file.filename != '':
        labels_filename = secure_filename(labels_file.filename)
        labels_path = Config.MODELS_DIR / labels_filename
        labels_file.save(str(labels_path))
    else:
        # Create default labels file
        labels_path = model_path.with_suffix('.json')
        default_labels = {
            "class_names": ["non_fire", "fire", "smoke"],
            "num_classes": 3,
            "description": "Forest fire and smoke detection classes",
            "alert_classes": ["fire", "smoke"],
            "created": datetime.now().isoformat()
        }
        with open(labels_path, 'w') as f:
            json.dump(default_labels, f, indent=2)
    
    flash('Model uploaded successfully!', 'success')
    return redirect(url_for('list_models'))

@app.route('/single_image', methods=['GET', 'POST'])
def single_image():
    """Single image detection"""
    if detector is None:
        flash('Please load a model first', 'warning')
        return redirect(url_for('index'))
    
    result = None
    image_data = None
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file provided', 'error')
            return redirect(url_for('single_image'))
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            flash('No image selected', 'error')
            return redirect(url_for('single_image'))
        
        # Save uploaded image
        image_filename = secure_filename(image_file.filename)
        image_path = Config.UPLOADS_DIR / 'images' / f"{get_timestamp()}_{image_filename}"
        image_file.save(str(image_path))
        
        # Get parameters
        threshold = float(request.form.get('threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD))
        augment = request.form.get('augment') == 'on'
        
        # Predict
        try:
            result = detector.predict(str(image_path), threshold=threshold, augment=augment)
            
            # Convert image to base64 for display
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Log detection
            logger.log_detection(result, str(image_path))
            
        except Exception as e:
            flash(f'Prediction error: {e}', 'error')
            return redirect(url_for('single_image'))
    
    return render_template('single_image.html', result=result, image_data=image_data)

@app.route('/batch_image', methods=['GET', 'POST'])
def batch_image():
    """Batch image detection"""
    if detector is None:
        flash('Please load a model first', 'warning')
        return redirect(url_for('index'))
    
    results = None
    
    if request.method == 'POST':
        if 'images' not in request.files:
            flash('No image files provided', 'error')
            return redirect(url_for('batch_image'))
        
        image_files = request.files.getlist('images')
        
        if not image_files or image_files[0].filename == '':
            flash('No images selected', 'error')
            return redirect(url_for('batch_image'))
        
        # Save uploaded images
        image_paths = []
        for image_file in image_files:
            image_filename = secure_filename(image_file.filename)
            image_path = Config.UPLOADS_DIR / 'images' / f"{get_timestamp()}_{image_filename}"
            image_file.save(str(image_path))
            image_paths.append(str(image_path))
        
        # Get parameters
        threshold = float(request.form.get('threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD))
        augment = request.form.get('augment') == 'on'
        
        # Predict batch
        try:
            results = detector.predict_batch(image_paths, threshold=threshold, augment=augment)
            
            # Log detections
            for result in results['results']:
                logger.log_detection(result, result['path'])
            
        except Exception as e:
            flash(f'Batch prediction error: {e}', 'error')
            return redirect(url_for('batch_image'))
    
    return render_template('batch_image.html', results=results)

@app.route('/video', methods=['GET', 'POST'])
def video():
    """Video processing"""
    if detector is None:
        flash('Please load a model first', 'warning')
        return redirect(url_for('index'))
    
    summary = None
    
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file provided', 'error')
            return redirect(url_for('video'))
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            flash('No video selected', 'error')
            return redirect(url_for('video'))
        
        # Save uploaded video
        video_filename = secure_filename(video_file.filename)
        video_path = Config.UPLOADS_DIR / 'videos' / f"{get_timestamp()}_{video_filename}"
        video_file.save(str(video_path))
        
        # Get parameters
        threshold = float(request.form.get('threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD))
        frame_skip = int(request.form.get('frame_skip', Config.DEFAULT_FRAME_SKIP))
        
        # Process video
        try:
            output_path = Config.OUTPUTS_DIR / f"processed_{video_filename}"
            summary = detector.process_video(
                str(video_path),
                output_path=str(output_path),
                frame_skip=frame_skip,
                threshold=threshold
            )
            
        except Exception as e:
            flash(f'Video processing error: {e}', 'error')
            return redirect(url_for('video'))
    
    return render_template('video.html', summary=summary)

@app.route('/webcam')
def webcam():
    """Webcam detection page"""
    if detector is None:
        flash('Please load a model first', 'warning')
        return redirect(url_for('index'))
    
    return render_template('webcam.html')

@app.route('/api/webcam_frame', methods=['POST'])
def webcam_frame():
    """Process webcam frame via API"""
    if detector is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = detector.process_webcam_frame(frame)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Processing failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report')
def report():
    """View detection report"""
    if detector is None:
        flash('Please load a model first', 'warning')
        return redirect(url_for('index'))
    
    stats = detector.get_statistics()
    recent = logger.get_recent_detections(20)
    
    # Normalize recent detections for template
    normalized_recent = []
    for d in recent:
        normalized_recent.append({
            'timestamp': d.get('timestamp', ''),
            'image': d.get('file', d.get('image', 'unknown')),
            'prediction': d.get('predicted_class', d.get('prediction', 'unknown')),
            'confidence': d.get('confidence', 0)
        })
    
    return render_template('report.html', stats=stats, recent_detections=normalized_recent)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate and download report"""
    if detector is None:
        flash('Please load a model first', 'warning')
        return redirect(url_for('index'))
    
    report_type = request.form.get('report_type', 'html')
    
    # Get recent detections
    recent = logger.get_recent_detections(100)
    
    if not recent:
        flash('No detections to report', 'warning')
        return redirect(url_for('report'))
    
    # Normalize detections for report generation
    normalized_recent = []
    for d in recent:
        normalized_recent.append({
            'file': d.get('file', d.get('image', 'unknown')),
            'predicted_class': d.get('predicted_class', d.get('prediction', 'unknown')),
            'confidence': d.get('confidence', 0),
            'inference_time': d.get('inference_time', 0),
            'above_threshold': d.get('above_threshold', True),
            'timestamp': d.get('timestamp', ''),
            'probabilities': d.get('probabilities', {})
        })
    
    # Generate report
    timestamp = get_timestamp()
    
    if report_type == 'html':
        output_path = Config.OUTPUTS_DIR / f'report_{timestamp}.html'
        report_service.generate_html_report(normalized_recent, str(output_path))
    else:
        output_path = Config.OUTPUTS_DIR / f'report_{timestamp}.json'
        report_service.generate_json_report(normalized_recent, str(output_path))
    
    return send_file(
        output_path,
        as_attachment=True,
        download_name=output_path.name,
        mimetype='text/html' if report_type == 'html' else 'application/json'
    )

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Application settings"""
    if request.method == 'POST':
        # Update settings
        if detector:
            threshold = float(request.form.get('threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD))
            device = request.form.get('device', Config.DEFAULT_DEVICE)
            
            detector.confidence_threshold = threshold
            detector.device = device
        
        flash('Settings updated successfully', 'success')
        return redirect(url_for('settings'))
    
    return render_template(
        'settings.html',
        current_threshold=detector.confidence_threshold if detector else Config.DEFAULT_CONFIDENCE_THRESHOLD,
        current_device=detector.device if detector else Config.DEFAULT_DEVICE,
        default_image_size=Config.DEFAULT_IMAGE_SIZE,
        model_path=detector.model_path if detector else None,
        class_names=detector.class_names if detector else [],
        total_predictions=detector.metrics['total_predictions'] if detector else 0,
        average_time=detector.metrics['average_time'] if detector else 0
    )

@app.route('/api/stats')
def api_stats():
    """Get detection statistics via API"""
    if detector is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    stats = detector.get_statistics()
    metrics = detector.get_metrics()
    
    return jsonify({
        'statistics': stats,
        'metrics': metrics,
        'model_info': {
            'model_path': detector.model_path,
            'class_names': detector.class_names,
            'confidence_threshold': detector.confidence_threshold,
            'model_id': detector.model_id
        }
    })

@app.route('/api/recent_detections')
def api_recent_detections():
    """Get recent detections via API"""
    limit = request.args.get('limit', 10, type=int)
    detections = logger.get_recent_detections(limit)
    return jsonify(detections)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    import numpy as np
    import cv2
    
    # Initial model scan
    get_available_models()
    
    # Auto-load first model if available
    if available_models:
        model_path = available_models[0]['path']
        init_detector(model_path)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)