import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'forest-fire-detection-secret-key')
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / 'models'
    UPLOADS_DIR = BASE_DIR / 'uploads'
    IMAGES_DIR = UPLOADS_DIR / 'images'
    VIDEOS_DIR = UPLOADS_DIR / 'videos'
    TEMP_DIR = UPLOADS_DIR / 'temp'
    LOGS_DIR = BASE_DIR / 'logs'
    OUTPUTS_DIR = BASE_DIR / 'outputs'
    
    # Upload limits
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    ALLOWED_MODEL_EXTENSIONS = {'h5', 'keras', 'hdf5'}
    
    # Detection defaults
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_IMAGE_SIZE = 64
    DEFAULT_FRAME_SKIP = 30
    DEFAULT_DEVICE = '/CPU:0'
    
    # Email (configure as needed)
    EMAIL_CONFIG = {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': '',
        'password': '',
        'from_email': '',
        'to_email': '',
        'alert_threshold': 0.7
    }
    
    @classmethod
    def init_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.MODELS_DIR,
            cls.IMAGES_DIR,
            cls.VIDEOS_DIR,
            cls.TEMP_DIR,
            cls.LOGS_DIR,
            cls.OUTPUTS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories