import cv2
import numpy as np

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
            brightness = 0.8 + np.random.random() * 0.4
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