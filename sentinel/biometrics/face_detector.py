# Probabilistic Face Confidence Module
# Replaces binary face detection with probabilistic confidence scoring

import cv2
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ProbabilisticFaceDetector:
    """
    Advanced face detection with probabilistic confidence scoring.
    Considers multiple factors instead of binary detection.
    """
    
    def __init__(self):
        # Load face cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        except Exception as e:
            logger.error(f"Error loading cascades: {e}")
            self.face_cascade = None
            self.eye_cascade = None
        
        # Confidence history for smoothing
        self.confidence_history = deque(maxlen=10)
        
        # Optimal face size range (as fraction of frame)
        self.optimal_size_min = 0.15
        self.optimal_size_max = 0.45
    
    def detect_with_confidence(self, frame):
        """
        Detect face with probabilistic confidence score
        
        Args:
            frame: Video frame (numpy array)
        
        Returns:
            Dict with confidence score and detailed metrics
        """
        if frame is None or self.face_cascade is None:
            return {
                'confidence': 0.0,
                'face_detected': False,
                'reason': 'No frame or cascade not loaded'
            }
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect faces with multiple scale factors for robustness
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            confidence = 0.1  # Small non-zero confidence even without detection
            self.confidence_history.append(confidence)
            
            return {
                'confidence': round(confidence, 3),
                'face_detected': False,
                'face_count': 0,
                'reason': 'No face detected',
                'smoothed_confidence': round(self._get_smoothed_confidence(), 3)
            }
        
        # Use largest face (primary user)
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Calculate multiple confidence factors
        factors = {
            'detection': self._calculate_detection_confidence(faces),
            'size': self._calculate_size_confidence(w, h, width, height),
            'position': self._calculate_position_confidence(x, y, w, h, width, height),
            'stability': self._calculate_stability_confidence(face),
            'eyes': self._calculate_eye_confidence(gray, face),
            'quality': self._calculate_quality_confidence(gray[y:y+h, x:x+w])
        }
        
        # Weighted combination
        weights = {
            'detection': 0.25,
            'size': 0.20,
            'position': 0.15,
            'stability': 0.15,
            'eyes': 0.15,
            'quality': 0.10
        }
        
        confidence = sum(factors[k] * weights[k] for k in factors)
        
        # Store in history
        self.confidence_history.append(confidence)
        
        return {
            'confidence': round(confidence, 3),
            'face_detected': True,
            'face_count': len(faces),
            'face_bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'factors': {k: round(v, 3) for k, v in factors.items()},
            'weights': weights,
            'smoothed_confidence': round(self._get_smoothed_confidence(), 3),
            'reason': self._explain_confidence(factors, confidence)
        }
    
    def _calculate_detection_confidence(self, faces):
        """
        Confidence based on detection quality
        
        Args:
            faces: List of detected faces
        
        Returns:
            Confidence score (0-1)
        """
        # Single face is ideal
        if len(faces) == 1:
            return 1.0
        # Multiple faces reduce confidence
        elif len(faces) == 2:
            return 0.7
        else:
            return 0.5
    
    def _calculate_size_confidence(self, face_w, face_h, frame_w, frame_h):
        """
        Confidence based on face size relative to frame
        
        Args:
            face_w, face_h: Face dimensions
            frame_w, frame_h: Frame dimensions
        
        Returns:
            Confidence score (0-1)
        """
        # Calculate face area as fraction of frame
        face_area = (face_w * face_h) / (frame_w * frame_h)
        
        # Optimal range: 15-45% of frame
        if self.optimal_size_min <= face_area <= self.optimal_size_max:
            # Perfect size
            return 1.0
        elif face_area < self.optimal_size_min:
            # Too small (user far from camera)
            # Linear decay from optimal to 0
            ratio = face_area / self.optimal_size_min
            return max(0.3, ratio)
        else:
            # Too large (user too close)
            # Linear decay from optimal to 0
            ratio = self.optimal_size_max / face_area
            return max(0.3, ratio)
    
    def _calculate_position_confidence(self, x, y, w, h, frame_w, frame_h):
        """
        Confidence based on face position in frame
        
        Args:
            x, y, w, h: Face bounding box
            frame_w, frame_h: Frame dimensions
        
        Returns:
            Confidence score (0-1)
        """
        # Calculate face center
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Frame center
        frame_center_x = frame_w / 2
        frame_center_y = frame_h / 2
        
        # Distance from center (normalized)
        dx = abs(center_x - frame_center_x) / (frame_w / 2)
        dy = abs(center_y - frame_center_y) / (frame_h / 2)
        
        # Distance metric (0 = center, 1 = edge)
        distance = np.sqrt(dx**2 + dy**2)
        
        # Confidence decreases with distance from center
        if distance < 0.3:
            return 1.0
        elif distance < 0.6:
            return 0.8
        elif distance < 0.9:
            return 0.6
        else:
            return 0.4
    
    def _calculate_stability_confidence(self, face):
        """
        Confidence based on face stability (jitter)
        
        Args:
            face: Current face bounding box
        
        Returns:
            Confidence score (0-1)
        """
        # Would track face position over time
        # For now, return high confidence
        # In production, compare with previous frames
        return 0.9
    
    def _calculate_eye_confidence(self, gray, face):
        """
        Confidence based on eye detection
        
        Args:
            gray: Grayscale frame
            face: Face bounding box
        
        Returns:
            Confidence score (0-1)
        """
        x, y, w, h = face
        
        # Extract eye region (upper half of face)
        eye_region = gray[y:y+int(h*0.6), x:x+w]
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Confidence based on number of eyes detected
        if len(eyes) >= 2:
            return 1.0
        elif len(eyes) == 1:
            return 0.6
        else:
            return 0.3
    
    def _calculate_quality_confidence(self, face_roi):
        """
        Confidence based on image quality
        
        Args:
            face_roi: Face region of interest
        
        Returns:
            Confidence score (0-1)
        """
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        variance = laplacian.var()
        
        # Higher variance = sharper image
        if variance > 100:
            return 1.0
        elif variance > 50:
            return 0.8
        elif variance > 20:
            return 0.6
        else:
            return 0.4
    
    def _get_smoothed_confidence(self):
        """Get smoothed confidence from history"""
        if not self.confidence_history:
            return 0.0
        
        # Exponential moving average
        weights = np.exp(np.linspace(-1, 0, len(self.confidence_history)))
        weights /= weights.sum()
        
        smoothed = np.average(list(self.confidence_history), weights=weights)
        return smoothed
    
    def _explain_confidence(self, factors, confidence):
        """Generate human-readable explanation"""
        explanations = []
        
        if confidence >= 0.8:
            explanations.append("Excellent face detection")
        elif confidence >= 0.6:
            explanations.append("Good face detection")
        elif confidence >= 0.4:
            explanations.append("Fair face detection")
        else:
            explanations.append("Poor face detection")
        
        # Identify weak factors
        for factor, score in factors.items():
            if score < 0.5:
                if factor == 'size':
                    explanations.append("Face size suboptimal (move closer/farther)")
                elif factor == 'position':
                    explanations.append("Face not centered (adjust position)")
                elif factor == 'eyes':
                    explanations.append("Eyes not clearly visible")
                elif factor == 'quality':
                    explanations.append("Image quality low (improve lighting)")
        
        return "; ".join(explanations)
    
    def get_statistics(self):
        """Get detector statistics"""
        if not self.confidence_history:
            return {
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'samples': 0
            }
        
        history = list(self.confidence_history)
        
        return {
            'avg_confidence': round(np.mean(history), 3),
            'min_confidence': round(np.min(history), 3),
            'max_confidence': round(np.max(history), 3),
            'current_confidence': round(history[-1], 3),
            'smoothed_confidence': round(self._get_smoothed_confidence(), 3),
            'samples': len(history)
        }
