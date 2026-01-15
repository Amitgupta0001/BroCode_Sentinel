import cv2
import numpy as np
from typing import Dict, List

class EyeLandmarkDetector:
    def __init__(self):
        self.eye_landmark_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        }
    
    def extract_eye_landmarks(self, face_landmarks: np.ndarray) -> Dict:
        """Extract detailed eye landmarks from facial landmarks"""
        if face_landmarks is None:
            return {}
        
        eye_data = {}
        
        for eye_name, indices in self.eye_landmark_indices.items():
            eye_points = face_landmarks[indices]
            eye_data[eye_name] = {
                'landmarks': eye_points,
                'bounding_box': self._calculate_eye_bounding_box(eye_points),
                'pupil_center': self._estimate_pupil_center(eye_points),
                'eye_openness': self._calculate_eye_openness(eye_points)
            }
        
        return eye_data
    
    def _calculate_eye_bounding_box(self, eye_points: np.ndarray) -> Dict:
        """Calculate bounding box around eye"""
        x_coords = eye_points[:, 0]
        y_coords = eye_points[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add padding
        padding = 5
        return {
            'x': int(x_min - padding),
            'y': int(y_min - padding),
            'width': int(x_max - x_min + 2 * padding),
            'height': int(y_max - y_min + 2 * padding)
        }
    
    def _estimate_pupil_center(self, eye_points: np.ndarray) -> Dict:
        """Estimate pupil center from eye landmarks"""
        # Use inner eye points for pupil estimation
        inner_eye_points = eye_points[1:5]  # Adjust indices based on your landmark model
        
        pupil_x = np.mean(inner_eye_points[:, 0])
        pupil_y = np.mean(inner_eye_points[:, 1])
        
        return {'x': pupil_x, 'y': pupil_y}
    
    def _calculate_eye_openness(self, eye_points: np.ndarray) -> float:
        """Calculate how open the eye is"""
        # Vertical distance between upper and lower eyelid
        upper_lid = eye_points[12]  # Upper eyelid point
        lower_lid = eye_points[16]  # Lower eyelid point
        
        vertical_distance = np.linalg.norm(upper_lid - lower_lid)
        
        # Horizontal eye width for normalization
        left_corner = eye_points[0]
        right_corner = eye_points[8]
        horizontal_distance = np.linalg.norm(left_corner - right_corner)
        
        if horizontal_distance == 0:
            return 0.0
        
        openness_ratio = vertical_distance / horizontal_distance
        return float(openness_ratio)
    
    def detect_eye_blinks(self, eye_data: Dict, threshold: float = 0.2) -> Dict:
        """Detect eye blinks from eye openness data"""
        blink_data = {}
        
        for eye_name, data in eye_data.items():
            openness = data['eye_openness']
            is_blinking = openness < threshold
            
            blink_data[eye_name] = {
                'is_blinking': is_blinking,
                'openness_score': openness,
                'blink_intensity': 1.0 - (openness / threshold) if is_blinking else 0.0
            }
        
        return blink_data
    
    def calculate_eye_asymmetry(self, left_eye_data: Dict, right_eye_data: Dict) -> float:
        """Calculate asymmetry between left and right eyes"""
        if not left_eye_data or not right_eye_data:
            return 0.0
        
        left_openness = left_eye_data.get('eye_openness', 0)
        right_openness = right_eye_data.get('eye_openness', 0)
        
        # Calculate asymmetry in eye openness
        openness_asymmetry = abs(left_openness - right_openness)
        
        # Calculate asymmetry in pupil position
        left_pupil = left_eye_data.get('pupil_center', {'x': 0, 'y': 0})
        right_pupil = right_eye_data.get('pupil_center', {'x': 0, 'y': 0})
        
        pupil_x_asymmetry = abs(left_pupil['x'] - right_pupil['x'])
        pupil_y_asymmetry = abs(left_pupil['y'] - right_pupil['y'])
        
        # Combine asymmetries
        total_asymmetry = (openness_asymmetry + pupil_x_asymmetry + pupil_y_asymmetry) / 3.0
        
        return float(total_asymmetry)
