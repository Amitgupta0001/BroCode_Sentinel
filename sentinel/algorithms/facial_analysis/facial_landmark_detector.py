import cv2
import dlib
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional

class FacialLandmarkDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        }
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 facial landmarks using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = []
            for landmark in results.multi_face_landmarks[0].landmark:
                h, w = frame.shape[:2]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                landmarks.append([x, y, z])
            
            return np.array(landmarks)
        except Exception as e:
            print(f"Error in landmark extraction: {e}")
            return None
    
    def calculate_facial_metrics(self, landmarks: np.ndarray) -> Dict:
        """Calculate various facial metrics from landmarks"""
        if landmarks is None:
            return {}
        
        metrics = {
            'eye_aspect_ratio': self._eye_aspect_ratio(landmarks),
            'mouth_aspect_ratio': self._mouth_aspect_ratio(landmarks),
            'eyebrow_raise_intensity': self._eyebrow_raise_intensity(landmarks),
            'lip_corner_displacement': self._lip_corner_displacement(landmarks),
            'facial_symmetry': self._calculate_symmetry(landmarks),
            'blink_detected': self._detect_blink(landmarks)
        }
        return metrics
    
    def _eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio for blink detection"""
        left_eye = landmarks[self.landmark_indices['left_eye']]
        right_eye = landmarks[self.landmark_indices['right_eye']]
        
        def ear(eye_points):
            # Vertical distances
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            # Horizontal distance
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C)
        
        left_ear = ear(left_eye)
        right_ear = ear(right_eye)
        return (left_ear + right_ear) / 2.0
    
    def _mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio"""
        mouth_points = landmarks[self.landmark_indices['mouth']]
        vertical = np.linalg.norm(mouth_points[13] - mouth_points[19])
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])
        return vertical / horizontal if horizontal != 0 else 0
    
    def _eyebrow_raise_intensity(self, landmarks: np.ndarray) -> float:
        """Calculate eyebrow raise intensity"""
        eyebrow_points = landmarks[self.landmark_indices['eyebrows']]
        eye_points = landmarks[self.landmark_indices['left_eye'][:6]]
        
        eyebrow_center = np.mean(eyebrow_points, axis=0)
        eye_center = np.mean(eye_points, axis=0)
        
        return abs(eyebrow_center[1] - eye_center[1])
    
    def _lip_corner_displacement(self, landmarks: np.ndarray) -> float:
        """Calculate lip corner displacement for smile detection"""
        lip_corners = [landmarks[61], landmarks[291]]  # Left and right lip corners
        neutral_reference = [landmarks[13], landmarks[14]]  # Nose reference points
        
        displacement = 0
        for lip, ref in zip(lip_corners, neutral_reference):
            displacement += np.linalg.norm(lip - ref)
        
        return displacement / 2.0
    
    def _calculate_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        left_side = landmarks[:234]  # Left half landmarks
        right_side = landmarks[234:]  # Right half landmarks
        
        # Flip right side for comparison
        right_flipped = np.flip(right_side, axis=0)
        
        symmetry_score = 1.0 - (np.mean(np.abs(left_side - right_flipped)) / 100.0)
        return max(0, min(1, symmetry_score))
    
    def _detect_blink(self, landmarks: np.ndarray) -> bool:
        """Detect if eyes are closed (blink)"""
        ear = self._eye_aspect_ratio(landmarks)
        return ear < 0.2  # Threshold for blink detection
