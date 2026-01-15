import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class GestureRecognizer:
    def __init__(self, model_path: str = None):
        self.gesture_categories = [
            "pointing", "waving", "thumbs_up", "thumbs_down", 
            "ok_sign", "victory", "crossed_arms", "hand_on_face",
            "touching_hair", "rubbing_hands", "none"
        ]
        
        if model_path:
            self.load_model(model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
    
    def extract_hand_features(self, hand_landmarks: np.ndarray) -> np.ndarray:
        """Extract features from hand landmarks for gesture recognition"""
        features = []
        
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return np.zeros(50)  # Return zero features if no hand detected
        
        # Finger distances and angles
        features.extend(self._extract_finger_features(hand_landmarks))
        
        # Palm orientation and size
        features.extend(self._extract_palm_features(hand_landmarks))
        
        # Gesture-specific features
        features.extend(self._extract_gesture_features(hand_landmarks))
        
        return np.array(features)
    
    def _extract_finger_features(self, landmarks: np.ndarray) -> List[float]:
        """Extract finger-related features"""
        features = []
        
        # Finger tip distances from wrist
        wrist = landmarks[0]
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
        
        for tip_idx in finger_tips:
            distance = np.linalg.norm(landmarks[tip_idx] - wrist)
            features.append(distance)
        
        # Finger bending angles
        finger_joints = [
            (2, 3, 4),   # Thumb
            (5, 6, 8),   # Index
            (9, 10, 12), # Middle
            (13, 14, 16), # Ring
            (17, 18, 20)  # Pinky
        ]
        
        for joint1, joint2, joint3 in finger_joints:
            angle = self._calculate_angle(landmarks[joint1], landmarks[joint2], landmarks[joint3])
            features.append(angle)
        
        return features
    
    def _extract_palm_features(self, landmarks: np.ndarray) -> List[float]:
        """Extract palm-related features"""
        features = []
        
        # Palm center and orientation
        palm_points = [0, 1, 5, 9, 13, 17]  # Wrist and palm base points
        palm_center = np.mean(landmarks[palm_points], axis=0)
        
        # Palm size (approximate)
        palm_size = np.linalg.norm(landmarks[0] - landmarks[9])  # Wrist to middle finger base
        features.append(palm_size)
        
        # Palm orientation vectors
        wrist_to_index = landmarks[5] - landmarks[0]  # Wrist to index base
        wrist_to_pinky = landmarks[17] - landmarks[0]  # Wrist to pinky base
        
        # Normalize vectors
        if np.linalg.norm(wrist_to_index) > 0:
            wrist_to_index = wrist_to_index / np.linalg.norm(wrist_to_index)
        if np.linalg.norm(wrist_to_pinky) > 0:
            wrist_to_pinky = wrist_to_pinky / np.linalg.norm(wrist_to_pinky)
        
        features.extend(wrist_to_index)
        features.extend(wrist_to_pinky)
        
        return features
    
    def _extract_gesture_features(self, landmarks: np.ndarray) -> List[float]:
        """Extract gesture-specific features"""
        features = []
        
        # Thumb-index pinch
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_distance = np.linalg.norm(thumb_tip - index_tip)
        features.append(pinch_distance)
        
        # Finger spreading
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
        tip_distances = []
        for i in range(len(finger_tips)):
            for j in range(i+1, len(finger_tips)):
                distance = np.linalg.norm(landmarks[finger_tips[i]] - landmarks[finger_tips[j]])
                tip_distances.append(distance)
        
        features.extend(tip_distances)
        
        # Hand openness (average distance from palm center to finger tips)
        palm_center = np.mean(landmarks[[0, 1, 5, 9, 13, 17]], axis=0)
        openness = np.mean([np.linalg.norm(landmarks[i] - palm_center) for i in [4, 8, 12, 16, 20]])
        features.append(openness)
        
        return features
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle between three points"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def recognize_gesture(self, hand_landmarks: np.ndarray, confidence_threshold: float = 0.6) -> Tuple[str, float]:
        """Recognize gesture from hand landmarks"""
        if not self.is_trained:
            return "none", 0.0
        
        features = self.extract_hand_features(hand_landmarks).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        probabilities = self.model.predict_proba(features_scaled)[0]
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        
        if max_prob >= confidence_threshold:
            gesture = self.gesture_categories[max_prob_idx]
            return gesture, float(max_prob)
        else:
            return "none", float(max_prob)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the gesture recognition model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def save_model(self, path: str):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path: str):
        """Load trained model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
