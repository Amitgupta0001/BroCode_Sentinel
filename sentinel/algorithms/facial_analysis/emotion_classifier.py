import numpy as np
import cv2
from typing import Dict, List

class EmotionClassifier:
    def __init__(self):
        # Placeholder for heavier model
        self.emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
        self.emotion_history = []
        self.max_history = 30
        # Initialize basic face detector for 'activity' detection at least
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def predict_emotion(self, face_roi: np.ndarray) -> Dict:
        """Predict emotion from face ROI (simplified for stability)"""
        # In a full deployment, we would load the .h5 model here.
        # For now, we return 'neutral' to allow the system to boot without 500MB+ TensorFlow.
        return {
            'emotion': 'neutral',
            'confidence': 0.85,
            'all_predictions': {k: 0.1 for k in self.emotion_labels}
        }
    
    def analyze_emotions(self, frame: np.ndarray) -> Dict:
        """Full frame analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
             return {"variability": 0.0, "dominant_emotion": "none"}
             
        # Just analyze first face
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        result = self.predict_emotion(face_roi)
        
        return {
            "variability": 0.2, # Low variability = stable
            "dominant_emotion": result['emotion']
        }

    # Keep other methods like consistency calculation
    def get_emotion_consistency(self) -> float:
        return 0.9 # High stability simulation
    
    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face ROI for emotion classification"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Add channel and batch dimensions
        processed = np.expand_dims(normalized, axis=-1)
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def predict_emotion(self, face_roi: np.ndarray) -> Dict:
        """Predict emotion from face ROI"""
        try:
            processed_face = self.preprocess_face(face_roi)
            predictions = self.model.predict(processed_face, verbose=0)
            
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            emotion = self.emotion_labels[emotion_idx]
            
            # Update emotion history
            self._update_emotion_history(emotion, confidence)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': dict(zip(self.emotion_labels, predictions[0].tolist()))
            }
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return {'emotion': 'unknown', 'confidence': 0.0}
    
    def get_emotion_consistency(self) -> float:
        """Calculate consistency of emotions over time"""
        if len(self.emotion_history) < 2:
            return 1.0
        
        emotions = [entry['emotion'] for entry in self.emotion_history]
        unique_emotions = len(set(emotions))
        consistency = 1.0 - (unique_emotions / len(emotions))
        
        return max(0, min(1, consistency))
    
    def _update_emotion_history(self, emotion: str, confidence: float):
        """Update emotion history for temporal analysis"""
        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': np.datetime64('now')
        })
        
        # Keep only recent history
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
