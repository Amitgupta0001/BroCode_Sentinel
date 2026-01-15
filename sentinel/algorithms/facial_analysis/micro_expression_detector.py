import numpy as np
import cv2
from typing import Dict, List
from collections import deque

class MicroExpressionDetector:
    def __init__(self, frame_rate: int = 30):
        self.frame_rate = frame_rate
        self.expression_buffer = deque(maxlen=frame_rate * 2)  # 2-second buffer
        self.micro_expression_threshold = 0.3
        self.expression_duration_threshold = 0.5  # seconds
        
    def detect_micro_expressions(self, current_emotion: Dict, landmarks: np.ndarray) -> List[Dict]:
        """Detect micro-expressions in the current frame"""
        micro_expressions = []
        
        # Add current expression to buffer
        current_data = {
            'emotion': current_emotion['emotion'],
            'confidence': current_emotion['confidence'],
            'landmarks': landmarks,
            'timestamp': np.datetime64('now')
        }
        self.expression_buffer.append(current_data)
        
        # Need at least 2 frames for comparison
        if len(self.expression_buffer) < 2:
            return micro_expressions
        
        # Analyze recent frames for micro-expressions
        for i in range(1, len(self.expression_buffer)):
            prev_data = self.expression_buffer[i-1]
            curr_data = self.expression_buffer[i]
            
            # Check for rapid emotion change
            emotion_change = self._detect_emotion_change(prev_data, curr_data)
            
            # Check for brief facial movement
            movement_brief = self._detect_brief_movement(prev_data, curr_data)
            
            if emotion_change and movement_brief:
                micro_expression = {
                    'type': 'micro_expression',
                    'emotion_transition': f"{prev_data['emotion']}->{curr_data['emotion']}",
                    'duration_frames': i,
                    'intensity': self._calculate_intensity(prev_data, curr_data),
                    'timestamp': curr_data['timestamp']
                }
                micro_expressions.append(micro_expression)
        
        return micro_expressions
    
    def _detect_emotion_change(self, prev_data: Dict, curr_data: Dict) -> bool:
        """Detect significant emotion change between frames"""
        if prev_data['emotion'] == curr_data['emotion']:
            return False
        
        # Emotion changed, check if confidence is reasonable
        confidence_threshold = 0.6
        return (prev_data['confidence'] > confidence_threshold and 
                curr_data['confidence'] > confidence_threshold)
    
    def _detect_brief_movement(self, prev_data: Dict, curr_data: Dict) -> bool:
        """Detect if facial movement is brief enough to be micro-expression"""
        if prev_data['landmarks'] is None or curr_data['landmarks'] is None:
            return False
        
        # Calculate facial movement magnitude
        movement = np.mean(np.abs(prev_data['landmarks'] - curr_data['landmarks']))
        
        # Should be significant but brief movement
        return (movement > 0.1 and  # Minimum movement threshold
                movement < 2.0)     # Maximum movement threshold
    
    def _calculate_intensity(self, prev_data: Dict, curr_data: Dict) -> float:
        """Calculate intensity of micro-expression"""
        if prev_data['landmarks'] is None or curr_data['landmarks'] is None:
            return 0.0
        
        # Intensity based on landmark displacement and emotion confidence
        landmark_displacement = np.mean(np.abs(prev_data['landmarks'] - curr_data['landmarks']))
        confidence_combined = (prev_data['confidence'] + curr_data['confidence']) / 2
        
        intensity = landmark_displacement * confidence_combined
        return float(intensity)
    
    def get_micro_expression_frequency(self, window_seconds: int = 10) -> float:
        """Calculate frequency of micro-expressions in given time window"""
        if not self.expression_buffer:
            return 0.0
        
        current_time = np.datetime64('now')
        window_start = current_time - np.timedelta64(window_seconds, 's')
        
        # Count micro-expressions in time window (simplified)
        recent_expressions = [expr for expr in self.expression_buffer 
                            if expr['timestamp'] > window_start]
        
        if len(recent_expressions) < 2:
            return 0.0
        
        # Estimate frequency based on emotion changes
        emotion_changes = 0
        for i in range(1, len(recent_expressions)):
            if recent_expressions[i-1]['emotion'] != recent_expressions[i]['emotion']:
                emotion_changes += 1
        
        frequency = emotion_changes / window_seconds
        return frequency
