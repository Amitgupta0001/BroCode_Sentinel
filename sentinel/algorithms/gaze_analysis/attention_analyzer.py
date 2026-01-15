import numpy as np
from typing import Dict, List
from collections import deque

class AttentionAnalyzer:
    def __init__(self, window_size: int = 90):  # 3 seconds at 30fps
        self.window_size = window_size
        self.attention_history = deque(maxlen=window_size)
        self.gaze_history = deque(maxlen=window_size)
        self.blink_history = deque(maxlen=window_size)
        # Load standard face detector
        try:
            import cv2
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.face_cascade = None

    def analyze_gaze_patterns(self, frame: np.ndarray) -> Dict:
        """
        Public interface called by main_authentication_system.py.
        Uses REAL OpenCV detection to verify user presence.
        """
        if self.face_cascade is None:
             # Fallback if cv2 fails for some reason
             return {"attention_score": 0.5, "stability": 0.5}

        # Convert to grayscale for detection
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Logic: If face is present -> Accredited. If vacant -> Penalized.
            if len(faces) > 0:
                # Face found: High trust
                return {
                    "attention_score": 0.9,
                    "stability": 0.9
                }
            else:
                # No face found: Very low trust (will trigger logout)
                return {
                    "attention_score": 0.1,
                    "stability": 0.1
                }
        except Exception as e:
            print(f"Gaze analysis error: {e}")
            return {"attention_score": 0.5, "stability": 0.5}

    def calculate_attention_score(self, gaze_data: Dict, eye_data: Dict, 
                                head_pose: Dict) -> Dict:
        """Calculate comprehensive attention score"""
        attention_metrics = {}
        
        # Gaze-based attention
        gaze_attention = self._calculate_gaze_attention(gaze_data)
        
        # Head pose-based attention
        head_attention = self._calculate_head_pose_attention(head_pose)
        
        # Eye-based attention (blink rate, pupil movement)
        eye_attention = self._calculate_eye_attention(eye_data)
        
        # Combine attention scores
        combined_attention = (gaze_attention * 0.5 + 
                            head_attention * 0.3 + 
                            eye_attention * 0.2)
        
        # Update history
        self._update_attention_history(combined_attention, gaze_data, eye_data)
        
        attention_metrics.update({
            'current_attention': combined_attention,
            'gaze_attention': gaze_attention,
            'head_attention': head_attention,
            'eye_attention': eye_attention,
            'attention_trend': self._calculate_attention_trend(),
            'attention_consistency': self._calculate_attention_consistency(),
            'distraction_level': self._calculate_distraction_level()
        })
        
        return attention_metrics
    
    def _calculate_gaze_attention(self, gaze_data: Dict) -> float:
        """Calculate attention based on gaze patterns"""
        if not gaze_data:
            return 0.5
        
        gaze_stability = gaze_data.get('attention_level', 0.5)
        gaze_consistency = gaze_data.get('gaze_consistency', 0.5) if 'gaze_consistency' in gaze_data else 0.5
        
        # Gaze directed forward indicates attention
        gaze_vector = gaze_data.get('gaze_vector', (0, 0))
        gaze_x, gaze_y = gaze_vector
        gaze_direction_score = 1.0 - min(abs(gaze_x) + abs(gaze_y), 1.0)
        
        attention_score = (gaze_stability * 0.4 + 
                         gaze_consistency * 0.3 + 
                         gaze_direction_score * 0.3)
        
        return max(0, min(1, attention_score))
    
    def _calculate_head_pose_attention(self, head_pose: Dict) -> float:
        """Calculate attention based on head pose"""
        if not head_pose:
            return 0.5
        
        pitch = abs(head_pose.get('pitch', 0))
        yaw = abs(head_pose.get('yaw', 0))
        
        # Head facing forward indicates attention
        head_stability = 1.0 - min((pitch + yaw) / 90.0, 1.0)
        
        return max(0, min(1, head_stability))
    
    def _calculate_eye_attention(self, eye_data: Dict) -> float:
        """Calculate attention based on eye metrics"""
        if not eye_data:
            return 0.5
        
        blink_rate = self._calculate_blink_rate(eye_data)
        pupil_stability = self._calculate_pupil_stability(eye_data)
        
        # Normal blink rate is around 15-20 blinks per minute
        ideal_blink_rate = 0.3  # blinks per second
        blink_score = 1.0 - min(abs(blink_rate - ideal_blink_rate) / ideal_blink_rate, 1.0)
        
        attention_score = (blink_score * 0.6 + pupil_stability * 0.4)
        return max(0, min(1, attention_score))
    
    def _calculate_blink_rate(self, eye_data: Dict) -> float:
        """Calculate current blink rate"""
        # This would be implemented based on blink detection history
        return 0.25  # Placeholder
    
    def _calculate_pupil_stability(self, eye_data: Dict) -> float:
        """Calculate how stable pupils are"""
        # This would analyze pupil movement over time
        return 0.8  # Placeholder
    
    def _calculate_attention_trend(self) -> str:
        """Calculate trend of attention over time"""
        if len(self.attention_history) < 5:
            return "stable"
        
        recent_attention = list(self.attention_history)[-5:]
        trend = np.polyfit(range(len(recent_attention)), recent_attention, 1)[0]
        
        if trend > 0.01:
            return "improving"
        elif trend < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _calculate_attention_consistency(self) -> float:
        """Calculate how consistent attention is over time"""
        if len(self.attention_history) < 2:
            return 1.0
        
        attention_values = list(self.attention_history)
        variance = np.var(attention_values)
        consistency = 1.0 / (1.0 + variance * 10)
        
        return float(consistency)
    
    def _calculate_distraction_level(self) -> float:
        """Calculate level of distraction based on various factors"""
        if len(self.attention_history) < 10:
            return 0.0
        
        recent_attention = list(self.attention_history)[-10:]
        avg_attention = np.mean(recent_attention)
        
        # Low attention indicates distraction
        distraction_level = 1.0 - avg_attention
        
        # Increase distraction level if there are frequent gaze shifts
        if len(self.gaze_history) > 5:
            gaze_shifts = sum(1 for g in self.gaze_history if g.get('gaze_consistency', 0.5) < 0.3)
            gaze_distraction = gaze_shifts / len(self.gaze_history)
            distraction_level = max(distraction_level, gaze_distraction)
        
        return float(distraction_level)
    
    def _update_attention_history(self, attention_score: float, gaze_data: Dict, eye_data: Dict):
        """Update attention analysis history"""
        self.attention_history.append(attention_score)
        self.gaze_history.append(gaze_data)
        self.blink_history.append(eye_data)
    
    def get_attention_summary(self, duration: int = 30) -> Dict:
        """Get summary of attention over specified duration"""
        if len(self.attention_history) < duration:
            available_data = len(self.attention_history)
        else:
            available_data = duration
        
        if available_data == 0:
            return {}
        
        recent_attention = list(self.attention_history)[-available_data:]
        
        summary = {
            'average_attention': float(np.mean(recent_attention)),
            'attention_std': float(np.std(recent_attention)),
            'min_attention': float(np.min(recent_attention)),
            'max_attention': float(np.max(recent_attention)),
            'attention_trend': self._calculate_attention_trend(),
            'distraction_episodes': self._count_distraction_episodes(recent_attention)
        }
        
        return summary
    
    def _count_distraction_episodes(self, attention_values: List[float]) -> int:
        """Count episodes where attention dropped significantly"""
        if len(attention_values) < 3:
            return 0
        
        distraction_threshold = 0.3
        distraction_episodes = 0
        in_distraction = False
        
        for attention in attention_values:
            if attention < distraction_threshold and not in_distraction:
                distraction_episodes += 1
                in_distraction = True
            elif attention >= distraction_threshold:
                in_distraction = False
        
        return distraction_episodes
