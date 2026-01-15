import cv2
import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestRegressor

class GazeTracker:
    def __init__(self):
        self.eye_model = self._initialize_eye_model()
        self.gaze_history = []
        self.max_history = 60  # 2 seconds at 30fps
        self.attention_threshold = 0.7
        
    def _initialize_eye_model(self) -> RandomForestRegressor:
        """Initialize gaze estimation model"""
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
    
    def preprocess_eye_region(self, eye_region: np.ndarray) -> np.ndarray:
        """Preprocess eye region for gaze estimation"""
        # Convert to grayscale
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        resized = cv2.resize(gray, (64, 32))
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(resized)
        
        # Normalize
        normalized = equalized.astype('float32') / 255.0
        
        # Flatten for model input
        flattened = normalized.flatten()
        
        return flattened
    
    def estimate_gaze_direction(self, left_eye: np.ndarray, right_eye: np.ndarray, 
                              head_pose: Dict) -> Dict:
        """Estimate gaze direction from eye regions and head pose"""
        try:
            # Preprocess both eyes
            left_eye_features = self.preprocess_eye_region(left_eye)
            right_eye_features = self.preprocess_eye_region(right_eye)
            
            # Combine features
            eye_features = np.concatenate([left_eye_features, right_eye_features])
            
            # Add head pose information if available
            if head_pose:
                head_features = [head_pose.get('pitch', 0), 
                               head_pose.get('yaw', 0), 
                               head_pose.get('roll', 0)]
                all_features = np.concatenate([eye_features, head_features])
            else:
                all_features = eye_features
            
            # Predict gaze (simplified - in practice, this would use a trained model)
            gaze_x, gaze_y = self._simulate_gaze_prediction(all_features)
            
            gaze_data = {
                'gaze_vector': (float(gaze_x), float(gaze_y)),
                'gaze_point': self._vector_to_screen_coordinates(gaze_x, gaze_y),
                'confidence': self._calculate_gaze_confidence(left_eye, right_eye),
                'attention_level': self._calculate_attention_level(gaze_x, gaze_y)
            }
            
            # Update gaze history
            self._update_gaze_history(gaze_data)
            
            return gaze_data
            
        except Exception as e:
            print(f"Error in gaze estimation: {e}")
            return {'gaze_vector': (0, 0), 'confidence': 0.0, 'attention_level': 0.0}
    
    def _simulate_gaze_prediction(self, features: np.ndarray) -> Tuple[float, float]:
        """Simulate gaze prediction (replace with actual trained model)"""
        # This is a simplified simulation
        # In practice, you would use a trained regression model
        gaze_x = np.sin(features[0] * 2 * np.pi) * 0.5
        gaze_y = np.cos(features[100] * 2 * np.pi) * 0.3
        return gaze_x, gaze_y
    
    def _vector_to_screen_coordinates(self, gaze_x: float, gaze_y: float) -> Dict:
        """Convert gaze vector to screen coordinates"""
        # Convert normalized gaze vector to screen coordinates
        screen_x = (gaze_x + 1) * 400  # Assuming 800px width
        screen_y = (gaze_y + 1) * 300  # Assuming 600px height
        
        return {'x': screen_x, 'y': screen_y}
    
    def _calculate_gaze_confidence(self, left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """Calculate confidence in gaze estimation"""
        # Simple confidence based on eye region quality
        left_quality = self._assess_eye_quality(left_eye)
        right_quality = self._assess_eye_quality(right_eye)
        
        return (left_quality + right_quality) / 2.0
    
    def _assess_eye_quality(self, eye_region: np.ndarray) -> float:
        """Assess quality of eye region for gaze estimation"""
        if eye_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(gray)
        
        # Calculate brightness (mean pixel value)
        brightness = np.mean(gray)
        
        # Ideal brightness is around 127
        brightness_score = 1.0 - abs(brightness - 127) / 127
        
        # Combine scores
        quality = (contrast / 100 + brightness_score) / 2
        return max(0, min(1, quality))
    
    def _calculate_attention_level(self, gaze_x: float, gaze_y: float) -> float:
        """Calculate attention level based on gaze stability"""
        if not self.gaze_history:
            return 1.0
        
        # Calculate gaze stability (how much gaze has moved recently)
        recent_gaze = self.gaze_history[-5:]  # Last 5 frames
        if len(recent_gaze) < 2:
            return 1.0
        
        gaze_changes = []
        for i in range(1, len(recent_gaze)):
            prev_gaze = recent_gaze[i-1]['gaze_vector']
            curr_gaze = recent_gaze[i]['gaze_vector']
            
            change = np.linalg.norm(np.array(curr_gaze) - np.array(prev_gaze))
            gaze_changes.append(change)
        
        avg_change = np.mean(gaze_changes) if gaze_changes else 0
        stability = 1.0 / (1.0 + avg_change * 10)  # Convert to stability score
        
        return float(stability)
    
    def analyze_gaze_patterns(self, duration: float = 5.0) -> Dict:
        """Analyze gaze patterns over time for authenticity detection"""
        if len(self.gaze_history) < 2:
            return {}
        
        # Get recent gaze data within duration
        recent_gaze = [g for g in self.gaze_history 
                      if len(self.gaze_history) - self.gaze_history.index(g) < duration * 30]
        
        if len(recent_gaze) < 2:
            return {}
        
        gaze_vectors = [g['gaze_vector'] for g in recent_gaze]
        
        patterns = {
            'gaze_consistency': self._calculate_gaze_consistency(gaze_vectors),
            'saccade_frequency': self._count_saccades(gaze_vectors),
            'fixation_duration': self._calculate_fixation_duration(gaze_vectors),
            'gaze_aversion': self._detect_gaze_aversion(gaze_vectors),
            'visual_scanning_pattern': self._analyze_scanning_pattern(gaze_vectors)
        }
        
        return patterns
    
    def _calculate_gaze_consistency(self, gaze_vectors: List[Tuple]) -> float:
        """Calculate how consistent gaze direction is"""
        if len(gaze_vectors) < 2:
            return 1.0
        
        variances = []
        for i in range(len(gaze_vectors)-1):
            diff = np.linalg.norm(np.array(gaze_vectors[i+1]) - np.array(gaze_vectors[i]))
            variances.append(diff)
        
        avg_variance = np.mean(variances)
        consistency = 1.0 / (1.0 + avg_variance)
        return float(consistency)
    
    def _count_saccades(self, gaze_vectors: List[Tuple]) -> int:
        """Count rapid eye movements (saccades)"""
        if len(gaze_vectors) < 3:
            return 0
        
        saccade_count = 0
        saccade_threshold = 0.1  # Minimum movement for saccade
        
        for i in range(1, len(gaze_vectors)-1):
            movement = np.linalg.norm(np.array(gaze_vectors[i]) - np.array(gaze_vectors[i-1]))
            if movement > saccade_threshold:
                saccade_count += 1
        
        return saccade_count
    
    def _calculate_fixation_duration(self, gaze_vectors: List[Tuple]) -> float:
        """Calculate average fixation duration"""
        if len(gaze_vectors) < 2:
            return 0.0
        
        fixation_threshold = 0.05
        fixation_durations = []
        current_fixation = 1
        
        for i in range(1, len(gaze_vectors)):
            movement = np.linalg.norm(np.array(gaze_vectors[i]) - np.array(gaze_vectors[i-1]))
            if movement < fixation_threshold:
                current_fixation += 1
            else:
                if current_fixation > 1:
                    fixation_durations.append(current_fixation)
                current_fixation = 1
        
        # Add last fixation
        if current_fixation > 1:
            fixation_durations.append(current_fixation)
        
        return np.mean(fixation_durations) / 30.0 if fixation_durations else 0.0  # Convert to seconds
    
    def _detect_gaze_aversion(self, gaze_vectors: List[Tuple]) -> float:
        """Detect gaze avoidance behaviors"""
        if not gaze_vectors:
            return 0.0
        
        aversion_count = 0
        for gaze in gaze_vectors:
            gaze_x, gaze_y = gaze
            # Looking far left/right or down often indicates aversion
            if abs(gaze_x) > 0.7 or gaze_y < -0.5:
                aversion_count += 1
        
        return aversion_count / len(gaze_vectors)
    
    def _analyze_scanning_pattern(self, gaze_vectors: List[Tuple]) -> str:
        """Analyze visual scanning pattern"""
        if len(gaze_vectors) < 5:
            return "insufficient_data"
        
        # Calculate movement patterns
        movements = []
        for i in range(1, len(gaze_vectors)):
            prev_x, prev_y = gaze_vectors[i-1]
            curr_x, curr_y = gaze_vectors[i]
            movements.append((curr_x - prev_x, curr_y - prev_y))
        
        # Analyze pattern
        horizontal_movements = sum(1 for dx, dy in movements if abs(dx) > abs(dy))
        vertical_movements = sum(1 for dx, dy in movements if abs(dy) > abs(dx))
        
        total_movements = len(movements)
        if total_movements == 0:
            return "stationary"
        
        horizontal_ratio = horizontal_movements / total_movements
        vertical_ratio = vertical_movements / total_movements
        
        if horizontal_ratio > 0.6:
            return "horizontal_scanning"
        elif vertical_ratio > 0.6:
            return "vertical_scanning"
        else:
            return "mixed_scanning"
    
    def _update_gaze_history(self, gaze_data: Dict):
        """Update gaze history for temporal analysis"""
        self.gaze_history.append(gaze_data)
        
        # Keep only recent history
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
