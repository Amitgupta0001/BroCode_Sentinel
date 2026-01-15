import numpy as np
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter
from dataclasses import dataclass

@dataclass
class MovementMetrics:
    activity_level: float
    movement_smoothness: float
    gesture_frequency: float
    fidgeting_score: float
    movement_pattern: str

class MovementAnalyzer:
    def __init__(self, window_size: int = 30, fps: int = 30):
        self.window_size = window_size
        self.fps = fps
        self.pose_history = []
        self.gesture_history = []
        
    def analyze_movement_patterns(self, pose_sequence: List[np.ndarray], 
                                gesture_sequence: List[str]) -> MovementMetrics:
        """Analyze movement patterns from pose and gesture sequences"""
        if len(pose_sequence) < 2:
            return self._get_default_metrics()
        
        # Calculate movement metrics
        activity_level = self._calculate_activity_level(pose_sequence)
        movement_smoothness = self._calculate_movement_smoothness(pose_sequence)
        gesture_frequency = self._calculate_gesture_frequency(gesture_sequence)
        fidgeting_score = self._calculate_fidgeting_score(pose_sequence)
        movement_pattern = self._classify_movement_pattern(
            activity_level, movement_smoothness, fidgeting_score
        )
        
        return MovementMetrics(
            activity_level=activity_level,
            movement_smoothness=movement_smoothness,
            gesture_frequency=gesture_frequency,
            fidgeting_score=fidgeting_score,
            movement_pattern=movement_pattern
        )
    
    def _calculate_activity_level(self, pose_sequence: List[np.ndarray]) -> float:
        """Calculate overall activity level (0-1)"""
        if len(pose_sequence) < 2:
            return 0.0
        
        total_movement = 0.0
        key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Major joints
        
        for i in range(1, len(pose_sequence)):
            for joint in key_joints:
                if (joint < len(pose_sequence[i]) and 
                    joint < len(pose_sequence[i-1])):
                    movement = np.linalg.norm(
                        pose_sequence[i][joint] - pose_sequence[i-1][joint]
                    )
                    total_movement += movement
        
        # Normalize by time and number of joints
        avg_movement = total_movement / (len(pose_sequence) - 1) / len(key_joints)
        
        # Convert to 0-1 scale (empirical threshold)
        activity_level = np.clip(avg_movement * 100, 0, 1)
        return float(activity_level)
    
    def _calculate_movement_smoothness(self, pose_sequence: List[np.ndarray]) -> float:
        """Calculate movement smoothness using jerk analysis"""
        if len(pose_sequence) < 3:
            return 1.0  # Default to smooth for short sequences
        
        jerks = []
        key_joints = [11, 12]  # Shoulders for simplicity
        
        for joint in key_joints:
            # Extract joint positions over time
            positions = np.array([pose[joint] for pose in pose_sequence if joint < len(pose)])
            
            if len(positions) < 3:
                continue
                
            # Calculate velocity (first derivative)
            velocity = np.diff(positions, axis=0) * self.fps
            
            # Calculate acceleration (second derivative)
            acceleration = np.diff(velocity, axis=0) * self.fps
            
            # Calculate jerk (third derivative)
            jerk = np.diff(acceleration, axis=0) * self.fps
            
            if len(jerk) > 0:
                avg_jerk = np.mean(np.linalg.norm(jerk, axis=1))
                jerks.append(avg_jerk)
        
        if not jerks:
            return 1.0
        
        # Convert jerk to smoothness score (lower jerk = smoother movement)
        avg_jerk = np.mean(jerks)
        smoothness = 1.0 / (1.0 + avg_jerk)  # Logistic transform
        return float(np.clip(smoothness, 0, 1))
    
    def _calculate_gesture_frequency(self, gesture_sequence: List[str]) -> float:
        """Calculate gestures per minute"""
        if not gesture_sequence:
            return 0.0
        
        # Count meaningful gestures (exclude 'none')
        meaningful_gestures = [g for g in gesture_sequence if g != 'none']
        
        duration_minutes = len(gesture_sequence) / (self.fps * 60.0)
        frequency = len(meaningful_gestures) / duration_minutes if duration_minutes > 0 else 0.0
        
        return float(frequency)
    
    def _calculate_fidgeting_score(self, pose_sequence: List[np.ndarray]) -> float:
        """Calculate fidgeting score based on small, repetitive movements"""
        if len(pose_sequence) < 10:
            return 0.0
        
        fidgeting_joints = [15, 16, 27, 28]  # Wrists and ankles
        fidget_movements = 0
        total_movements = 0
        
        for joint in fidgeting_joints:
            # Extract joint positions
            positions = np.array([pose[joint] for pose in pose_sequence if joint < len(pose)])
            
            if len(positions) < 10:
                continue
                
            # Calculate movement directions
            movements = np.diff(positions, axis=0)
            movement_magnitudes = np.linalg.norm(movements, axis=1)
            
            # Small, rapid movements indicate fidgeting
            small_movements = movement_magnitudes < 0.05  # Threshold for small movements
            
            # Look for rapid direction changes (high frequency)
            if len(movements) > 2:
                directions = movements / (movement_magnitudes[:, np.newaxis] + 1e-8)
                direction_changes = np.arccos(np.clip(
                    np.sum(directions[1:] * directions[:-1], axis=1), -1, 1
                ))
                
                rapid_changes = direction_changes > np.radians(90)  # Large direction changes
                fidget_movements += np.sum(small_movements[1:] & rapid_changes)
                total_movements += len(small_movements[1:])
        
        if total_movements == 0:
            return 0.0
        
        fidgeting_score = fidget_movements / total_movements
        return float(fidgeting_score)
    
    def _classify_movement_pattern(self, activity_level: float, 
                                 smoothness: float, fidgeting: float) -> str:
        """Classify overall movement pattern"""
        if activity_level < 0.1:
            return "still"
        elif fidgeting > 0.3:
            return "fidgeting"
        elif smoothness < 0.5:
            return "jerky"
        elif activity_level > 0.7:
            return "active"
        elif activity_level > 0.3:
            return "moderate"
        else:
            return "calm"
    
    def _get_default_metrics(self) -> MovementMetrics:
        """Return default metrics when insufficient data"""
        return MovementMetrics(
            activity_level=0.0,
            movement_smoothness=1.0,
            gesture_frequency=0.0,
            fidgeting_score=0.0,
            movement_pattern="still"
        )
