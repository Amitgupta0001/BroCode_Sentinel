import cv2
import numpy as np
from typing import Dict, Tuple

class HeadPoseEstimator:
    def __init__(self):
        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera intrinsic parameters (approximate)
        self.camera_matrix = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    
    def estimate_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Dict:
        """Estimate head pose from facial landmarks"""
        if landmarks is None:
            return {}
        
        # Update camera matrix based on frame size
        h, w = frame_shape[:2]
        self.camera_matrix[0, 2] = w / 2
        self.camera_matrix[1, 2] = h / 2
        
        # 2D image points from landmarks (using MediaPipe indices)
        image_points = np.array([
            landmarks[1],    # Nose tip
            landmarks[152],  # Chin
            landmarks[33],   # Left eye left corner
            landmarks[263],  # Right eye right corner
            landmarks[61],   # Left mouth corner
            landmarks[291]   # Right mouth corner
        ], dtype=np.float64)
        
        try:
            # Solve PnP problem
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points, image_points, 
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                
                # Calculate Euler angles
                euler_angles = self._rotation_matrix_to_euler_angles(rotation_mat)
                
                # Calculate head pose metrics
                pose_metrics = {
                    'pitch': float(euler_angles[0]),   # Nodding
                    'yaw': float(euler_angles[1]),     # Shaking
                    'roll': float(euler_angles[2]),    # Tilting
                    'translation': translation_vec.flatten().tolist(),
                    'rotation_matrix': rotation_mat.tolist(),
                    'gaze_direction': self._estimate_gaze_direction(euler_angles)
                }
                
                return pose_metrics
            else:
                return {}
                
        except Exception as e:
            print(f"Error in head pose estimation: {e}")
            return {}
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        euler_angles = np.array([x, y, z]) * 180.0 / np.pi
        return euler_angles
    
    def _estimate_gaze_direction(self, euler_angles: np.ndarray) -> str:
        """Estimate general gaze direction from head pose"""
        pitch, yaw, roll = euler_angles
        
        if abs(yaw) > 20:
            return "left" if yaw > 0 else "right"
        elif abs(pitch) > 15:
            return "down" if pitch > 0 else "up"
        else:
            return "forward"
    
    def calculate_head_movement_metrics(self, pose_history: List[Dict]) -> Dict:
        """Calculate head movement patterns over time"""
        if len(pose_history) < 2:
            return {}
        
        recent_poses = pose_history[-10:]  # Last 10 frames
        
        pitches = [pose['pitch'] for pose in recent_poses if 'pitch' in pose]
        yaws = [pose['yaw'] for pose in recent_poses if 'yaw' in pose]
        rolls = [pose['roll'] for pose in recent_poses if 'roll' in pose]
        
        if not pitches or not yaws or not rolls:
            return {}
        
        metrics = {
            'head_movement_frequency': self._calculate_movement_frequency(recent_poses),
            'head_stability': self._calculate_head_stability(pitches, yaws, rolls),
            'nodding_detected': self._detect_nodding(pitches),
            'shaking_detected': self._detect_shaking(yaws)
        }
        
        return metrics
    
    def _calculate_movement_frequency(self, poses: List[Dict]) -> float:
        """Calculate frequency of head movements"""
        if len(poses) < 3:
            return 0.0
        
        movement_changes = 0
        for i in range(1, len(poses)-1):
            # Check if direction changed significantly
            pitch_change = abs(poses[i]['pitch'] - poses[i-1]['pitch']) > 2.0
            yaw_change = abs(poses[i]['yaw'] - poses[i-1]['yaw']) > 2.0
            
            if pitch_change or yaw_change:
                movement_changes += 1
        
        return movement_changes / len(poses)
    
    def _calculate_head_stability(self, pitches: List[float], yaws: List[float], rolls: List[float]) -> float:
        """Calculate how stable the head position is"""
        pitch_var = np.var(pitches)
        yaw_var = np.var(yaws)
        roll_var = np.var(rolls)
        
        total_variance = (pitch_var + yaw_var + roll_var) / 3.0
        stability = 1.0 / (1.0 + total_variance)  # Inverse relationship
        
        return float(stability)
    
    def _detect_nodding(self, pitches: List[float]) -> bool:
        """Detect nodding motion"""
        if len(pitches) < 5:
            return False
        
        # Simple nodding detection based on pitch oscillations
        pitch_changes = np.diff(pitches)
        zero_crossings = np.where(np.diff(np.sign(pitch_changes)))[0]
        
        return len(zero_crossings) >= 2  # At least one complete cycle
    
    def _detect_shaking(self, yaws: List[float]) -> bool:
        """Detect head shaking motion"""
        if len(yaws) < 5:
            return False
        
        # Simple shaking detection based on yaw oscillations
        yaw_changes = np.diff(yaws)
        zero_crossings = np.where(np.diff(np.sign(yaw_changes)))[0]
        
        return len(zero_crossings) >= 2  # At least one complete cycle
