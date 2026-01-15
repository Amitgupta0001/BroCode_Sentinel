import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class PoseLandmarks:
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray

@dataclass
class PoseAnalysis:
    posture_quality: float
    symmetry_score: float
    movement_smoothness: float
    detected_actions: List[str]
    body_angles: Dict[str, float]

class PoseEstimator:
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1):
        self.mp_pose = None
        self.pose = None
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions'):
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=static_image_mode,
                    model_complexity=model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            else:
                print("[WARN] MediaPipe solutions not found. Using fallback pose estimation.")
        except Exception as e:
            print(f"[WARN] MediaPipe init failed: {e}. Using fallback.")

        self.landmark_names = {
            0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
            # ... (truncated for brevity)
        }
        
    def estimate_pose(self, image: np.ndarray) -> Optional[PoseLandmarks]:
        """Estimate pose landmarks from image"""
        if self.pose is None:
             # Fallback mock for when MediaPipe is broken/missing
             # Returns a generic "frontal" pose so the app doesn't crash
             mock_landmarks = np.zeros((33, 3))
             mock_world = np.zeros((33, 3))
             mock_vis = np.ones(33)
             mock_pres = np.ones(33)
             return PoseLandmarks(mock_landmarks, mock_world, mock_vis, mock_pres)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            results = self.pose.process(rgb_image)
            if not results.pose_landmarks:
                return None
            
            # Extract landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
            visibility = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
            presence = np.array([lm.presence for lm in results.pose_landmarks.landmark])
            
            return PoseLandmarks(landmarks, world_landmarks, visibility, presence)
        except Exception as e:
            print(f"Pose processing error: {e}")
            return None
    
    def analyze_posture(self, landmarks: np.ndarray) -> PoseAnalysis:
        """Analyze posture quality and body symmetry"""
        # Calculate body angles
        body_angles = self._calculate_body_angles(landmarks)
        
        # Posture quality based on spine alignment
        posture_quality = self._calculate_posture_quality(landmarks)
        
        # Body symmetry
        symmetry_score = self._calculate_symmetry_score(landmarks)
        
        # Movement smoothness (requires temporal data)
        movement_smoothness = self._calculate_movement_smoothness(landmarks)
        
        # Detect actions/poses
        detected_actions = self._detect_actions(landmarks, body_angles)
        
        return PoseAnalysis(
            posture_quality=posture_quality,
            symmetry_score=symmetry_score,
            movement_smoothness=movement_smoothness,
            detected_actions=detected_actions,
            body_angles=body_angles
        )
    
    def _calculate_body_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate key body joint angles"""
        angles = {}
        
        # Shoulder angles
        angles['left_shoulder'] = self._calculate_angle(
            landmarks[11], landmarks[13], landmarks[15]  # Left shoulder, elbow, wrist
        )
        angles['right_shoulder'] = self._calculate_angle(
            landmarks[12], landmarks[14], landmarks[16]  # Right shoulder, elbow, wrist
        )
        
        # Hip angles
        angles['left_hip'] = self._calculate_angle(
            landmarks[23], landmarks[25], landmarks[27]  # Left hip, knee, ankle
        )
        angles['right_hip'] = self._calculate_angle(
            landmarks[24], landmarks[26], landmarks[28]  # Right hip, knee, ankle
        )
        
        # Spine angle (simplified)
        angles['spine'] = self._calculate_angle(
            landmarks[11], landmarks[23], landmarks[25]  # Shoulder, hip, knee
        )
        
        return angles
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle between three points"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_posture_quality(self, landmarks: np.ndarray) -> float:
        """Calculate posture quality score (0-1)"""
        # Analyze spine alignment
        shoulder_avg = (landmarks[11] + landmarks[12]) / 2
        hip_avg = (landmarks[23] + landmarks[24]) / 2
        
        # Vertical alignment (ideal posture has vertical spine)
        spine_vector = shoulder_avg - hip_avg
        vertical = np.array([0, 1, 0])  # Assuming y-axis is vertical
        
        # Calculate deviation from vertical
        deviation = np.arccos(np.clip(
            np.dot(spine_vector, vertical) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical)),
            -1.0, 1.0
        ))
        
        # Convert to quality score (0-1)
        max_deviation = np.radians(30)  # 30 degrees maximum acceptable
        quality = 1.0 - (deviation / max_deviation)
        
        return float(np.clip(quality, 0, 1))
    
    def _calculate_symmetry_score(self, landmarks: np.ndarray) -> float:
        """Calculate body symmetry score (0-1)"""
        left_right_pairs = [
            (11, 12), (13, 14), (15, 16),  # Upper body
            (23, 24), (25, 26), (27, 28)   # Lower body
        ]
        
        symmetry_scores = []
        for left_idx, right_idx in left_right_pairs:
            left_pos = landmarks[left_idx]
            right_pos = landmarks[right_idx]
            
            # Mirror right side for comparison
            right_pos_mirrored = right_pos * np.array([-1, 1, 1])  # Flip x-axis
            
            # Calculate symmetry error
            error = np.linalg.norm(left_pos - right_pos_mirrored)
            symmetry_scores.append(1.0 - np.clip(error / 0.2, 0, 1))  # Normalize
        
        return float(np.mean(symmetry_scores))
    
    def _calculate_movement_smoothness(self, landmarks: np.ndarray) -> float:
        """Calculate movement smoothness (requires temporal context)"""
        # This would typically use multiple frames
        # For single frame, return baseline
        return 0.8
    
    def _detect_actions(self, landmarks: np.ndarray, body_angles: Dict[str, float]) -> List[str]:
        """Detect specific actions or poses"""
        actions = []
        
        # Sitting detection
        if body_angles['left_hip'] < 90 and body_angles['right_hip'] < 90:
            actions.append("sitting")
        else:
            actions.append("standing")
        
        # Arm crossing detection
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        if abs(left_wrist[0] - right_wrist[0]) < 0.1:  # Wrists close together
            actions.append("arms_crossed")
        
        # Leaning detection
        spine_angle = body_angles['spine']
        if spine_angle < 70:
            actions.append("leaning_forward")
        elif spine_angle > 110:
            actions.append("leaning_backward")
        
        return actions
