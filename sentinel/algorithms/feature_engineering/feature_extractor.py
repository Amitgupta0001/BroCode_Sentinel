import numpy as np
from typing import Dict, List, Tuple, Any
import cv2
from scipy import stats
from dataclasses import dataclass

@dataclass
class BehavioralFeatures:
    temporal_features: Dict[str, float]
    spatial_features: Dict[str, float]
    statistical_features: Dict[str, float]
    spectral_features: Dict[str, float]

class BehavioralFeatureExtractor:
    def __init__(self, window_size: int = 30, fps: int = 30):
        self.window_size = window_size
        self.fps = fps
        
    def extract_comprehensive_features(self, 
                                     gaze_data: List[Dict],
                                     pose_data: List[Dict],
                                     facial_data: List[Dict],
                                     gesture_data: List[Dict]) -> BehavioralFeatures:
        """Extract comprehensive behavioral features from multimodal data"""
        
        temporal_features = self._extract_temporal_features(gaze_data, pose_data, facial_data)
        spatial_features = self._extract_spatial_features(pose_data, gaze_data)
        statistical_features = self._extract_statistical_features(gaze_data, pose_data, facial_data)
        spectral_features = self._extract_spectral_features(gaze_data, pose_data)
        
        return BehavioralFeatures(
            temporal_features=temporal_features,
            spatial_features=spatial_features,
            statistical_features=statistical_features,
            spectral_features=spectral_features
        )
    
    def _extract_temporal_features(self, 
                                 gaze_data: List[Dict],
                                 pose_data: List[Dict],
                                 facial_data: List[Dict]) -> Dict[str, float]:
        """Extract temporal dynamics features"""
        features = {}
        
        # Gaze temporal features
        if gaze_data:
            gaze_points = np.array([(g['x'], g['y']) for g in gaze_data])
            gaze_velocities = np.linalg.norm(np.diff(gaze_points, axis=0), axis=1)
            
            features['gaze_velocity_mean'] = float(np.mean(gaze_velocities))
            features['gaze_velocity_std'] = float(np.std(gaze_velocities))
            features['gaze_velocity_entropy'] = float(stats.entropy(np.histogram(gaze_velocities, bins=10)[0]))
            
            # Fixation and saccade analysis
            fixations = gaze_velocities < 50.0  # pixels per second threshold
            features['fixation_ratio'] = float(np.mean(fixations))
            features['saccade_frequency'] = float(np.sum(~fixations) / (len(gaze_data) / self.fps))
        
        # Pose temporal features
        if pose_data and len(pose_data) > 1:
            # Use shoulder movement as proxy for overall movement
            shoulder_positions = np.array([(p['landmarks'][11], p['landmarks'][12]) 
                                        for p in pose_data if 'landmarks' in p])
            if len(shoulder_positions) > 1:
                shoulder_movement = np.linalg.norm(np.diff(shoulder_positions.mean(axis=1), axis=0))
                features['body_movement_energy'] = float(np.mean(shoulder_movement))
        
        # Facial expression temporal features
        if facial_data:
            emotion_changes = []
            for i in range(1, len(facial_data)):
                if 'emotion' in facial_data[i] and 'emotion' in facial_data[i-1]:
                    if facial_data[i]['emotion'] != facial_data[i-1]['emotion']:
                        emotion_changes.append(1)
                    else:
                        emotion_changes.append(0)
            
            if emotion_changes:
                features['emotion_transition_rate'] = float(np.mean(emotion_changes))
        
        return features
    
    def _extract_spatial_features(self, 
                                pose_data: List[Dict],
                                gaze_data: List[Dict]) -> Dict[str, float]:
        """Extract spatial configuration features"""
        features = {}
        
        if pose_data:
            # Body symmetry features
            latest_pose = pose_data[-1] if pose_data else {}
            if 'landmarks' in latest_pose:
                landmarks = latest_pose['landmarks']
                
                # Shoulder symmetry
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                shoulder_symmetry = 1.0 - np.linalg.norm(left_shoulder - right_shoulder)
                features['shoulder_symmetry'] = float(shoulder_symmetry)
                
                # Posture openness (arm positions)
                left_elbow = landmarks[13]
                right_elbow = landmarks[14]
                elbow_separation = np.linalg.norm(left_elbow - right_elbow)
                features['posture_openness'] = float(elbow_separation)
        
        if gaze_data:
            # Gaze distribution features
            gaze_points = np.array([(g['x'], g['y']) for g in gaze_data])
            if len(gaze_points) > 1:
                gaze_centroid = np.mean(gaze_points, axis=0)
                gaze_spread = np.mean(np.linalg.norm(gaze_points - gaze_centroid, axis=1))
                features['gaze_spread'] = float(gaze_spread)
                
                # Gaze quadrant distribution
                screen_center = np.array([0.5, 0.5])
                quadrants = [0, 0, 0, 0]
                for point in gaze_points:
                    if point[0] < 0.5 and point[1] < 0.5:
                        quadrants[0] += 1
                    elif point[0] >= 0.5 and point[1] < 0.5:
                        quadrants[1] += 1
                    elif point[0] < 0.5 and point[1] >= 0.5:
                        quadrants[2] += 1
                    else:
                        quadrants[3] += 1
                
                quadrant_entropy = stats.entropy(quadrants)
                features['gaze_distribution_entropy'] = float(quadrant_entropy)
        
        return features
    
    def _extract_statistical_features(self,
                                    gaze_data: List[Dict],
                                    pose_data: List[Dict],
                                    facial_data: List[Dict]) -> Dict[str, float]:
        """Extract statistical distribution features"""
        features = {}
        
        # Gaze statistical features
        if gaze_data:
            gaze_x = [g['x'] for g in gaze_data]
            gaze_y = [g['y'] for g in gaze_data]
            
            features['gaze_x_skewness'] = float(stats.skew(gaze_x))
            features['gaze_y_skewness'] = float(stats.skew(gaze_y))
            features['gaze_x_kurtosis'] = float(stats.kurtosis(gaze_x))
            features['gaze_y_kurtosis'] = float(stats.kurtosis(gaze_y))
        
        # Pose statistical features
        if pose_data:
            # Use head position variability
            head_positions = []
            for pose in pose_data:
                if 'landmarks' in pose and len(pose['landmarks']) > 0:
                    head_positions.append(pose['landmarks'][0])  # Nose landmark
            
            if head_positions:
                head_positions = np.array(head_positions)
                head_variability = np.mean(np.std(head_positions, axis=0))
                features['head_position_variability'] = float(head_variability)
        
        # Facial expression statistical features
        if facial_data:
            emotion_values = []
            for data in facial_data:
                if 'emotion_confidence' in data:
                    emotion_values.append(data['emotion_confidence'])
            
            if emotion_values:
                features['emotion_intensity_mean'] = float(np.mean(emotion_values))
                features['emotion_intensity_variance'] = float(np.var(emotion_values))
        
        return features
    
    def _extract_spectral_features(self,
                                 gaze_data: List[Dict],
                                 pose_data: List[Dict]) -> Dict[str, float]:
        """Extract frequency domain features"""
        features = {}
        
        # Gaze spectral features
        if len(gaze_data) >= self.window_size:
            gaze_x = [g['x'] for g in gaze_data]
            gaze_y = [g['y'] for g in gaze_data]
            
            # Simple spectral analysis using FFT
            gaze_x_fft = np.fft.fft(gaze_x)
            gaze_y_fft = np.fft.fft(gaze_y)
            
            # Dominant frequencies
            gaze_x_freq = np.argmax(np.abs(gaze_x_fft[1:len(gaze_x_fft)//2]))
            gaze_y_freq = np.argmax(np.abs(gaze_y_fft[1:len(gaze_y_fft)//2]))
            
            features['gaze_x_dominant_freq'] = float(gaze_x_freq)
            features['gaze_y_dominant_freq'] = float(gaze_y_freq)
            
            # Spectral energy
            features['gaze_spectral_energy'] = float(np.sum(np.abs(gaze_x_fft)**2) + 
                                                   np.sum(np.abs(gaze_y_fft)**2))
        
        return features
    
    def extract_interpersonal_features(self, 
                                     person1_features: BehavioralFeatures,
                                     person2_features: BehavioralFeatures) -> Dict[str, float]:
        """Extract features for interpersonal behavior analysis"""
        features = {}
        
        # Synchrony features
        p1_gaze = person1_features.temporal_features.get('gaze_velocity_mean', 0)
        p2_gaze = person2_features.temporal_features.get('gaze_velocity_mean', 0)
        features['gaze_velocity_synchrony'] = float(1.0 - abs(p1_gaze - p2_gaze) / max(p1_gaze, p2_gaze, 1e-8))
        
        # Mirroring features
        p1_movement = person1_features.temporal_features.get('body_movement_energy', 0)
        p2_movement = person2_features.temporal_features.get('body_movement_energy', 0)
        features['movement_mirroring'] = float(1.0 - abs(p1_movement - p2_movement) / max(p1_movement, p2_movement, 1e-8))
        
        # Attention coordination
        p1_gaze_entropy = person1_features.spatial_features.get('gaze_distribution_entropy', 0)
        p2_gaze_entropy = person2_features.spatial_features.get('gaze_distribution_entropy', 0)
        features['gaze_pattern_similarity'] = float(1.0 - abs(p1_gaze_entropy - p2_gaze_entropy) / 
                                                  max(p1_gaze_entropy, p2_gaze_entropy, 1e-8))
        
        return features
