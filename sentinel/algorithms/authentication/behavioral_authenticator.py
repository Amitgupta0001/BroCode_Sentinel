import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial.distance import mahalanobis
from dataclasses import dataclass
from sentinel.ml.io import save_model, load_model
import time

@dataclass
class BehavioralTemplate: # Added stub to prevent definition errors
    behavioral_features: Dict
    feature_weights: Dict
    confidence_threshold: float = 0.6
    metadata: Dict = None

class BehavioralEnroller: # Placeholder to ensure imports work
    def enroll_user(self, user_id, samples):
        return BehavioralTemplate({}, {}, 0.6)

@dataclass
class AuthenticationResult:
    is_authenticated: bool
    confidence_score: float
    risk_factors: List[str]
    behavioral_consistency: float
    timestamp: float

class BehavioralAuthenticator:
    def __init__(self, model_path: str = None, template_database: Dict[str, BehavioralTemplate] = None):
        self.model_path = model_path
        self.model = None
        self.template_database = template_database if template_database is not None else {}
        self.continuous_confidence = {}
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def save(self, path: str, metadata: dict = None):
        save_model(path, self.model, metadata=metadata)

    def load(self, path: str):
        mdl, meta = load_model(path)
        self.model = mdl
        return self.model

    def authenticate_user(self, user_id: str, current_behavior: Dict, 
                         video_frame: np.ndarray = None) -> AuthenticationResult:
        """Authenticate user based on current behavior"""
        if user_id not in self.template_database:
            return AuthenticationResult(
                is_authenticated=False,
                confidence_score=0.0,
                risk_factors=["user_not_enrolled"],
                behavioral_consistency=0.0,
                timestamp=time.time()
            )
        
        template = self.template_database[user_id]
        
        # Extract features from current behavior
        current_features = self._extract_authentication_features(current_behavior, video_frame)
        
        # Calculate similarity score
        similarity_score = self._calculate_similarity_score(current_features, template)
        
        # Check for behavioral anomalies
        risk_factors = self._detect_risk_factors(current_features, template)
        
        # Update continuous confidence
        self._update_continuous_confidence(user_id, similarity_score)
        
        # Calculate behavioral consistency
        consistency = self._calculate_behavioral_consistency(user_id)
        
        # Make authentication decision
        is_authenticated = self._make_authentication_decision(
            similarity_score, template.confidence_threshold, risk_factors
        )
        
        return AuthenticationResult(
            is_authenticated=is_authenticated,
            confidence_score=similarity_score,
            risk_factors=risk_factors,
            behavioral_consistency=consistency,
            timestamp=time.time()
        )
    
    def _extract_authentication_features(self, behavior: Dict, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features for authentication"""
        features = {}
        
        # Gaze-based features
        if 'gaze_analysis' in behavior:
            gaze = behavior['gaze_analysis']
            features['gaze_stability'] = np.array([gaze.get('gaze_stability', 0.5)])
            features['attention_score'] = np.array([gaze.get('attention_score', 0.5)])
            features['saccade_frequency'] = np.array([gaze.get('saccade_frequency', 0.0)])
        
        # Facial features
        if 'facial_analysis' in behavior:
            facial = behavior['facial_analysis']
            features['blink_rate'] = np.array([facial.get('blink_rate', 0.2)])
            features['head_movement'] = np.array([facial.get('head_movement_variance', 0.1)])
        
        # Body movement features
        if 'body_analysis' in behavior:
            body = behavior['body_analysis']
            features['posture_consistency'] = np.array([body.get('posture_consistency', 0.5)])
            features['gesture_frequency'] = np.array([body.get('gesture_frequency', 0.0)])
        
        # Temporal patterns
        if 'temporal_analysis' in behavior:
            temporal = behavior['temporal_analysis']
            features['behavior_rhythm'] = np.array([temporal.get('rhythm_consistency', 0.5)])
        
        return features
    
    def _calculate_similarity_score(self, current_features: Dict[str, np.ndarray], 
                                  template: BehavioralTemplate) -> float:
        """Calculate similarity between current behavior and enrolled template"""
        similarity_scores = []
        
        for feature_type, current_value in current_features.items():
            if feature_type + '_mean' in template.behavioral_features:
                # Get template statistics
                template_mean = template.behavioral_features[feature_type + '_mean']
                template_std = template.behavioral_features[feature_type + '_std']
                feature_weight = template.feature_weights.get(feature_type, 1.0)
                
                # Calculate normalized similarity
                if np.any(template_std > 0):
                    # Use Mahalanobis distance for correlated features
                    z_score = np.abs((current_value - template_mean) / template_std)
                    feature_similarity = np.exp(-np.mean(z_score))
                else:
                    # Simple absolute difference for constant features
                    feature_similarity = 1.0 - min(np.abs(current_value - template_mean), 1.0)
                
                # Apply feature weight
                weighted_similarity = feature_similarity * feature_weight
                similarity_scores.append(weighted_similarity)
        
        return float(np.mean(similarity_scores)) if similarity_scores else 0.0
    
    def _detect_risk_factors(self, current_features: Dict[str, np.ndarray], 
                           template: BehavioralTemplate) -> List[str]:
        """Detect potential risk factors for authentication"""
        risk_factors = []
        
        # Check for significant deviations
        for feature_type, current_value in current_features.items():
            if feature_type + '_mean' in template.behavioral_features:
                template_mean = template.behavioral_features[feature_type + '_mean']
                template_std = template.behavioral_features[feature_type + '_std']
                
                if np.any(template_std > 0):
                    z_score = np.abs((current_value - template_mean) / template_std)
                    if np.any(z_score > 3.0):  # 3 standard deviations
                        risk_factors.append(f"abnormal_{feature_type}")
        
        # Check for missing key features
        required_features = ['gaze_stability', 'posture_consistency']
        for req_feature in required_features:
            if req_feature not in current_features:
                risk_factors.append(f"missing_{req_feature}")
        
        return risk_factors
    
    def _update_continuous_confidence(self, user_id: str, current_score: float):
        """Update continuous authentication confidence"""
        if user_id not in self.continuous_confidence:
            self.continuous_confidence[user_id] = []
        
        self.continuous_confidence[user_id].append(current_score)
        
        # Keep only recent scores (sliding window)
        if len(self.continuous_confidence[user_id]) > 10:
            self.continuous_confidence[user_id].pop(0)
    
    def _calculate_behavioral_consistency(self, user_id: str) -> float:
        """Calculate consistency of recent behavioral patterns"""
        if user_id not in self.continuous_confidence or len(self.continuous_confidence[user_id]) < 2:
            return 0.0
        
        scores = self.continuous_confidence[user_id]
        consistency = 1.0 - np.std(scores)  # Higher std = lower consistency
        return float(max(consistency, 0.0))
    
    def _make_authentication_decision(self, similarity_score: float, 
                                   threshold: float, risk_factors: List[str]) -> bool:
        """Make final authentication decision"""
        # Adjust threshold based on risk factors
        adjusted_threshold = threshold
        if risk_factors:
            # Increase threshold if risk factors are present
            risk_penalty = len(risk_factors) * 0.1
            adjusted_threshold = min(threshold + risk_penalty, 0.9)
        
        return similarity_score >= adjusted_threshold and len(risk_factors) < 3
