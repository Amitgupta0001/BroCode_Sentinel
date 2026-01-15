import numpy as np
from typing import Dict, List, Optional
import json
import hashlib
from dataclasses import dataclass

@dataclass
class BehavioralTemplate:
    user_id: str
    behavioral_features: Dict[str, np.ndarray]
    feature_weights: Dict[str, float]
    confidence_threshold: float
    enrollment_timestamp: float

class BehavioralEnroller:
    def __init__(self, min_enrollment_samples: int = 10):
        self.min_enrollment_samples = min_enrollment_samples
        self.enrollment_data = {}
        
    def enroll_user(self, user_id: str, behavioral_samples: List[Dict]) -> BehavioralTemplate:
        """Enroll user by collecting behavioral samples"""
        if len(behavioral_samples) < self.min_enrollment_samples:
            raise ValueError(f"Need at least {self.min_enrollment_samples} samples for enrollment")
        
        # Extract and aggregate features
        aggregated_features = self._aggregate_enrollment_features(behavioral_samples)
        
        # Calculate feature weights based on consistency
        feature_weights = self._calculate_feature_weights(behavioral_samples)
        
        # Determine confidence threshold
        confidence_threshold = self._calculate_confidence_threshold(behavioral_samples)
        
        template = BehavioralTemplate(
            user_id=user_id,
            behavioral_features=aggregated_features,
            feature_weights=feature_weights,
            confidence_threshold=confidence_threshold,
            enrollment_timestamp=time.time()
        )
        
        self.enrollment_data[user_id] = template
        return template
    
    def _aggregate_enrollment_features(self, samples: List[Dict]) -> Dict[str, np.ndarray]:
        """Aggregate features from multiple enrollment samples"""
        feature_accumulator = {}
        
        for sample in samples:
            for feature_type, features in sample.items():
                if feature_type not in feature_accumulator:
                    feature_accumulator[feature_type] = []
                feature_accumulator[feature_type].append(features)
        
        # Calculate mean and standard deviation for each feature
        aggregated = {}
        for feature_type, feature_list in feature_accumulator.items():
            feature_array = np.array(feature_list)
            aggregated[f"{feature_type}_mean"] = np.mean(feature_array, axis=0)
            aggregated[f"{feature_type}_std"] = np.std(feature_array, axis=0)
            
        return aggregated
    
    def _calculate_feature_weights(self, samples: List[Dict]) -> Dict[str, float]:
        """Calculate weights for each feature based on user consistency"""
        consistency_scores = {}
        
        for sample in samples:
            for feature_type, features in sample.items():
                if feature_type not in consistency_scores:
                    consistency_scores[feature_type] = []
                # Calculate feature stability (inverse of variance)
                if hasattr(features, 'std'):
                    consistency_scores[feature_type].append(1.0 / (features.std() + 1e-8))
        
        # Normalize weights to sum to 1
        total_consistency = sum(np.mean(scores) for scores in consistency_scores.values())
        weights = {
            feature_type: np.mean(scores) / total_consistency
            for feature_type, scores in consistency_scores.items()
        }
        
        return weights
    
    def _calculate_confidence_threshold(self, samples: List[Dict]) -> float:
        """Calculate authentication threshold based on enrollment data"""
        # Use statistical analysis of enrollment samples
        similarity_scores = []
        
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                similarity = self._calculate_sample_similarity(samples[i], samples[j])
                similarity_scores.append(similarity)
        
        # Set threshold at mean - 2 standard deviations
        threshold = np.mean(similarity_scores) - 2 * np.std(similarity_scores)
        return max(threshold, 0.3)  # Minimum threshold
