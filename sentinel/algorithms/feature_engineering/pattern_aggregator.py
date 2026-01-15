import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import statistics

@dataclass
class AggregatedPattern:
    pattern_id: str
    features: Dict[str, float]
    confidence: float
    temporal_evolution: List[float]
    metadata: Dict[str, Any]

class PatternAggregator:
    def __init__(self, aggregation_window: int = 30, overlap: float = 0.5):
        self.aggregation_window = aggregation_window
        self.overlap = overlap
        self.pattern_buffer = []
        self.feature_history = defaultdict(list)
        
    def aggregate_temporal_patterns(self, 
                                  patterns: List[Dict[str, Any]],
                                  timestamps: List[float]) -> List[AggregatedPattern]:
        """Aggregate patterns over time windows"""
        if not patterns or len(patterns) != len(timestamps):
            return []
        
        aggregated_patterns = []
        window_start = 0
        
        while window_start < len(patterns):
            window_end = min(window_start + self.aggregation_window, len(patterns))
            window_patterns = patterns[window_start:window_end]
            window_timestamps = timestamps[window_start:window_end]
            
            if len(window_patterns) >= self.aggregation_window // 2:  # Minimum patterns for aggregation
                aggregated_pattern = self._aggregate_window_patterns(window_patterns, window_timestamps)
                aggregated_patterns.append(aggregated_pattern)
            
            # Move window with overlap
            window_start += int(self.aggregation_window * (1 - self.overlap))
        
        return aggregated_patterns
    
    def _aggregate_window_patterns(self, 
                                 patterns: List[Dict[str, Any]],
                                 timestamps: List[float]) -> AggregatedPattern:
        """Aggregate patterns within a single time window"""
        # Extract all features from patterns
        feature_vectors = []
        for pattern in patterns:
            features = self._extract_features_from_pattern(pattern)
            feature_vectors.append(features)
        
        # Aggregate features
        aggregated_features = self._aggregate_feature_vectors(feature_vectors)
        
        # Calculate confidence
        confidence = self._calculate_aggregation_confidence(feature_vectors)
        
        # Track temporal evolution
        temporal_evolution = self._extract_temporal_evolution(feature_vectors)
        
        # Generate pattern ID
        pattern_id = self._generate_pattern_id(aggregated_features, timestamps[0])
        
        return AggregatedPattern(
            pattern_id=pattern_id,
            features=aggregated_features,
            confidence=confidence,
            temporal_evolution=temporal_evolution,
            metadata={
                'window_start': timestamps[0],
                'window_end': timestamps[-1],
                'pattern_count': len(patterns),
                'feature_variability': self._calculate_feature_variability(feature_vectors)
            }
        )
    
    def _extract_features_from_pattern(self, pattern: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from a pattern object"""
        features = {}
        
        # Extract basic pattern properties
        if 'pattern_type' in pattern:
            # Encode pattern type as numerical feature
            pattern_types = ['rhythmic', 'stable', 'erratic', 'trending', 'variable']
            if pattern['pattern_type'] in pattern_types:
                features['pattern_type_encoded'] = pattern_types.index(pattern['pattern_type']) / len(pattern_types)
        
        # Extract temporal features
        if 'periodicity' in pattern:
            features['periodicity'] = pattern['periodicity']
        if 'strength' in pattern:
            features['strength'] = pattern['strength']
        if 'consistency' in pattern:
            features['consistency'] = pattern['consistency']
        
        # Extract segment information
        if 'segments' in pattern:
            segments = pattern['segments']
            if segments:
                segment_lengths = [end - start for start, end in segments]
                features['segment_count'] = len(segments)
                features['avg_segment_length'] = np.mean(segment_lengths)
                features['segment_length_variance'] = np.var(segment_lengths)
        
        return features
    
    def _aggregate_feature_vectors(self, feature_vectors: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate multiple feature vectors into a single representation"""
        if not feature_vectors:
            return {}
        
        # Collect all features across vectors
        all_features = defaultdict(list)
        for feature_vec in feature_vectors:
            for feature_name, value in feature_vec.items():
                all_features[feature_name].append(value)
        
        # Aggregate each feature
        aggregated_features = {}
        for feature_name, values in all_features.items():
            # Use robust aggregation methods
            aggregated_features[f"{feature_name}_mean"] = float(np.mean(values))
            aggregated_features[f"{feature_name}_median"] = float(np.median(values))
            aggregated_features[f"{feature_name}_std"] = float(np.std(values))
            aggregated_features[f"{feature_name}_q25"] = float(np.percentile(values, 25))
            aggregated_features[f"{feature_name}_q75"] = float(np.percentile(values, 75))
            
            # For some features, also include min/max
            if feature_name in ['strength', 'consistency', 'confidence']:
                aggregated_features[f"{feature_name}_min"] = float(np.min(values))
                aggregated_features[f"{feature_name}_max"] = float(np.max(values))
        
        return aggregated_features
    
    def _calculate_aggregation_confidence(self, feature_vectors: List[Dict[str, float]]) -> float:
        """Calculate confidence in the aggregation"""
        if len(feature_vectors) < 2:
            return 0.5
        
        # Confidence based on feature consistency
        consistency_scores = []
        
        for feature_name in feature_vectors[0].keys():
            values = [vec[feature_name] for vec in feature_vectors if feature_name in vec]
            if len(values) > 1:
                # Calculate coefficient of variation (inverse for confidence)
                cv = np.std(values) / (np.mean(values) + 1e-8)
                feature_consistency = 1.0 / (1.0 + cv)
                consistency_scores.append(feature_consistency)
        
        if consistency_scores:
            overall_consistency = np.mean(consistency_scores)
            # Adjust for sample size
            sample_size_confidence = min(len(feature_vectors) / 10.0, 1.0)
            confidence = overall_consistency * sample_size_confidence
        else:
            confidence = 0.5
        
        return float(np.clip(confidence, 0, 1))
    
    def _extract_temporal_evolution(self, feature_vectors: List[Dict[str, float]]) -> List[float]:
        """Extract temporal evolution of key features"""
        if len(feature_vectors) < 3:
            return []
        
        # Track evolution of consistency feature
        consistency_evolution = []
        for feature_vec in feature_vectors:
            if 'consistency' in feature_vec:
                consistency_evolution.append(feature_vec['consistency'])
            else:
                consistency_evolution.append(0.5)
        
        return consistency_evolution
    
    def _calculate_feature_variability(self, feature_vectors: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate variability metrics for features"""
        variability = {}
        
        if len(feature_vectors) < 2:
            return variability
        
        for feature_name in feature_vectors[0].keys():
            values = [vec[feature_name] for vec in feature_vectors if feature_name in vec]
            if len(values) > 1:
                variability[feature_name] = {
                    'range': float(np.max(values) - np.min(values)),
                    'variance': float(np.var(values)),
                    'cv': float(np.std(values) / (np.mean(values) + 1e-8))
                }
        
        return variability
    
    def _generate_pattern_id(self, features: Dict[str, float], timestamp: float) -> str:
        """Generate a unique pattern ID based on features and timestamp"""
        # Use hash of key feature values and timestamp
        key_features = [
            features.get('consistency_mean', 0),
            features.get('periodicity_mean', 0),
            features.get('strength_mean', 0)
        ]
        
        feature_hash = hash(tuple(key_features + [timestamp]))
        return f"pattern_{abs(feature_hash) % 10000:04d}"
    
    def cross_modality_aggregation(self, 
                                 modality_patterns: Dict[str, List[AggregatedPattern]]) -> AggregatedPattern:
        """Aggregate patterns across different modalities"""
        all_features = []
        all_confidences = []
        
        for modality, patterns in modality_patterns.items():
            for pattern in patterns:
                # Add modality prefix to features
                modality_features = {}
                for feature_name, value in pattern.features.items():
                    modality_features[f"{modality}_{feature_name}"] = value
                
                all_features.append(modality_features)
                all_confidences.append(pattern.confidence)
        
        if not all_features:
            return self._get_empty_aggregated_pattern()
        
        # Aggregate cross-modality features
        aggregated_features = self._aggregate_feature_vectors(all_features)
        
        # Calculate cross-modality confidence
        cross_modality_confidence = np.mean(all_confidences) if all_confidences else 0.5
        
        return AggregatedPattern(
            pattern_id="cross_modality_aggregated",
            features=aggregated_features,
            confidence=float(cross_modality_confidence),
            temporal_evolution=[],
            metadata={
                'modalities': list(modality_patterns.keys()),
                'total_patterns': sum(len(patterns) for patterns in modality_patterns.values())
            }
        )
    
    def _get_empty_aggregated_pattern(self) -> AggregatedPattern:
        """Return an empty aggregated pattern"""
        return AggregatedPattern(
            pattern_id="empty",
            features={},
            confidence=0.0,
            temporal_evolution=[],
            metadata={'empty': True}
        )
    
    def update_feature_history(self, pattern: AggregatedPattern):
        """Update feature history for long-term analysis"""
        for feature_name, value in pattern.features.items():
            self.feature_history[feature_name].append(value)
            
            # Keep only recent history
            if len(self.feature_history[feature_name]) > 100:
                self.feature_history[feature_name].pop(0)
    
    def get_feature_trends(self) -> Dict[str, Dict[str, float]]:
        """Calculate trends in feature values over time"""
        trends = {}
        
        for feature_name, values in self.feature_history.items():
            if len(values) >= 5:
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trends[feature_name] = {
                    'slope': float(slope),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'trend_strength': abs(float(slope)) * r_value**2
                }
        
        return trends
