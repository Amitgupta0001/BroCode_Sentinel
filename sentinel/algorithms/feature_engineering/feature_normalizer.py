import numpy as np
from typing import Dict, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from dataclasses import dataclass

@dataclass
class NormalizationConfig:
    method: str  # 'standard', 'minmax', 'robust'
    feature_ranges: Dict[str, tuple]
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]

class BehavioralFeatureNormalizer:
    def __init__(self, normalization_method: str = 'standard'):
        self.normalization_method = normalization_method
        self.scalers = {}
        self.feature_configs = {}
        self.is_fitted = False
        
    def fit(self, feature_data: List[Dict[str, float]]):
        """Fit normalizers on feature data"""
        if not feature_data:
            raise ValueError("No feature data provided for fitting")
        
        # Convert list of dicts to feature matrix
        feature_names = list(feature_data[0].keys())
        feature_matrix = np.array([[sample[feature] for feature in feature_names] 
                                 for sample in feature_data])
        
        # Initialize and fit scalers for each feature
        for i, feature_name in enumerate(feature_names):
            feature_values = feature_matrix[:, i]
            
            if self.normalization_method == 'standard':
                scaler = StandardScaler()
            elif self.normalization_method == 'minmax':
                scaler = MinMaxScaler(feature_range=(-1, 1))
            elif self.normalization_method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported normalization method: {self.normalization_method}")
            
            # Fit scaler
            scaler.fit(feature_values.reshape(-1, 1))
            self.scalers[feature_name] = scaler
            
            # Store feature statistics
            self.feature_configs[feature_name] = {
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values)),
                'q25': float(np.percentile(feature_values, 25)),
                'q75': float(np.percentile(feature_values, 75))
            }
        
        self.is_fitted = True
    
    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """Transform features using fitted normalizers"""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transformation")
        
        normalized_features = {}
        
        for feature_name, value in features.items():
            if feature_name in self.scalers:
                scaler = self.scalers[feature_name]
                normalized_value = scaler.transform([[value]])[0][0]
                normalized_features[feature_name] = float(normalized_value)
            else:
                # If feature wasn't seen during training, use identity transformation
                normalized_features[feature_name] = value
                print(f"Warning: Feature '{feature_name}' not seen during training, using original value")
        
        return normalized_features
    
    def inverse_transform(self, normalized_features: Dict[str, float]) -> Dict[str, float]:
        """Transform normalized features back to original scale"""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transformation")
        
        original_features = {}
        
        for feature_name, norm_value in normalized_features.items():
            if feature_name in self.scalers:
                scaler = self.scalers[feature_name]
                original_value = scaler.inverse_transform([[norm_value]])[0][0]
                original_features[feature_name] = float(original_value)
            else:
                original_features[feature_name] = norm_value
        
        return original_features
    
    def normalize_feature_group(self, 
                              feature_group: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normalize a group of feature sets (e.g., temporal, spatial features)"""
        normalized_group = {}
        
        for group_name, features in feature_group.items():
            normalized_group[group_name] = self.transform(features)
        
        return normalized_group
    
    def adaptive_normalization(self, 
                             features: Dict[str, float],
                             reference_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Perform adaptive normalization using reference statistics"""
        normalized_features = {}
        
        for feature_name, value in features.items():
            if feature_name in reference_stats:
                stats = reference_stats[feature_name]
                
                if self.normalization_method == 'standard':
                    # Z-score normalization
                    normalized_value = (value - stats['mean']) / max(stats['std'], 1e-8)
                elif self.normalization_method == 'minmax':
                    # Min-max normalization to [-1, 1]
                    value_range = stats['max'] - stats['min']
                    if value_range > 1e-8:
                        normalized_value = 2 * (value - stats['min']) / value_range - 1
                    else:
                        normalized_value = 0.0
                elif self.normalization_method == 'robust':
                    # Robust scaling using IQR
                    iqr = stats['q75'] - stats['q25']
                    if iqr > 1e-8:
                        normalized_value = (value - stats['median']) / iqr
                    else:
                        normalized_value = (value - stats['median']) / max(stats['std'], 1e-8)
                
                normalized_features[feature_name] = float(normalized_value)
            else:
                normalized_features[feature_name] = value
        
        return normalized_features
    
    def handle_outliers(self, features: Dict[str, float], 
                       method: str = 'clip') -> Dict[str, float]:
        """Handle outliers in features"""
        processed_features = {}
        
        for feature_name, value in features.items():
            if feature_name in self.feature_configs:
                stats = self.feature_configs[feature_name]
                
                if method == 'clip':
                    # Clip to reasonable range (mean ± 3 std)
                    lower_bound = stats['mean'] - 3 * stats['std']
                    upper_bound = stats['mean'] + 3 * stats['std']
                    clipped_value = np.clip(value, lower_bound, upper_bound)
                    processed_features[feature_name] = float(clipped_value)
                
                elif method == 'winsorize':
                    # Winsorize to 1st and 99th percentiles
                    lower_bound = stats.get('q1', stats['min'])
                    upper_bound = stats.get('q99', stats['max'])
                    winsorized_value = np.clip(value, lower_bound, upper_bound)
                    processed_features[feature_name] = float(winsorized_value)
                
                else:
                    processed_features[feature_name] = value
            else:
                processed_features[feature_name] = value
        
        return processed_features
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all features"""
        return self.feature_configs.copy()
    
    def save_normalizer(self, filepath: str):
        """Save fitted normalizer to file"""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before saving")
        
        save_data = {
            'scalers': self.scalers,
            'feature_configs': self.feature_configs,
            'normalization_method': self.normalization_method,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
    
    def load_normalizer(self, filepath: str):
        """Load normalizer from file"""
        load_data = joblib.load(filepath)
        
        self.scalers = load_data['scalers']
        self.feature_configs = load_data['feature_configs']
        self.normalization_method = load_data['normalization_method']
        self.is_fitted = load_data['is_fitted']
