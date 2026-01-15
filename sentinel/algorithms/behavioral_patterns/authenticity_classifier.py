import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from dataclasses import dataclass

@dataclass
class AuthenticityResult:
    is_authentic: bool
    confidence: float
    authenticity_score: float
    indicators: Dict[str, float]
    risk_factors: List[str]

class AuthenticityClassifier:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        # Define authenticity indicators (based on psychological research)
        self.authenticity_indicators = {
            'consistency': 'Behavioral consistency over time',
            'congruence': 'Congruence between verbal and non-verbal cues',
            'spontaneity': 'Natural, spontaneous behavior patterns',
            'micro_expression_congruence': 'Congruence between macro and micro expressions',
            'response_latency': 'Appropriate response timing',
            'gaze_patterns': 'Natural gaze patterns and eye contact',
            'gesture_synchrony': 'Synchrony between speech and gestures'
        }
    
    def extract_authenticity_features(self, behavioral_data: Dict) -> np.ndarray:
        """Extract features relevant for authenticity classification"""
        features = []
        feature_names = []
        
        # Consistency features
        if 'temporal_consistency' in behavioral_data:
            consistency = behavioral_data['temporal_consistency']
            features.extend([
                consistency.get('behavior_consistency', 0.8),
                consistency.get('emotion_consistency', 0.7),
                consistency.get('response_consistency', 0.6)
            ])
            feature_names.extend(['behavior_consistency', 'emotion_consistency', 'response_consistency'])
        
        # Congruence features
        if 'multimodal_congruence' in behavioral_data:
            congruence = behavioral_data['multimodal_congruence']
            features.extend([
                congruence.get('verbal_nonverbal_congruence', 0.5),
                congruence.get('facial_vocal_congruence', 0.5),
                congruence.get('gesture_speech_synchrony', 0.5)
            ])
            feature_names.extend(['verbal_nonverbal_cong', 'facial_vocal_cong', 'gesture_speech_sync'])
        
        # Spontaneity features
        if 'behavioral_naturalness' in behavioral_data:
            naturalness = behavioral_data['behavioral_naturalness']
            features.extend([
                naturalness.get('movement_smoothness', 0.8),
                naturalness.get('rehearsal_indicators', 0.1),
                naturalness.get('spontaneous_gestures', 0.6)
            ])
            feature_names.extend(['movement_smoothness', 'rehearsal_indicators', 'spontaneous_gestures'])
        
        # Micro-expression analysis
        if 'micro_expression_analysis' in behavioral_data:
            micro_expr = behavioral_data['micro_expression_analysis']
            features.extend([
                micro_expr.get('congruence_score', 0.7),
                micro_expr.get('suppression_indicators', 0.2),
                micro_expr.get('leakage_frequency', 0.1)
            ])
            feature_names.extend(['micro_expr_congruence', 'suppression_indicators', 'leakage_frequency'])
        
        # Response timing features
        if 'response_analysis' in behavioral_data:
            response = behavioral_data['response_analysis']
            features.extend([
                response.get('response_latency_consistency', 0.8),
                response.get('processing_time_variability', 0.2),
                response.get('patterned_responses', 0.1)
            ])
            feature_names.extend(['response_latency_consistency', 'processing_variability', 'patterned_responses'])
        
        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)
    
    def classify_authenticity(self, behavioral_data: Dict) -> AuthenticityResult:
        """Classify behavioral authenticity"""
        if not self.is_trained:
            return self._rule_based_authenticity(behavioral_data)
        
        try:
            # Extract features
            features = self.extract_authenticity_features(behavioral_data)
            features_scaled = self.scaler.transform(features)
            
            # Predict authenticity
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]
            
            # Calculate authenticity score
            authenticity_score = self._calculate_authenticity_score(features[0])
            
            # Identify indicators and risk factors
            indicators = self._analyze_authenticity_indicators(features[0])
            risk_factors = self._identify_risk_factors(features[0])
            
            return AuthenticityResult(
                is_authentic=bool(prediction),
                confidence=float(confidence),
                authenticity_score=float(authenticity_score),
                indicators=indicators,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            print(f"Authenticity classification failed: {e}")
            return self._rule_based_authenticity(behavioral_data)
    
    def _calculate_authenticity_score(self, features: np.ndarray) -> float:
        """Calculate overall authenticity score (0-1)"""
        if len(features) != len(self.feature_names):
            return 0.5
        
        feature_dict = dict(zip(self.feature_names, features))
        
        # Weight different aspects of authenticity
        weights = {
            'behavior_consistency': 0.15,
            'emotion_consistency': 0.15,
            'verbal_nonverbal_cong': 0.20,
            'movement_smoothness': 0.10,
            'micro_expr_congruence': 0.15,
            'response_latency_consistency': 0.10,
            'spontaneous_gestures': 0.15
        }
        
        score = 0.0
        total_weight = 0.0
        
        for feature_name, weight in weights.items():
            if feature_name in feature_dict:
                # Some features are inverse indicators
                if feature_name in ['rehearsal_indicators', 'suppression_indicators', 'patterned_responses']:
                    score += (1 - feature_dict[feature_name]) * weight
                else:
                    score += feature_dict[feature_name] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.5
    
    def _analyze_authenticity_indicators(self, features: np.ndarray) -> Dict[str, float]:
        """Analyze individual authenticity indicators"""
        if len(features) != len(self.feature_names):
            return {}
        
        feature_dict = dict(zip(self.feature_names, features))
        indicators = {}
        
        # Map features to psychological indicators
        indicator_mapping = {
            'consistency': ['behavior_consistency', 'emotion_consistency', 'response_consistency'],
            'congruence': ['verbal_nonverbal_cong', 'facial_vocal_cong', 'gesture_speech_sync'],
            'spontaneity': ['movement_smoothness', 'spontaneous_gestures', 'rehearsal_indicators'],
            'micro_expression_congruence': ['micro_expr_congruence', 'suppression_indicators'],
            'response_latency': ['response_latency_consistency', 'processing_variability'],
            'gaze_patterns': [],  # Would need gaze-specific features
            'gesture_synchrony': ['gesture_speech_sync']
        }
        
        for indicator, related_features in indicator_mapping.items():
            relevant_features = [f for f in related_features if f in feature_dict]
            if relevant_features:
                # For inverse indicators, subtract from 1
                values = []
                for feature in relevant_features:
                    if feature in ['rehearsal_indicators', 'suppression_indicators', 'patterned_responses']:
                        values.append(1 - feature_dict[feature])
                    else:
                        values.append(feature_dict[feature])
                
                indicators[indicator] = float(np.mean(values))
        
        return indicators
    
    def _identify_risk_factors(self, features: np.ndarray) -> List[str]:
        """Identify specific risk factors for inauthenticity"""
        if len(features) != len(self.feature_names):
            return []
        
        feature_dict = dict(zip(self.feature_names, features))
        risk_factors = []
        
        risk_thresholds = {
            'behavior_consistency': 0.3,
            'verbal_nonverbal_cong': 0.4,
            'movement_smoothness': 0.3,
            'micro_expr_congruence': 0.4,
            'rehearsal_indicators': 0.7,
            'suppression_indicators': 0.7,
            'patterned_responses': 0.6
        }
        
        for feature, threshold in risk_thresholds.items():
            if feature in feature_dict:
                if feature in ['rehearsal_indicators', 'suppression_indicators', 'patterned_responses']:
                    # High values are risky for these features
                    if feature_dict[feature] > threshold:
                        risk_factors.append(feature)
                else:
                    # Low values are risky for these features
                    if feature_dict[feature] < threshold:
                        risk_factors.append(feature)
        
        return risk_factors
    
    def _rule_based_authenticity(self, behavioral_data: Dict) -> AuthenticityResult:
        """Fallback rule-based authenticity assessment"""
        # Simple heuristic-based approach
        authenticity_score = 0.5
        risk_factors = []
        
        if 'temporal_consistency' in behavioral_data:
            consistency = behavioral_data['temporal_consistency']
            if consistency.get('behavior_consistency', 0.8) < 0.4:
                authenticity_score -= 0.2
                risk_factors.append('low_behavior_consistency')
        
        if 'multimodal_congruence' in behavioral_data:
            congruence = behavioral_data['multimodal_congruence']
            if congruence.get('verbal_nonverbal_congruence', 0.5) < 0.3:
                authenticity_score -= 0.3
                risk_factors.append('low_congruence')
        
        is_authentic = authenticity_score > 0.6
        confidence = min(abs(authenticity_score - 0.5) * 2, 1.0)  # Confidence based on distance from 0.5
        
        return AuthenticityResult(
            is_authentic=is_authentic,
            confidence=confidence,
            authenticity_score=authenticity_score,
            indicators={},
            risk_factors=risk_factors
        )
    
    def train(self, X: List[Dict], y: List[bool]):
        """Train the authenticity classifier"""
        if len(X) != len(y):
            raise ValueError("Features and labels must have the same length")
        
        # Extract features from all samples
        features_list = []
        for data in X:
            features = self.extract_authenticity_features(data)
            features_list.append(features.flatten())
        
        X_array = np.array(features_list)
        y_array = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Train model
        self.model.fit(X_scaled, y_array)
        self.is_trained = True
        
        # Evaluate training performance
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y_array, y_pred)
        
        print(f"Authenticity classifier trained on {len(X)} samples")
        print(f"Training accuracy: {accuracy:.3f}")
        print(classification_report(y_array, y_pred, target_names=['inauthentic', 'authentic']))
    
    def save_model(self, path: str):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, path)
    
    def load_model(self, path: str):
        """Load trained model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        self.feature_names = data['feature_names']
