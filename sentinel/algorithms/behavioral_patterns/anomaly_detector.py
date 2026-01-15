import numpy as np
from typing import List, Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, asdict
import warnings

@dataclass
class AnomalyDetection:
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    confidence: float
    dominant_features: List[str]
    explanation: str

class BehavioralAnomalyDetector:
    """
    Detects anomalies across gaze, pose, facial, and keystroke behavior.
    Uses IsolationForest + rule-based fallback + feature explainability.
    """
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.normal_ranges = {
            "attention_score": (0.3, 0.9),
            "gaze_stability": (0.3, 0.9),
            "movement_smoothness": (0.5, 1.0),
            "emotion_variability": (0.1, 0.8),
            "trust_score": (0.5, 1.0),
        }

    def extract_features(self, data: Dict) -> np.ndarray:
        """Extract core behavioral metrics."""
        features = []
        names = []
        for key in ["attention_score", "gaze_stability", "movement_smoothness",
                    "emotion_variability", "trust_score"]:
            value = data.get(key, 0.5)
            features.append(value)
            names.append(key)
        self.feature_names = names
        return np.array(features).reshape(1, -1)

    def detect(self, data: Dict) -> AnomalyDetection:
        """Run anomaly detection with explainability."""
        try:
            x = self.extract_features(data)
            if not self.is_fitted:
                return self._rule_based(data)

            x_scaled = self.scaler.transform(x)
            prediction = self.model.predict(x_scaled)[0]
            score = -self.model.score_samples(x_scaled)[0]
            anomaly = prediction == -1

            anomaly_type, confidence = self._classify_anomaly_type(data, score)
            explanation = self._generate_explanation(data)

            return AnomalyDetection(
                is_anomaly=bool(anomaly),
                anomaly_score=float(score),
                anomaly_type=anomaly_type,
                confidence=float(confidence),
                dominant_features=self.feature_names,
                explanation=explanation
            )

        except Exception as e:
            warnings.warn(f"ML anomaly detection failed, fallback to rule-based: {e}")
            return self._rule_based(data)

    def _rule_based(self, data: Dict) -> AnomalyDetection:
        score = 0.0
        issues = []
        for k, (low, high) in self.normal_ranges.items():
            v = data.get(k, 0.5)
            if v < low or v > high:
                score += 0.2
                issues.append(k)
        return AnomalyDetection(
            is_anomaly=score > 0.5,
            anomaly_score=score,
            anomaly_type="rule_based",
            confidence=min(1.0, score),
            dominant_features=issues or ["normal"],
            explanation="Rule-based detection triggered" if score > 0.5 else "Normal behavior"
        )

    def _classify_anomaly_type(self, data: Dict, score: float):
        if data.get("attention_score", 0.5) < 0.2:
            return "inattentive", min(1.0, score)
        if data.get("emotion_variability", 0.5) > 0.9:
            return "emotional_instability", min(1.0, score)
        if data.get("movement_smoothness", 1.0) < 0.3:
            return "nervous_motion", min(1.0, score)
        return "behavioral_irregularity", min(1.0, score)

    def _generate_explanation(self, data: Dict):
        sorted_features = sorted(data.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)
        top = [f"{k}: {v:.2f}" for k, v in sorted_features[:3]]
        return "Key deviations → " + ", ".join(top)

    def fit(self, normal_samples: List[Dict]):
        if not normal_samples:
            raise ValueError("No behavioral samples provided.")
        X = np.array([self.extract_features(s).flatten() for s in normal_samples])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        print(f"[INFO] Anomaly detector trained on {len(X)} samples with {X.shape[1]} features.")
