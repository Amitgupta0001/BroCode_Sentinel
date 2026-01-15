import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from sentinel.algorithms.behavioral_patterns.anomaly_detector import BehavioralAnomalyDetector

@dataclass
class SecurityAlert:
    alert_level: str
    alert_type: str
    description: str
    confidence: float
    timestamp: float

class ContinuousBehavioralMonitor:
    """
    Monitors continuous behavioral data and computes trust score trends.
    Integrates anomaly detection to trigger alerts dynamically.
    """
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.sessions = {}
        self.alert_callback = alert_callback
        self.detector = BehavioralAnomalyDetector()
        self.alert_history = deque(maxlen=100)

    def start_session(self, user_id: str):
        self.sessions[user_id] = {
            "start_time": time.time(),
            "trust_history": deque(maxlen=50),
            "current_trust": 0.8,
            "risk": "low"
        }

    def update_behavioral_analysis(self, user_id: str, behavioral_data: Dict):
        if user_id not in self.sessions:
            self.start_session(user_id)
        session = self.sessions[user_id]

        # 1️⃣ Run anomaly detection
        anomaly_result = self.detector.detect(behavioral_data)
        trust_delta = -0.1 if anomaly_result.is_anomaly else 0.05
        session["current_trust"] = np.clip(session["current_trust"] + trust_delta, 0, 1)
        session["trust_history"].append(session["current_trust"])

        # 2️⃣ Assign risk level adaptively
        risk = self._determine_risk(session["current_trust"])
        session["risk"] = risk

        # 3️⃣ Trigger alerts if needed
        if anomaly_result.is_anomaly:
            alert = SecurityAlert(
                alert_level="high" if session["current_trust"] < 0.4 else "medium",
                alert_type=anomaly_result.anomaly_type,
                description=anomaly_result.explanation,
                confidence=anomaly_result.confidence,
                timestamp=time.time()
            )
            self.alert_history.append(alert)
            if self.alert_callback:
                self.alert_callback(user_id, alert)

        return {
            "trust_score": round(session["current_trust"], 3),
            "risk": risk,
            "anomaly": anomaly_result.is_anomaly,
            "anomaly_type": anomaly_result.anomaly_type,
            "explanation": anomaly_result.explanation,
        }

    def _determine_risk(self, trust: float) -> str:
        if trust > 0.7:
            return "low"
        elif trust > 0.5:
            return "medium"
        return "high"
