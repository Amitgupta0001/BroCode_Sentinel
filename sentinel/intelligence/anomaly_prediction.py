# Anomaly Prediction System
# Predicts security incidents before they occur using time-series analysis

import json
import os
import time
import logging
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class AnomalyPredictor:
    """
    Predicts future anomalies based on historical patterns.
    Uses simple time-series analysis and pattern recognition.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.history_file = os.path.join(storage_dir, "anomaly_history.json")
        self.predictions_file = os.path.join(storage_dir, "anomaly_predictions.json")
        
        self.history = self.load_history()
        self.predictions = self.load_predictions()
        
        # Prediction parameters
        self.lookback_window = 50  # Number of past events to consider
        self.prediction_horizon = 3600  # Predict 1 hour ahead (seconds)
        self.confidence_threshold = 0.7  # Minimum confidence for alerts
    
    def load_history(self):
        """Load anomaly history"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                return {}
        return {}
    
    def save_history(self):
        """Save anomaly history"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def load_predictions(self):
        """Load predictions"""
        if os.path.exists(self.predictions_file):
            try:
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading predictions: {e}")
                return {}
        return {}
    
    def save_predictions(self):
        """Save predictions"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.predictions_file, 'w') as f:
                json.dump(self.predictions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def record_event(self, user_id, event_type, severity, metadata=None):
        """
        Record a security event
        
        Args:
            user_id: User identifier
            event_type: Type of event (trust_drop, new_device, etc.)
            severity: Event severity (low, medium, high, critical)
            metadata: Additional event data
        """
        if user_id not in self.history:
            self.history[user_id] = {
                "events": [],
                "created_at": time.time()
            }
        
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "severity": severity,
            "metadata": metadata or {}
        }
        
        self.history[user_id]["events"].append(event)
        
        # Keep only recent events
        if len(self.history[user_id]["events"]) > 1000:
            self.history[user_id]["events"] = self.history[user_id]["events"][-1000:]
        
        self.save_history()
        
        # Trigger prediction update
        self.update_predictions(user_id)
    
    def predict_anomaly(self, user_id):
        """
        Predict if anomaly will occur in the near future
        
        Returns:
            Dict with prediction results
        """
        if user_id not in self.history:
            return {
                "prediction": "no_data",
                "confidence": 0,
                "risk_level": "unknown"
            }
        
        events = self.history[user_id]["events"]
        
        if len(events) < 10:
            return {
                "prediction": "insufficient_data",
                "confidence": 0,
                "risk_level": "unknown"
            }
        
        # Analyze recent patterns
        recent_events = events[-self.lookback_window:]
        
        # Calculate event frequency
        time_window = 3600  # 1 hour
        current_time = time.time()
        recent_count = sum(1 for e in recent_events if current_time - e["timestamp"] < time_window)
        
        # Calculate severity trend
        severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        recent_severities = [severity_scores.get(e["severity"], 1) for e in recent_events[-10:]]
        avg_severity = np.mean(recent_severities) if recent_severities else 0
        
        # Detect patterns
        patterns = self._detect_patterns(recent_events)
        
        # Calculate prediction confidence
        confidence = self._calculate_confidence(recent_count, avg_severity, patterns)
        
        # Determine risk level
        if confidence > 0.8:
            risk_level = "critical"
            prediction = "anomaly_imminent"
        elif confidence > 0.6:
            risk_level = "high"
            prediction = "anomaly_likely"
        elif confidence > 0.4:
            risk_level = "medium"
            prediction = "anomaly_possible"
        else:
            risk_level = "low"
            prediction = "normal"
        
        result = {
            "user_id": user_id,
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "risk_level": risk_level,
            "recent_event_count": recent_count,
            "average_severity": round(avg_severity, 2),
            "patterns_detected": patterns,
            "predicted_at": time.time(),
            "prediction_horizon": self.prediction_horizon
        }
        
        return result
    
    def _detect_patterns(self, events):
        """Detect patterns in event sequence"""
        patterns = []
        
        if len(events) < 3:
            return patterns
        
        # Check for escalating severity
        severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        recent_severities = [severity_scores.get(e["severity"], 1) for e in events[-5:]]
        
        if len(recent_severities) >= 3:
            if all(recent_severities[i] <= recent_severities[i+1] for i in range(len(recent_severities)-1)):
                patterns.append("escalating_severity")
        
        # Check for rapid succession
        recent_times = [e["timestamp"] for e in events[-5:]]
        if len(recent_times) >= 3:
            time_diffs = [recent_times[i+1] - recent_times[i] for i in range(len(recent_times)-1)]
            if all(diff < 300 for diff in time_diffs):  # < 5 minutes apart
                patterns.append("rapid_succession")
        
        # Check for repeated event types
        recent_types = [e["event_type"] for e in events[-10:]]
        type_counts = {}
        for t in recent_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        for event_type, count in type_counts.items():
            if count >= 3:
                patterns.append(f"repeated_{event_type}")
        
        return patterns
    
    def _calculate_confidence(self, event_count, avg_severity, patterns):
        """Calculate prediction confidence"""
        confidence = 0.0
        
        # Event frequency contribution
        if event_count >= 10:
            confidence += 0.4
        elif event_count >= 5:
            confidence += 0.2
        elif event_count >= 2:
            confidence += 0.1
        
        # Severity contribution
        if avg_severity >= 3.5:
            confidence += 0.3
        elif avg_severity >= 2.5:
            confidence += 0.2
        elif avg_severity >= 1.5:
            confidence += 0.1
        
        # Pattern contribution
        if "escalating_severity" in patterns:
            confidence += 0.2
        if "rapid_succession" in patterns:
            confidence += 0.15
        if any(p.startswith("repeated_") for p in patterns):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def update_predictions(self, user_id):
        """Update predictions for a user"""
        prediction = self.predict_anomaly(user_id)
        
        if user_id not in self.predictions:
            self.predictions[user_id] = {
                "predictions": [],
                "alerts_sent": 0
            }
        
        self.predictions[user_id]["predictions"].append(prediction)
        
        # Keep only recent predictions
        if len(self.predictions[user_id]["predictions"]) > 100:
            self.predictions[user_id]["predictions"] = self.predictions[user_id]["predictions"][-100:]
        
        self.predictions[user_id]["last_updated"] = time.time()
        
        # Check if alert should be sent
        if prediction["confidence"] >= self.confidence_threshold:
            self.predictions[user_id]["alerts_sent"] += 1
            logger.warning(f"Anomaly prediction alert for {user_id}: {prediction['prediction']} (confidence: {prediction['confidence']})")
        
        self.save_predictions()
        
        return prediction
    
    def get_user_predictions(self, user_id, limit=10):
        """Get recent predictions for a user"""
        if user_id not in self.predictions:
            return []
        
        predictions = self.predictions[user_id]["predictions"]
        return predictions[-limit:]
    
    def get_high_risk_users(self, threshold=0.7, limit=10):
        """Get users with high anomaly risk"""
        high_risk = []
        
        for user_id, data in self.predictions.items():
            predictions = data.get("predictions", [])
            if predictions:
                latest = predictions[-1]
                if latest["confidence"] >= threshold:
                    high_risk.append({
                        "user_id": user_id,
                        "prediction": latest["prediction"],
                        "confidence": latest["confidence"],
                        "risk_level": latest["risk_level"],
                        "predicted_at": latest["predicted_at"]
                    })
        
        # Sort by confidence descending
        high_risk.sort(key=lambda x: x["confidence"], reverse=True)
        
        return high_risk[:limit]
    
    def validate_prediction(self, user_id, prediction_id, actual_occurred):
        """
        Validate a prediction against actual outcome
        
        Args:
            user_id: User identifier
            prediction_id: Prediction timestamp
            actual_occurred: Whether anomaly actually occurred
        """
        if user_id not in self.predictions:
            return
        
        predictions = self.predictions[user_id]["predictions"]
        
        for pred in predictions:
            if abs(pred["predicted_at"] - prediction_id) < 1:
                pred["validated"] = True
                pred["actual_occurred"] = actual_occurred
                pred["validated_at"] = time.time()
                
                # Calculate accuracy
                predicted_anomaly = pred["confidence"] >= self.confidence_threshold
                pred["correct"] = predicted_anomaly == actual_occurred
                
                self.save_predictions()
                break
    
    def get_prediction_accuracy(self):
        """Calculate overall prediction accuracy"""
        total_validated = 0
        correct = 0
        
        for data in self.predictions.values():
            for pred in data.get("predictions", []):
                if pred.get("validated"):
                    total_validated += 1
                    if pred.get("correct"):
                        correct += 1
        
        accuracy = correct / total_validated if total_validated > 0 else 0
        
        return {
            "total_predictions": sum(len(d.get("predictions", [])) for d in self.predictions.values()),
            "validated_predictions": total_validated,
            "correct_predictions": correct,
            "accuracy": round(accuracy, 3)
        }
    
    def get_statistics(self):
        """Get anomaly prediction statistics"""
        total_users = len(self.history)
        total_events = sum(len(h.get("events", [])) for h in self.history.values())
        total_predictions = sum(len(p.get("predictions", [])) for p in self.predictions.values())
        total_alerts = sum(p.get("alerts_sent", 0) for p in self.predictions.values())
        
        accuracy_stats = self.get_prediction_accuracy()
        
        return {
            "total_users": total_users,
            "total_events": total_events,
            "total_predictions": total_predictions,
            "total_alerts": total_alerts,
            "prediction_accuracy": accuracy_stats["accuracy"],
            "validated_predictions": accuracy_stats["validated_predictions"],
            "confidence_threshold": self.confidence_threshold,
            "prediction_horizon_hours": self.prediction_horizon / 3600
        }
