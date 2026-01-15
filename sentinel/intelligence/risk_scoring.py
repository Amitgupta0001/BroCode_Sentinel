# Risk Scoring Engine
# Comprehensive risk assessment for authentication and authorization decisions

import json
import os
import time
import logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class RiskScoringEngine:
    """
    Multi-factor risk scoring engine.
    Calculates comprehensive risk scores based on multiple signals.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.risk_profiles_file = os.path.join(storage_dir, "risk_profiles.json")
        self.risk_profiles = self.load_risk_profiles()
        
        # Risk factor weights
        self.weights = {
            "trust_score": 0.25,
            "device_risk": 0.15,
            "location_risk": 0.15,
            "behavioral_risk": 0.15,
            "temporal_risk": 0.10,
            "anomaly_risk": 0.10,
            "historical_risk": 0.10
        }
        
        # Risk thresholds
        self.thresholds = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9
        }
    
    def load_risk_profiles(self):
        """Load user risk profiles"""
        if os.path.exists(self.risk_profiles_file):
            try:
                with open(self.risk_profiles_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading risk profiles: {e}")
                return {}
        return {}
    
    def save_risk_profiles(self):
        """Save risk profiles"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.risk_profiles_file, 'w') as f:
                json.dump(self.risk_profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving risk profiles: {e}")
    
    def calculate_risk_score(self, user_id, context):
        """
        Calculate comprehensive risk score
        
        Args:
            user_id: User identifier
            context: Dict with risk factors:
                - trust_score: Current trust score (0-1)
                - device_known: Boolean
                - device_flagged: Boolean
                - location_known: Boolean
                - impossible_travel: Boolean
                - behavioral_anomaly: Boolean
                - time_unusual: Boolean
                - recent_failures: Integer
                - session_age: Seconds
        
        Returns:
            Dict with risk assessment
        """
        
        # Calculate individual risk components
        components = {
            "trust_score": self._calculate_trust_risk(context.get("trust_score", 0.5)),
            "device_risk": self._calculate_device_risk(context),
            "location_risk": self._calculate_location_risk(context),
            "behavioral_risk": self._calculate_behavioral_risk(context),
            "temporal_risk": self._calculate_temporal_risk(context),
            "anomaly_risk": self._calculate_anomaly_risk(context),
            "historical_risk": self._calculate_historical_risk(user_id)
        }
        
        # Weighted risk score
        total_risk = sum(
            components[factor] * self.weights[factor]
            for factor in components
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(total_risk)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(components, risk_level)
        
        # Store risk assessment
        self._store_risk_assessment(user_id, total_risk, components)
        
        return {
            "user_id": user_id,
            "risk_score": round(total_risk, 3),
            "risk_level": risk_level,
            "components": {k: round(v, 3) for k, v in components.items()},
            "weights": self.weights,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
    
    def _calculate_trust_risk(self, trust_score):
        """Convert trust score to risk score (inverse)"""
        return 1.0 - trust_score
    
    def _calculate_device_risk(self, context):
        """Calculate device-based risk"""
        risk = 0.0
        
        if not context.get("device_known", True):
            risk += 0.6  # Unknown device is high risk
        
        if context.get("device_flagged", False):
            risk += 0.4  # Flagged device adds more risk
        
        return min(1.0, risk)
    
    def _calculate_location_risk(self, context):
        """Calculate location-based risk"""
        risk = 0.0
        
        if not context.get("location_known", True):
            risk += 0.4  # Unknown location is medium risk
        
        if context.get("impossible_travel", False):
            risk += 0.6  # Impossible travel is critical
        
        return min(1.0, risk)
    
    def _calculate_behavioral_risk(self, context):
        """Calculate behavioral anomaly risk"""
        if context.get("behavioral_anomaly", False):
            return 0.7
        return 0.1
    
    def _calculate_temporal_risk(self, context):
        """Calculate time-based risk"""
        risk = 0.0
        
        if context.get("time_unusual", False):
            risk += 0.3
        
        # Session age risk (very old sessions are riskier)
        session_age = context.get("session_age", 0)
        if session_age > 24 * 3600:  # > 24 hours
            risk += 0.4
        elif session_age > 8 * 3600:  # > 8 hours
            risk += 0.2
        
        return min(1.0, risk)
    
    def _calculate_anomaly_risk(self, context):
        """Calculate anomaly-based risk"""
        recent_failures = context.get("recent_failures", 0)
        
        if recent_failures >= 5:
            return 0.9
        elif recent_failures >= 3:
            return 0.6
        elif recent_failures >= 1:
            return 0.3
        
        return 0.0
    
    def _calculate_historical_risk(self, user_id):
        """Calculate risk based on historical behavior"""
        if user_id not in self.risk_profiles:
            return 0.2  # New users have slight risk
        
        profile = self.risk_profiles[user_id]
        
        # Average recent risk scores
        recent_scores = profile.get("recent_scores", [])
        if recent_scores:
            avg_risk = np.mean(recent_scores[-10:])  # Last 10 assessments
            return avg_risk
        
        return 0.2
    
    def _determine_risk_level(self, risk_score):
        """Determine risk level from score"""
        if risk_score >= self.thresholds["critical"]:
            return "critical"
        elif risk_score >= self.thresholds["high"]:
            return "high"
        elif risk_score >= self.thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, components, risk_level):
        """Generate security recommendations based on risk"""
        recommendations = []
        
        if risk_level in ["critical", "high"]:
            recommendations.append("🚨 Require additional authentication")
            recommendations.append("🔒 Limit access to sensitive resources")
        
        if components["device_risk"] > 0.5:
            recommendations.append("📱 Verify device ownership")
        
        if components["location_risk"] > 0.5:
            recommendations.append("🌍 Verify location via secondary channel")
        
        if components["behavioral_risk"] > 0.5:
            recommendations.append("👤 Request behavioral re-verification")
        
        if components["anomaly_risk"] > 0.5:
            recommendations.append("⚠️ Investigate recent failed attempts")
        
        if risk_level == "low":
            recommendations.append("✅ Normal access granted")
        
        return recommendations
    
    def _store_risk_assessment(self, user_id, risk_score, components):
        """Store risk assessment for historical analysis"""
        if user_id not in self.risk_profiles:
            self.risk_profiles[user_id] = {
                "created_at": time.time(),
                "recent_scores": [],
                "total_assessments": 0,
                "high_risk_count": 0
            }
        
        profile = self.risk_profiles[user_id]
        
        # Add to recent scores
        profile["recent_scores"].append(risk_score)
        
        # Keep only last 50 scores
        if len(profile["recent_scores"]) > 50:
            profile["recent_scores"] = profile["recent_scores"][-50:]
        
        profile["total_assessments"] += 1
        profile["last_assessment"] = time.time()
        
        if risk_score >= self.thresholds["high"]:
            profile["high_risk_count"] += 1
        
        self.save_risk_profiles()
    
    def get_user_risk_profile(self, user_id):
        """Get user's risk profile"""
        if user_id not in self.risk_profiles:
            return None
        
        profile = self.risk_profiles[user_id]
        recent_scores = profile.get("recent_scores", [])
        
        return {
            "user_id": user_id,
            "total_assessments": profile.get("total_assessments", 0),
            "high_risk_count": profile.get("high_risk_count", 0),
            "average_risk": round(np.mean(recent_scores), 3) if recent_scores else 0,
            "current_risk": recent_scores[-1] if recent_scores else 0,
            "risk_trend": self._calculate_trend(recent_scores),
            "created_at": profile.get("created_at"),
            "last_assessment": profile.get("last_assessment")
        }
    
    def _calculate_trend(self, scores):
        """Calculate risk trend (improving/stable/worsening)"""
        if len(scores) < 3:
            return "insufficient_data"
        
        recent = scores[-5:]
        
        # Simple linear trend
        increasing = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        decreasing = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        
        if increasing > decreasing + 1:
            return "worsening"
        elif decreasing > increasing + 1:
            return "improving"
        else:
            return "stable"
    
    def get_high_risk_users(self, threshold=0.7, limit=10):
        """Get users with high current risk"""
        high_risk_users = []
        
        for user_id, profile in self.risk_profiles.items():
            recent_scores = profile.get("recent_scores", [])
            if recent_scores and recent_scores[-1] >= threshold:
                high_risk_users.append({
                    "user_id": user_id,
                    "risk_score": recent_scores[-1],
                    "last_assessment": profile.get("last_assessment")
                })
        
        # Sort by risk score descending
        high_risk_users.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return high_risk_users[:limit]
    
    def get_statistics(self):
        """Get risk scoring statistics"""
        total_users = len(self.risk_profiles)
        
        all_scores = []
        high_risk_count = 0
        
        for profile in self.risk_profiles.values():
            recent_scores = profile.get("recent_scores", [])
            if recent_scores:
                all_scores.extend(recent_scores)
                if recent_scores[-1] >= self.thresholds["high"]:
                    high_risk_count += 1
        
        return {
            "total_users": total_users,
            "total_assessments": sum(p.get("total_assessments", 0) for p in self.risk_profiles.values()),
            "high_risk_users": high_risk_count,
            "average_risk": round(np.mean(all_scores), 3) if all_scores else 0,
            "median_risk": round(np.median(all_scores), 3) if all_scores else 0,
            "thresholds": self.thresholds,
            "weights": self.weights
        }
    
    def update_weights(self, new_weights):
        """Update risk factor weights"""
        # Validate weights sum to 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.weights.update(new_weights)
        logger.info(f"Updated risk weights: {self.weights}")
    
    def update_thresholds(self, new_thresholds):
        """Update risk level thresholds"""
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated risk thresholds: {self.thresholds}")
