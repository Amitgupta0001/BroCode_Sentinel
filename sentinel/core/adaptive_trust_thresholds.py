# Adaptive Trust Thresholds
# Dynamically adjusts security thresholds based on context

import json
import os
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AdaptiveTrustManager:
    """
    Manages context-aware trust thresholds.
    Adjusts security levels based on user activity, risk level, and context.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.config_file = os.path.join(storage_dir, "trust_thresholds.json")
        self.config = self.load_config()
        
        # Default thresholds
        self.default_thresholds = {
            "critical": 0.3,   # Force logout
            "warning": 0.5,    # Show warning
            "normal": 0.7,     # All good
            "excellent": 0.9   # High trust
        }
        
        # Context-based adjustments
        self.context_modifiers = {
            "high_risk_action": -0.2,      # Stricter for sensitive operations
            "low_risk_action": +0.1,       # More lenient for safe operations
            "new_device": -0.15,           # Stricter on new devices
            "trusted_device": +0.05,       # More lenient on known devices
            "unusual_location": -0.1,      # Stricter in new locations
            "usual_location": +0.05,       # More lenient in known locations
            "unusual_time": -0.05,         # Slightly stricter at unusual times
            "usual_time": +0.05,           # More lenient during usual hours
            "recent_anomaly": -0.15,       # Stricter after recent anomalies
            "clean_history": +0.1          # More lenient with clean history
        }
    
    def load_config(self):
        """Load threshold configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return {}
        return {}
    
    def save_config(self):
        """Save threshold configuration"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_adaptive_thresholds(self, user_id, context):
        """
        Get adaptive thresholds based on context
        
        Args:
            user_id: User identifier
            context: Dict containing:
                - action_type: 'view', 'edit', 'delete', 'admin', 'financial'
                - device_known: Boolean
                - location_known: Boolean
                - time_usual: Boolean
                - recent_anomalies: Integer
                - session_age: Seconds
        
        Returns:
            Dict with adjusted thresholds
        """
        
        # Start with default thresholds
        thresholds = self.default_thresholds.copy()
        
        # Get user-specific overrides if any
        if user_id in self.config:
            user_config = self.config[user_id]
            thresholds.update(user_config.get("thresholds", {}))
        
        # Apply context-based adjustments
        adjustments = []
        total_adjustment = 0
        
        # Action type adjustment
        action_type = context.get("action_type", "view")
        if action_type in ["admin", "financial", "delete"]:
            total_adjustment += self.context_modifiers["high_risk_action"]
            adjustments.append(("high_risk_action", self.context_modifiers["high_risk_action"]))
        elif action_type in ["view", "read"]:
            total_adjustment += self.context_modifiers["low_risk_action"]
            adjustments.append(("low_risk_action", self.context_modifiers["low_risk_action"]))
        
        # Device familiarity
        if not context.get("device_known", True):
            total_adjustment += self.context_modifiers["new_device"]
            adjustments.append(("new_device", self.context_modifiers["new_device"]))
        else:
            total_adjustment += self.context_modifiers["trusted_device"]
            adjustments.append(("trusted_device", self.context_modifiers["trusted_device"]))
        
        # Location familiarity
        if not context.get("location_known", True):
            total_adjustment += self.context_modifiers["unusual_location"]
            adjustments.append(("unusual_location", self.context_modifiers["unusual_location"]))
        else:
            total_adjustment += self.context_modifiers["usual_location"]
            adjustments.append(("usual_location", self.context_modifiers["usual_location"]))
        
        # Time familiarity
        if not context.get("time_usual", True):
            total_adjustment += self.context_modifiers["unusual_time"]
            adjustments.append(("unusual_time", self.context_modifiers["unusual_time"]))
        else:
            total_adjustment += self.context_modifiers["usual_time"]
            adjustments.append(("usual_time", self.context_modifiers["usual_time"]))
        
        # Recent anomalies
        recent_anomalies = context.get("recent_anomalies", 0)
        if recent_anomalies > 0:
            total_adjustment += self.context_modifiers["recent_anomaly"]
            adjustments.append(("recent_anomaly", self.context_modifiers["recent_anomaly"]))
        else:
            total_adjustment += self.context_modifiers["clean_history"]
            adjustments.append(("clean_history", self.context_modifiers["clean_history"]))
        
        # Apply adjustments (with limits)
        adjusted_thresholds = {}
        for key, base_value in thresholds.items():
            adjusted_value = base_value + total_adjustment
            # Clamp between 0.1 and 0.95
            adjusted_value = max(0.1, min(0.95, adjusted_value))
            adjusted_thresholds[key] = round(adjusted_value, 2)
        
        return {
            "thresholds": adjusted_thresholds,
            "base_thresholds": thresholds,
            "adjustments": adjustments,
            "total_adjustment": round(total_adjustment, 2),
            "context": context
        }
    
    def evaluate_trust(self, trust_score, user_id, context):
        """
        Evaluate trust score against adaptive thresholds
        
        Returns:
            Dict with evaluation results
        """
        
        adaptive = self.get_adaptive_thresholds(user_id, context)
        thresholds = adaptive["thresholds"]
        
        # Determine trust level
        if trust_score >= thresholds["excellent"]:
            level = "excellent"
            action = "allow"
            message = "Excellent security posture"
        elif trust_score >= thresholds["normal"]:
            level = "normal"
            action = "allow"
            message = "Normal security level"
        elif trust_score >= thresholds["warning"]:
            level = "warning"
            action = "warn"
            message = "Security warning - re-authentication recommended"
        elif trust_score >= thresholds["critical"]:
            level = "critical"
            action = "challenge"
            message = "Critical security level - additional verification required"
        else:
            level = "blocked"
            action = "deny"
            message = "Access denied - trust score too low"
        
        return {
            "trust_score": trust_score,
            "level": level,
            "action": action,
            "message": message,
            "thresholds": thresholds,
            "base_thresholds": adaptive["base_thresholds"],
            "adjustments": adaptive["adjustments"],
            "total_adjustment": adaptive["total_adjustment"]
        }
    
    def set_user_preferences(self, user_id, preferences):
        """
        Set user-specific threshold preferences
        
        preferences: {
            "security_level": "strict" | "balanced" | "lenient",
            "custom_thresholds": {...}
        }
        """
        
        if user_id not in self.config:
            self.config[user_id] = {}
        
        security_level = preferences.get("security_level", "balanced")
        
        # Preset security levels
        if security_level == "strict":
            self.config[user_id]["thresholds"] = {
                "critical": 0.4,
                "warning": 0.6,
                "normal": 0.8,
                "excellent": 0.95
            }
        elif security_level == "lenient":
            self.config[user_id]["thresholds"] = {
                "critical": 0.2,
                "warning": 0.4,
                "normal": 0.6,
                "excellent": 0.85
            }
        else:  # balanced (default)
            self.config[user_id]["thresholds"] = self.default_thresholds.copy()
        
        # Apply custom thresholds if provided
        if "custom_thresholds" in preferences:
            self.config[user_id]["thresholds"].update(preferences["custom_thresholds"])
        
        self.config[user_id]["security_level"] = security_level
        self.config[user_id]["updated_at"] = time.time()
        
        self.save_config()
        
        logger.info(f"Updated trust thresholds for {user_id}: {security_level}")
    
    def get_user_preferences(self, user_id):
        """Get user's threshold preferences"""
        return self.config.get(user_id, {
            "security_level": "balanced",
            "thresholds": self.default_thresholds
        })
    
    def get_statistics(self):
        """Get threshold statistics"""
        total_users = len(self.config)
        
        security_levels = {"strict": 0, "balanced": 0, "lenient": 0}
        for user_config in self.config.values():
            level = user_config.get("security_level", "balanced")
            security_levels[level] = security_levels.get(level, 0) + 1
        
        return {
            "total_users": total_users,
            "security_levels": security_levels,
            "default_thresholds": self.default_thresholds,
            "context_modifiers": self.context_modifiers
        }
