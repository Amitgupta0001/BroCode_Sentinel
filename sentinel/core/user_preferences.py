# User Preferences System
# Allows users to customize their security settings

import json
import os
import time
import logging

logger = logging.getLogger(__name__)

class UserPreferences:
    """
    Manages user-specific preferences for security and UX settings.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.prefs_file = os.path.join(storage_dir, "user_preferences.json")
        self.preferences = self.load_preferences()
        
        # Default preferences
        self.defaults = {
            "security": {
                "level": "balanced",  # strict, balanced, lenient
                "enable_liveness": True,
                "enable_voice": False,
                "enable_deepfake_detection": True,
                "enable_behavioral_learning": True,
                "auto_logout_enabled": True,
                "reauth_grace_period": 30  # seconds
            },
            "notifications": {
                "email_enabled": False,
                "sms_enabled": False,
                "in_app_enabled": True,
                "notify_new_device": True,
                "notify_impossible_travel": True,
                "notify_trust_drop": True,
                "notify_weekly_report": False
            },
            "ui": {
                "theme": "dark",  # dark, light, auto
                "show_trust_score": True,
                "show_explanations": True,
                "show_component_breakdown": True,
                "dashboard_refresh_rate": 5  # seconds
            },
            "privacy": {
                "store_voice_samples": True,
                "store_behavioral_patterns": True,
                "store_location_history": True,
                "data_retention_days": 90
            }
        }
    
    def load_preferences(self):
        """Load user preferences from disk"""
        if os.path.exists(self.prefs_file):
            try:
                with open(self.prefs_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading preferences: {e}")
                return {}
        return {}
    
    def save_preferences(self):
        """Save preferences to disk"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.prefs_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
    
    def get_preferences(self, user_id):
        """Get user preferences with defaults"""
        if user_id not in self.preferences:
            return self.defaults.copy()
        
        # Merge with defaults to ensure all keys exist
        user_prefs = self.defaults.copy()
        self._deep_merge(user_prefs, self.preferences[user_id])
        
        return user_prefs
    
    def update_preferences(self, user_id, updates):
        """
        Update user preferences
        
        updates: Dict with preference updates
        """
        if user_id not in self.preferences:
            self.preferences[user_id] = self.defaults.copy()
        
        # Deep merge updates
        self._deep_merge(self.preferences[user_id], updates)
        
        # Add metadata
        if "metadata" not in self.preferences[user_id]:
            self.preferences[user_id]["metadata"] = {}
        
        self.preferences[user_id]["metadata"]["updated_at"] = time.time()
        self.preferences[user_id]["metadata"]["update_count"] = \
            self.preferences[user_id]["metadata"].get("update_count", 0) + 1
        
        self.save_preferences()
        
        logger.info(f"Updated preferences for user {user_id}")
        
        return self.preferences[user_id]
    
    def _deep_merge(self, base, updates):
        """Deep merge two dictionaries"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get_security_level(self, user_id):
        """Get user's security level"""
        prefs = self.get_preferences(user_id)
        return prefs["security"]["level"]
    
    def set_security_level(self, user_id, level):
        """Set user's security level"""
        if level not in ["strict", "balanced", "lenient"]:
            raise ValueError(f"Invalid security level: {level}")
        
        return self.update_preferences(user_id, {
            "security": {"level": level}
        })
    
    def get_notification_preferences(self, user_id):
        """Get user's notification preferences"""
        prefs = self.get_preferences(user_id)
        return prefs["notifications"]
    
    def update_notification_preferences(self, user_id, notifications):
        """Update user's notification preferences"""
        return self.update_preferences(user_id, {
            "notifications": notifications
        })
    
    def is_feature_enabled(self, user_id, feature):
        """Check if a security feature is enabled for user"""
        prefs = self.get_preferences(user_id)
        feature_map = {
            "liveness": prefs["security"]["enable_liveness"],
            "voice": prefs["security"]["enable_voice"],
            "deepfake": prefs["security"]["enable_deepfake_detection"],
            "behavioral": prefs["security"]["enable_behavioral_learning"],
            "auto_logout": prefs["security"]["auto_logout_enabled"]
        }
        return feature_map.get(feature, True)
    
    def get_ui_preferences(self, user_id):
        """Get user's UI preferences"""
        prefs = self.get_preferences(user_id)
        return prefs["ui"]
    
    def get_privacy_settings(self, user_id):
        """Get user's privacy settings"""
        prefs = self.get_preferences(user_id)
        return prefs["privacy"]
    
    def export_preferences(self, user_id):
        """Export user preferences as JSON"""
        prefs = self.get_preferences(user_id)
        return json.dumps(prefs, indent=2)
    
    def import_preferences(self, user_id, prefs_json):
        """Import user preferences from JSON"""
        try:
            prefs = json.loads(prefs_json)
            return self.update_preferences(user_id, prefs)
        except Exception as e:
            logger.error(f"Error importing preferences: {e}")
            raise ValueError("Invalid preferences JSON")
    
    def reset_to_defaults(self, user_id):
        """Reset user preferences to defaults"""
        self.preferences[user_id] = self.defaults.copy()
        self.preferences[user_id]["metadata"] = {
            "reset_at": time.time(),
            "updated_at": time.time()
        }
        self.save_preferences()
        logger.info(f"Reset preferences to defaults for user {user_id}")
        return self.preferences[user_id]
    
    def get_statistics(self):
        """Get preferences statistics"""
        total_users = len(self.preferences)
        
        security_levels = {"strict": 0, "balanced": 0, "lenient": 0}
        email_enabled = 0
        sms_enabled = 0
        
        for user_prefs in self.preferences.values():
            level = user_prefs.get("security", {}).get("level", "balanced")
            security_levels[level] = security_levels.get(level, 0) + 1
            
            if user_prefs.get("notifications", {}).get("email_enabled"):
                email_enabled += 1
            if user_prefs.get("notifications", {}).get("sms_enabled"):
                sms_enabled += 1
        
        return {
            "total_users": total_users,
            "security_levels": security_levels,
            "email_enabled": email_enabled,
            "sms_enabled": sms_enabled
        }
