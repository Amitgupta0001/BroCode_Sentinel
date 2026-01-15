# Behavioral Pattern Learning
# Learns and adapts to user's normal behavior patterns

import json
import os
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class BehavioralPatternLearner:
    """
    Learns user's normal behavior patterns and detects deviations.
    Tracks temporal, interaction, and environmental patterns.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.patterns_file = os.path.join(storage_dir, "behavioral_patterns.json")
        self.patterns = self.load_patterns()
        
    def load_patterns(self):
        """Load stored behavioral patterns"""
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
                return {}
        return {}
    
    def save_patterns(self):
        """Save behavioral patterns to disk"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.patterns_file, 'w') as f:
                json.dump(self.patterns, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def record_behavior(self, user_id, behavior_data):
        """
        Record a behavior event for learning
        
        Args:
            user_id: User identifier
            behavior_data: Dict containing:
                - timestamp: Event time
                - device_hash: Device identifier
                - location: Location data
                - trust_score: Current trust score
                - session_duration: How long session lasted
                - keystroke_count: Number of keystrokes
                - anomaly_detected: Boolean
        """
        if user_id not in self.patterns:
            self.patterns[user_id] = {
                "temporal": {
                    "login_hours": [],
                    "session_durations": [],
                    "active_days": []
                },
                "interaction": {
                    "keystroke_counts": [],
                    "trust_scores": []
                },
                "environmental": {
                    "devices": defaultdict(int),
                    "locations": defaultdict(int)
                },
                "statistics": {
                    "total_sessions": 0,
                    "total_anomalies": 0,
                    "first_seen": time.time(),
                    "last_seen": time.time()
                }
            }
        
        pattern = self.patterns[user_id]
        
        # Update temporal patterns
        dt = datetime.fromtimestamp(behavior_data.get("timestamp", time.time()))
        pattern["temporal"]["login_hours"].append(dt.hour)
        pattern["temporal"]["active_days"].append(dt.weekday())
        
        if "session_duration" in behavior_data:
            pattern["temporal"]["session_durations"].append(behavior_data["session_duration"])
        
        # Update interaction patterns
        if "keystroke_count" in behavior_data:
            pattern["interaction"]["keystroke_counts"].append(behavior_data["keystroke_count"])
        
        if "trust_score" in behavior_data:
            pattern["interaction"]["trust_scores"].append(behavior_data["trust_score"])
        
        # Update environmental patterns
        if "device_hash" in behavior_data:
            device = behavior_data["device_hash"][:16]  # First 16 chars
            if isinstance(pattern["environmental"]["devices"], dict):
                pattern["environmental"]["devices"][device] = pattern["environmental"]["devices"].get(device, 0) + 1
        
        if "location" in behavior_data and behavior_data["location"]:
            location = f"{behavior_data['location'].get('city', 'Unknown')}, {behavior_data['location'].get('country', 'Unknown')}"
            if isinstance(pattern["environmental"]["locations"], dict):
                pattern["environmental"]["locations"][location] = pattern["environmental"]["locations"].get(location, 0) + 1
        
        # Update statistics
        pattern["statistics"]["total_sessions"] += 1
        pattern["statistics"]["last_seen"] = time.time()
        
        if behavior_data.get("anomaly_detected"):
            pattern["statistics"]["total_anomalies"] += 1
        
        # Limit history size (keep last 100 entries)
        for key in pattern["temporal"]:
            if len(pattern["temporal"][key]) > 100:
                pattern["temporal"][key] = pattern["temporal"][key][-100:]
        
        for key in pattern["interaction"]:
            if len(pattern["interaction"][key]) > 100:
                pattern["interaction"][key] = pattern["interaction"][key][-100:]
        
        self.save_patterns()
    
    def get_user_profile(self, user_id):
        """Get learned behavior profile for user"""
        if user_id not in self.patterns:
            return None
        
        pattern = self.patterns[user_id]
        
        # Compute statistics
        profile = {
            "temporal": {
                "typical_login_hours": self._compute_typical_hours(pattern["temporal"]["login_hours"]),
                "typical_days": self._compute_typical_days(pattern["temporal"]["active_days"]),
                "avg_session_duration": np.mean(pattern["temporal"]["session_durations"]) if pattern["temporal"]["session_durations"] else 0
            },
            "interaction": {
                "avg_keystroke_count": np.mean(pattern["interaction"]["keystroke_counts"]) if pattern["interaction"]["keystroke_counts"] else 0,
                "avg_trust_score": np.mean(pattern["interaction"]["trust_scores"]) if pattern["interaction"]["trust_scores"] else 0.5
            },
            "environmental": {
                "primary_devices": self._get_top_items(pattern["environmental"]["devices"], 3),
                "primary_locations": self._get_top_items(pattern["environmental"]["locations"], 3)
            },
            "statistics": pattern["statistics"]
        }
        
        return profile
    
    def predict_normal_behavior(self, user_id, current_behavior):
        """
        Predict if current behavior is normal for this user
        
        Returns:
            score: 0.0 (abnormal) to 1.0 (normal)
            details: Dict with component scores
        """
        profile = self.get_user_profile(user_id)
        
        if not profile or profile["statistics"]["total_sessions"] < 5:
            # Not enough data, assume normal
            return 1.0, {"reason": "insufficient_data"}
        
        scores = {}
        
        # Temporal score
        current_hour = datetime.fromtimestamp(current_behavior.get("timestamp", time.time())).hour
        typical_hours = profile["temporal"]["typical_login_hours"]
        
        if typical_hours:
            hour_score = 1.0 if current_hour in typical_hours else 0.5
        else:
            hour_score = 1.0
        
        scores["temporal"] = hour_score
        
        # Device score
        current_device = current_behavior.get("device_hash", "")[:16]
        primary_devices = profile["environmental"]["primary_devices"]
        
        if primary_devices:
            device_score = 1.0 if current_device in primary_devices else 0.6
        else:
            device_score = 1.0
        
        scores["device"] = device_score
        
        # Location score
        current_location = current_behavior.get("location", {})
        if current_location:
            location_str = f"{current_location.get('city', 'Unknown')}, {current_location.get('country', 'Unknown')}"
            primary_locations = profile["environmental"]["primary_locations"]
            
            if primary_locations:
                location_score = 1.0 if location_str in primary_locations else 0.7
            else:
                location_score = 1.0
        else:
            location_score = 1.0
        
        scores["location"] = location_score
        
        # Trust score deviation
        current_trust = current_behavior.get("trust_score", 0.5)
        avg_trust = profile["interaction"]["avg_trust_score"]
        
        trust_deviation = abs(current_trust - avg_trust)
        trust_score = max(0, 1.0 - trust_deviation)
        
        scores["trust"] = trust_score
        
        # Weighted average
        final_score = (
            scores["temporal"] * 0.2 +
            scores["device"] * 0.3 +
            scores["location"] * 0.2 +
            scores["trust"] * 0.3
        )
        
        return final_score, scores
    
    def _compute_typical_hours(self, hours):
        """Find most common login hours"""
        if not hours:
            return []
        
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        
        # Return hours that appear in top 50% of frequency
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        threshold = sorted_hours[0][1] * 0.5 if sorted_hours else 0
        
        return [hour for hour, count in sorted_hours if count >= threshold]
    
    def _compute_typical_days(self, days):
        """Find most common active days"""
        if not days:
            return []
        
        day_counts = defaultdict(int)
        for day in days:
            day_counts[day] += 1
        
        sorted_days = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)
        threshold = sorted_days[0][1] * 0.5 if sorted_days else 0
        
        return [day for day, count in sorted_days if count >= threshold]
    
    def _get_top_items(self, items_dict, n=3):
        """Get top N items from a frequency dict"""
        if not items_dict:
            return []
        
        sorted_items = sorted(items_dict.items(), key=lambda x: x[1], reverse=True)
        return [item for item, count in sorted_items[:n]]
    
    def get_statistics(self):
        """Get overall statistics"""
        total_users = len(self.patterns)
        total_sessions = sum(p["statistics"]["total_sessions"] for p in self.patterns.values())
        total_anomalies = sum(p["statistics"]["total_anomalies"] for p in self.patterns.values())
        
        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "total_anomalies": total_anomalies,
            "anomaly_rate": total_anomalies / total_sessions if total_sessions > 0 else 0
        }
