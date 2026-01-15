# Context-Aware Trust Module
# Adjusts trust thresholds based on context (time, location, resource sensitivity)

import json
import os
import time
import logging
from datetime import datetime, time as dt_time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ContextAwareTrust:
    """
    Implements context-aware trust scoring that considers:
    - Time of day
    - Day of week
    - Location changes
    - Resource sensitivity
    - User role
    - Historical patterns
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.context_file = os.path.join(storage_dir, "context_profiles.json")
        
        # User context profiles
        self.context_profiles = {}
        
        # Default context rules
        self.default_rules = {
            'time_of_day': {
                'work_hours': {'start': 9, 'end': 17, 'trust_modifier': 0.0},
                'evening': {'start': 17, 'end': 22, 'trust_modifier': -0.05},
                'night': {'start': 22, 'end': 6, 'trust_modifier': -0.15},
                'early_morning': {'start': 6, 'end': 9, 'trust_modifier': -0.05}
            },
            'day_of_week': {
                'weekday': {'trust_modifier': 0.0},
                'weekend': {'trust_modifier': -0.1}
            },
            'resource_sensitivity': {
                'public': {'base_threshold': 0.3, 'trust_modifier': 0.0},
                'normal': {'base_threshold': 0.5, 'trust_modifier': 0.0},
                'sensitive': {'base_threshold': 0.7, 'trust_modifier': -0.1},
                'high_security': {'base_threshold': 0.85, 'trust_modifier': -0.2}
            }
        }
        
        # Load existing profiles
        self.load_profiles()
    
    def load_profiles(self):
        """Load context profiles from disk"""
        if os.path.exists(self.context_file):
            try:
                with open(self.context_file, 'r') as f:
                    self.context_profiles = json.load(f)
            except Exception as e:
                logger.error(f"Error loading context profiles: {e}")
                self.context_profiles = {}
    
    def save_profiles(self):
        """Save context profiles to disk"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.context_file, 'w') as f:
                json.dump(self.context_profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving context profiles: {e}")
    
    def calculate_context_aware_trust(self, user_id, base_trust_score, context):
        """
        Calculate context-aware trust score
        
        Args:
            user_id: User identifier
            base_trust_score: Base trust score (0-1)
            context: Dict with context information:
                - timestamp: Current timestamp
                - location: Current location
                - resource_type: Type of resource being accessed
                - action_type: Type of action (read, write, delete, etc.)
                - user_role: User's role
        
        Returns:
            Dict with adjusted trust score and explanation
        """
        # Initialize user profile if needed
        if user_id not in self.context_profiles:
            self.context_profiles[user_id] = {
                'user_id': user_id,
                'typical_hours': [],
                'typical_locations': [],
                'access_patterns': {},
                'created_at': time.time()
            }
        
        profile = self.context_profiles[user_id]
        
        # Extract context
        timestamp = context.get('timestamp', time.time())
        location = context.get('location')
        resource_type = context.get('resource_type', 'normal')
        action_type = context.get('action_type', 'read')
        user_role = context.get('user_role', 'user')
        
        # Calculate modifiers
        modifiers = {}
        explanations = []
        
        # Time of day modifier
        time_modifier, time_explanation = self._calculate_time_modifier(timestamp, profile)
        modifiers['time'] = time_modifier
        explanations.extend(time_explanation)
        
        # Location modifier
        location_modifier, location_explanation = self._calculate_location_modifier(location, profile)
        modifiers['location'] = location_modifier
        explanations.extend(location_explanation)
        
        # Resource sensitivity modifier
        resource_modifier, resource_explanation = self._calculate_resource_modifier(resource_type)
        modifiers['resource'] = resource_modifier
        explanations.extend(resource_explanation)
        
        # Action type modifier
        action_modifier, action_explanation = self._calculate_action_modifier(action_type)
        modifiers['action'] = action_modifier
        explanations.extend(action_explanation)
        
        # Role modifier
        role_modifier, role_explanation = self._calculate_role_modifier(user_role)
        modifiers['role'] = role_modifier
        explanations.extend(role_explanation)
        
        # Calculate total modifier
        total_modifier = sum(modifiers.values())
        
        # Apply modifier to base trust
        adjusted_trust = base_trust_score + total_modifier
        adjusted_trust = max(0.0, min(1.0, adjusted_trust))
        
        # Determine required threshold based on resource sensitivity
        required_threshold = self.default_rules['resource_sensitivity'][resource_type]['base_threshold']
        
        # Update user profile
        self._update_profile(user_id, timestamp, location, resource_type, action_type)
        
        return {
            'base_trust_score': round(base_trust_score, 3),
            'adjusted_trust_score': round(adjusted_trust, 3),
            'total_modifier': round(total_modifier, 3),
            'modifiers': {k: round(v, 3) for k, v in modifiers.items()},
            'required_threshold': required_threshold,
            'access_allowed': adjusted_trust >= required_threshold,
            'explanations': explanations,
            'context': context
        }
    
    def _calculate_time_modifier(self, timestamp, profile):
        """Calculate time-based trust modifier"""
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        weekday = dt.weekday()  # 0 = Monday, 6 = Sunday
        
        modifier = 0.0
        explanations = []
        
        # Time of day
        if 9 <= hour < 17:
            period = 'work_hours'
        elif 17 <= hour < 22:
            period = 'evening'
        elif 22 <= hour or hour < 6:
            period = 'night'
        else:
            period = 'early_morning'
        
        time_mod = self.default_rules['time_of_day'][period]['trust_modifier']
        modifier += time_mod
        
        if time_mod < 0:
            explanations.append(f"Access during {period.replace('_', ' ')} ({time_mod:+.2f})")
        
        # Day of week
        if weekday >= 5:  # Weekend
            weekend_mod = self.default_rules['day_of_week']['weekend']['trust_modifier']
            modifier += weekend_mod
            explanations.append(f"Weekend access ({weekend_mod:+.2f})")
        
        # Check against user's typical hours
        if profile['typical_hours']:
            if hour not in profile['typical_hours']:
                modifier -= 0.05
                explanations.append("Unusual time for this user (-0.05)")
        
        return modifier, explanations
    
    def _calculate_location_modifier(self, location, profile):
        """Calculate location-based trust modifier"""
        modifier = 0.0
        explanations = []
        
        if not location:
            return modifier, explanations
        
        # Check if location is in typical locations
        if profile['typical_locations']:
            if location not in profile['typical_locations']:
                modifier -= 0.1
                explanations.append("New/unusual location (-0.10)")
        
        # Check for impossible travel (would need previous location and time)
        # Simplified for now
        
        return modifier, explanations
    
    def _calculate_resource_modifier(self, resource_type):
        """Calculate resource sensitivity modifier"""
        if resource_type not in self.default_rules['resource_sensitivity']:
            resource_type = 'normal'
        
        modifier = self.default_rules['resource_sensitivity'][resource_type]['trust_modifier']
        explanations = []
        
        if modifier < 0:
            explanations.append(f"{resource_type.replace('_', ' ').title()} resource ({modifier:+.2f})")
        
        return modifier, explanations
    
    def _calculate_action_modifier(self, action_type):
        """Calculate action type modifier"""
        modifier = 0.0
        explanations = []
        
        # Risky actions require higher trust
        risky_actions = {
            'delete': -0.15,
            'admin': -0.20,
            'financial': -0.25,
            'export': -0.10,
            'modify': -0.05
        }
        
        if action_type in risky_actions:
            modifier = risky_actions[action_type]
            explanations.append(f"{action_type.title()} action ({modifier:+.2f})")
        
        return modifier, explanations
    
    def _calculate_role_modifier(self, user_role):
        """Calculate role-based modifier"""
        modifier = 0.0
        explanations = []
        
        # Higher roles get slight trust boost
        role_modifiers = {
            'admin': +0.05,
            'manager': +0.03,
            'user': 0.0,
            'guest': -0.10
        }
        
        if user_role in role_modifiers:
            modifier = role_modifiers[user_role]
            if modifier != 0:
                explanations.append(f"{user_role.title()} role ({modifier:+.2f})")
        
        return modifier, explanations
    
    def _update_profile(self, user_id, timestamp, location, resource_type, action_type):
        """Update user's context profile"""
        profile = self.context_profiles[user_id]
        
        # Update typical hours
        hour = datetime.fromtimestamp(timestamp).hour
        if hour not in profile['typical_hours']:
            profile['typical_hours'].append(hour)
            # Keep only most common hours (max 12)
            if len(profile['typical_hours']) > 12:
                profile['typical_hours'] = profile['typical_hours'][-12:]
        
        # Update typical locations
        if location and location not in profile['typical_locations']:
            profile['typical_locations'].append(location)
            # Keep only most common locations (max 5)
            if len(profile['typical_locations']) > 5:
                profile['typical_locations'] = profile['typical_locations'][-5:]
        
        # Update access patterns
        pattern_key = f"{resource_type}:{action_type}"
        if pattern_key not in profile['access_patterns']:
            profile['access_patterns'][pattern_key] = 0
        profile['access_patterns'][pattern_key] += 1
        
        # Save periodically
        if sum(profile['access_patterns'].values()) % 10 == 0:
            self.save_profiles()
    
    def get_user_context_profile(self, user_id):
        """Get user's context profile"""
        if user_id not in self.context_profiles:
            return None
        
        profile = self.context_profiles[user_id]
        
        return {
            'user_id': user_id,
            'typical_hours': sorted(profile['typical_hours']),
            'typical_locations': profile['typical_locations'],
            'access_patterns': profile['access_patterns'],
            'profile_age_days': round((time.time() - profile['created_at']) / 86400, 1)
        }
    
    def get_statistics(self):
        """Get context-aware trust statistics"""
        return {
            'total_profiles': len(self.context_profiles),
            'avg_typical_hours': round(
                sum(len(p['typical_hours']) for p in self.context_profiles.values()) / len(self.context_profiles)
                if self.context_profiles else 0,
                1
            ),
            'avg_typical_locations': round(
                sum(len(p['typical_locations']) for p in self.context_profiles.values()) / len(self.context_profiles)
                if self.context_profiles else 0,
                1
            )
        }
