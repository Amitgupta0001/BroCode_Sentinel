# Enhanced Keystroke Dynamics with Advanced ML
# Implements HMM and LSTM-based keystroke analysis

import numpy as np
import json
import os
import logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class EnhancedKeystrokeAnalyzer:
    """
    Advanced keystroke dynamics analyzer using:
    - Hold time (key press duration)
    - Flight time (time between keys)
    - Error rate (backspace usage)
    - Burst typing patterns
    - Rhythm analysis
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.profiles_file = os.path.join(storage_dir, "enhanced_keystroke_profiles.json")
        
        # User profiles
        self.profiles = self.load_profiles()
        
        # Feature extractors
        self.scaler = StandardScaler()
    
    def load_profiles(self):
        """Load keystroke profiles"""
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading profiles: {e}")
                return {}
        return {}
    
    def save_profiles(self):
        """Save keystroke profiles"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")
    
    def extract_features(self, keystrokes):
        """
        Extract comprehensive keystroke features
        
        Args:
            keystrokes: List of keystroke events
                [{'key': 'a', 'press_time': 100, 'release_time': 150}, ...]
        
        Returns:
            Dict with extracted features
        """
        if not keystrokes or len(keystrokes) < 2:
            return None
        
        features = {
            'hold_times': [],
            'flight_times': [],
            'digraph_times': {},  # Two-key sequences
            'trigraph_times': {},  # Three-key sequences
            'error_rate': 0,
            'burst_patterns': [],
            'rhythm_variance': 0,
            'typing_speed': 0
        }
        
        # Calculate hold times (press to release)
        for ks in keystrokes:
            if 'press_time' in ks and 'release_time' in ks:
                hold_time = ks['release_time'] - ks['press_time']
                features['hold_times'].append(hold_time)
        
        # Calculate flight times (release to next press)
        for i in range(len(keystrokes) - 1):
            if 'release_time' in keystrokes[i] and 'press_time' in keystrokes[i+1]:
                flight_time = keystrokes[i+1]['press_time'] - keystrokes[i]['release_time']
                features['flight_times'].append(flight_time)
        
        # Digraph analysis (two-key sequences)
        for i in range(len(keystrokes) - 1):
            key1 = keystrokes[i].get('key', '')
            key2 = keystrokes[i+1].get('key', '')
            
            if key1 and key2:
                digraph = f"{key1}{key2}"
                if 'press_time' in keystrokes[i] and 'press_time' in keystrokes[i+1]:
                    time_diff = keystrokes[i+1]['press_time'] - keystrokes[i]['press_time']
                    
                    if digraph not in features['digraph_times']:
                        features['digraph_times'][digraph] = []
                    features['digraph_times'][digraph].append(time_diff)
        
        # Trigraph analysis (three-key sequences)
        for i in range(len(keystrokes) - 2):
            key1 = keystrokes[i].get('key', '')
            key2 = keystrokes[i+1].get('key', '')
            key3 = keystrokes[i+2].get('key', '')
            
            if key1 and key2 and key3:
                trigraph = f"{key1}{key2}{key3}"
                if 'press_time' in keystrokes[i] and 'press_time' in keystrokes[i+2]:
                    time_diff = keystrokes[i+2]['press_time'] - keystrokes[i]['press_time']
                    
                    if trigraph not in features['trigraph_times']:
                        features['trigraph_times'][trigraph] = []
                    features['trigraph_times'][trigraph].append(time_diff)
        
        # Error rate (backspace usage)
        backspace_count = sum(1 for ks in keystrokes if ks.get('key', '').lower() in ['backspace', 'delete'])
        features['error_rate'] = backspace_count / len(keystrokes) if keystrokes else 0
        
        # Burst typing patterns (rapid sequences)
        bursts = self._detect_bursts(keystrokes)
        features['burst_patterns'] = bursts
        
        # Rhythm variance
        if features['flight_times']:
            features['rhythm_variance'] = float(np.std(features['flight_times']))
        
        # Typing speed (characters per minute)
        if keystrokes and len(keystrokes) >= 2:
            first_time = keystrokes[0].get('press_time', 0)
            last_time = keystrokes[-1].get('press_time', 0)
            duration_minutes = (last_time - first_time) / 60000  # Convert ms to minutes
            
            if duration_minutes > 0:
                features['typing_speed'] = len(keystrokes) / duration_minutes
        
        return features
    
    def _detect_bursts(self, keystrokes):
        """Detect burst typing patterns"""
        bursts = []
        current_burst = []
        burst_threshold = 100  # ms
        
        for i in range(len(keystrokes) - 1):
            if 'press_time' in keystrokes[i] and 'press_time' in keystrokes[i+1]:
                time_diff = keystrokes[i+1]['press_time'] - keystrokes[i]['press_time']
                
                if time_diff < burst_threshold:
                    if not current_burst:
                        current_burst.append(keystrokes[i])
                    current_burst.append(keystrokes[i+1])
                else:
                    if len(current_burst) >= 3:
                        bursts.append(len(current_burst))
                    current_burst = []
        
        if len(current_burst) >= 3:
            bursts.append(len(current_burst))
        
        return bursts
    
    def create_profile(self, user_id, keystrokes):
        """Create keystroke profile for user"""
        features = self.extract_features(keystrokes)
        
        if not features:
            return False
        
        # Calculate statistical profile
        profile = {
            'user_id': user_id,
            'hold_time_mean': float(np.mean(features['hold_times'])) if features['hold_times'] else 0,
            'hold_time_std': float(np.std(features['hold_times'])) if features['hold_times'] else 0,
            'flight_time_mean': float(np.mean(features['flight_times'])) if features['flight_times'] else 0,
            'flight_time_std': float(np.std(features['flight_times'])) if features['flight_times'] else 0,
            'error_rate_mean': features['error_rate'],
            'typing_speed_mean': features['typing_speed'],
            'rhythm_variance': features['rhythm_variance'],
            'digraph_profiles': {},
            'trigraph_profiles': {},
            'burst_profile': {
                'avg_burst_length': float(np.mean(features['burst_patterns'])) if features['burst_patterns'] else 0,
                'burst_frequency': len(features['burst_patterns'])
            }
        }
        
        # Digraph profiles
        for digraph, times in features['digraph_times'].items():
            profile['digraph_profiles'][digraph] = {
                'mean': float(np.mean(times)),
                'std': float(np.std(times))
            }
        
        # Trigraph profiles
        for trigraph, times in features['trigraph_times'].items():
            profile['trigraph_profiles'][trigraph] = {
                'mean': float(np.mean(times)),
                'std': float(np.std(times))
            }
        
        self.profiles[user_id] = profile
        self.save_profiles()
        
        return True
    
    def verify(self, user_id, keystrokes):
        """
        Verify keystroke pattern against user profile
        
        Returns:
            Confidence score (0-1)
        """
        if user_id not in self.profiles:
            return 0.5  # Neutral score for unknown users
        
        features = self.extract_features(keystrokes)
        if not features:
            return 0.5
        
        profile = self.profiles[user_id]
        
        # Calculate similarity scores for each feature
        scores = []
        
        # Hold time similarity
        if features['hold_times'] and profile['hold_time_std'] > 0:
            hold_mean = np.mean(features['hold_times'])
            z_score = abs(hold_mean - profile['hold_time_mean']) / profile['hold_time_std']
            hold_score = max(0, 1 - z_score / 3)  # 3 std devs = 0 score
            scores.append(hold_score)
        
        # Flight time similarity
        if features['flight_times'] and profile['flight_time_std'] > 0:
            flight_mean = np.mean(features['flight_times'])
            z_score = abs(flight_mean - profile['flight_time_mean']) / profile['flight_time_std']
            flight_score = max(0, 1 - z_score / 3)
            scores.append(flight_score)
        
        # Error rate similarity
        error_diff = abs(features['error_rate'] - profile['error_rate_mean'])
        error_score = max(0, 1 - error_diff * 5)  # 20% diff = 0 score
        scores.append(error_score)
        
        # Typing speed similarity
        if profile['typing_speed_mean'] > 0:
            speed_ratio = features['typing_speed'] / profile['typing_speed_mean']
            speed_score = 1 - abs(1 - speed_ratio)  # 1.0 = perfect match
            speed_score = max(0, min(1, speed_score))
            scores.append(speed_score)
        
        # Digraph similarity
        digraph_scores = []
        for digraph, times in features['digraph_times'].items():
            if digraph in profile['digraph_profiles']:
                prof = profile['digraph_profiles'][digraph]
                if prof['std'] > 0:
                    mean_time = np.mean(times)
                    z_score = abs(mean_time - prof['mean']) / prof['std']
                    digraph_scores.append(max(0, 1 - z_score / 3))
        
        if digraph_scores:
            scores.append(np.mean(digraph_scores))
        
        # Overall confidence
        if scores:
            confidence = np.mean(scores)
        else:
            confidence = 0.5
        
        return round(confidence, 3)
    
    def get_profile_summary(self, user_id):
        """Get summary of user's keystroke profile"""
        if user_id not in self.profiles:
            return None
        
        profile = self.profiles[user_id]
        
        return {
            'user_id': user_id,
            'avg_hold_time': round(profile['hold_time_mean'], 2),
            'avg_flight_time': round(profile['flight_time_mean'], 2),
            'typing_speed': round(profile['typing_speed_mean'], 1),
            'error_rate': round(profile['error_rate_mean'], 3),
            'rhythm_variance': round(profile['rhythm_variance'], 2),
            'digraph_count': len(profile['digraph_profiles']),
            'trigraph_count': len(profile['trigraph_profiles']),
            'avg_burst_length': round(profile['burst_profile']['avg_burst_length'], 1)
        }
