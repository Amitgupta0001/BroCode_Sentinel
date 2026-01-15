# Temporal Trust Smoothing Module
# Prevents false logouts from momentary trust drops

import json
import os
import time
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class TemporalTrustSmoother:
    """
    Implements temporal smoothing for trust scores to prevent false positives.
    Uses exponential moving average and rolling windows.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.history_file = os.path.join(storage_dir, "trust_history.json")
        
        # Smoothing parameters
        self.alpha = 0.3  # Weight for current score (0.3 = 30% current, 70% history)
        self.window_size = 10  # Rolling window size
        self.min_samples = 3  # Minimum samples before smoothing
        
        # User trust histories
        self.trust_histories = {}
        
        # Load existing histories
        self.load_histories()
    
    def load_histories(self):
        """Load trust histories from disk"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    # Convert lists back to deques
                    for user_id, history in data.items():
                        self.trust_histories[user_id] = {
                            'scores': deque(history['scores'], maxlen=self.window_size),
                            'timestamps': deque(history['timestamps'], maxlen=self.window_size),
                            'smoothed_score': history.get('smoothed_score', 0.5)
                        }
            except Exception as e:
                logger.error(f"Error loading trust histories: {e}")
    
    def save_histories(self):
        """Save trust histories to disk"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            # Convert deques to lists for JSON serialization
            data = {}
            for user_id, history in self.trust_histories.items():
                data[user_id] = {
                    'scores': list(history['scores']),
                    'timestamps': list(history['timestamps']),
                    'smoothed_score': history['smoothed_score']
                }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trust histories: {e}")
    
    def smooth_trust_score(self, user_id, current_score, timestamp=None):
        """
        Apply temporal smoothing to trust score
        
        Args:
            user_id: User identifier
            current_score: Current raw trust score (0-1)
            timestamp: Optional timestamp (defaults to now)
        
        Returns:
            Dict with smoothed score and metadata
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize history for new users
        if user_id not in self.trust_histories:
            self.trust_histories[user_id] = {
                'scores': deque(maxlen=self.window_size),
                'timestamps': deque(maxlen=self.window_size),
                'smoothed_score': current_score  # First score is unsmoothed
            }
        
        history = self.trust_histories[user_id]
        
        # Add current score to history
        history['scores'].append(current_score)
        history['timestamps'].append(timestamp)
        
        # Calculate smoothed score
        if len(history['scores']) < self.min_samples:
            # Not enough samples, use current score
            smoothed_score = current_score
        else:
            # Exponential moving average
            prev_smoothed = history['smoothed_score']
            smoothed_score = self.alpha * current_score + (1 - self.alpha) * prev_smoothed
        
        # Update smoothed score
        history['smoothed_score'] = smoothed_score
        
        # Calculate additional metrics
        result = {
            'user_id': user_id,
            'raw_score': current_score,
            'smoothed_score': round(smoothed_score, 3),
            'previous_score': history['scores'][-2] if len(history['scores']) >= 2 else current_score,
            'trend': self._calculate_trend(history['scores']),
            'volatility': self._calculate_volatility(history['scores']),
            'confidence': self._calculate_confidence(history['scores']),
            'timestamp': timestamp
        }
        
        # Save periodically
        if len(history['scores']) % 10 == 0:
            self.save_histories()
        
        return result
    
    def _calculate_trend(self, scores):
        """Calculate trust score trend"""
        if len(scores) < 3:
            return 'stable'
        
        recent = list(scores)[-5:]
        
        # Simple linear trend
        increasing = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        decreasing = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        
        if increasing > decreasing + 1:
            return 'improving'
        elif decreasing > increasing + 1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_volatility(self, scores):
        """Calculate trust score volatility (standard deviation)"""
        if len(scores) < 2:
            return 0.0
        
        return round(np.std(list(scores)), 3)
    
    def _calculate_confidence(self, scores):
        """Calculate confidence in the smoothed score"""
        if len(scores) < self.min_samples:
            return 0.5  # Low confidence with few samples
        
        # Confidence based on:
        # 1. Number of samples (more = better)
        # 2. Low volatility (stable = better)
        
        sample_confidence = min(len(scores) / self.window_size, 1.0)
        volatility = self._calculate_volatility(scores)
        stability_confidence = max(0, 1.0 - volatility * 2)  # Lower volatility = higher confidence
        
        confidence = 0.5 * sample_confidence + 0.5 * stability_confidence
        
        return round(confidence, 3)
    
    def get_user_history(self, user_id, limit=None):
        """Get user's trust score history"""
        if user_id not in self.trust_histories:
            return None
        
        history = self.trust_histories[user_id]
        scores = list(history['scores'])
        timestamps = list(history['timestamps'])
        
        if limit:
            scores = scores[-limit:]
            timestamps = timestamps[-limit:]
        
        return {
            'user_id': user_id,
            'scores': scores,
            'timestamps': timestamps,
            'current_smoothed': history['smoothed_score'],
            'trend': self._calculate_trend(history['scores']),
            'volatility': self._calculate_volatility(history['scores'])
        }
    
    def should_trigger_action(self, user_id, smoothed_score, threshold=0.3):
        """
        Determine if action should be triggered based on smoothed score
        
        Uses sustained low trust instead of momentary drops
        """
        if user_id not in self.trust_histories:
            return smoothed_score < threshold
        
        history = self.trust_histories[user_id]
        
        # Check if trust has been consistently low
        if len(history['scores']) >= 3:
            recent_scores = list(history['scores'])[-3:]
            all_low = all(score < threshold for score in recent_scores)
            
            if all_low:
                return True
        
        # Check if smoothed score is significantly below threshold
        return smoothed_score < (threshold - 0.05)
    
    def reset_user_history(self, user_id):
        """Reset user's trust history"""
        if user_id in self.trust_histories:
            del self.trust_histories[user_id]
            self.save_histories()
    
    def get_statistics(self):
        """Get smoothing statistics"""
        total_users = len(self.trust_histories)
        
        avg_volatility = 0
        if total_users > 0:
            volatilities = [
                self._calculate_volatility(h['scores'])
                for h in self.trust_histories.values()
                if len(h['scores']) >= 2
            ]
            avg_volatility = np.mean(volatilities) if volatilities else 0
        
        return {
            'total_users': total_users,
            'average_volatility': round(avg_volatility, 3),
            'window_size': self.window_size,
            'smoothing_alpha': self.alpha,
            'min_samples': self.min_samples
        }
