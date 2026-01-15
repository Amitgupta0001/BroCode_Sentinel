# Session Risk Ledger
# Maintains comprehensive per-session risk history and analytics

import json
import os
import time
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class SessionRiskLedger:
    """
    Maintains detailed risk ledger for each session.
    Tracks trust history, anomalies, recovery attempts, and provides analytics.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.ledger_file = os.path.join(storage_dir, "session_risk_ledger.json")
        
        # Active session ledgers
        self.ledgers = {}
        
        # Load existing ledgers
        self.load_ledgers()
    
    def load_ledgers(self):
        """Load session ledgers from disk"""
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, 'r') as f:
                    self.ledgers = json.load(f)
            except Exception as e:
                logger.error(f"Error loading session ledgers: {e}")
                self.ledgers = {}
    
    def save_ledgers(self):
        """Save session ledgers to disk"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.ledger_file, 'w') as f:
                json.dump(self.ledgers, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session ledgers: {e}")
    
    def create_session(self, session_id, user_id, metadata=None):
        """
        Create a new session ledger
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            metadata: Additional session metadata
        """
        self.ledgers[session_id] = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': time.time(),
            'last_updated': time.time(),
            'metadata': metadata or {},
            'trust_history': [],
            'anomalies': [],
            'recovery_attempts': [],
            'escalations': [],
            'challenges': [],
            'statistics': {
                'total_checks': 0,
                'avg_trust_score': 0,
                'min_trust_score': 1.0,
                'max_trust_score': 0.0,
                'anomaly_count': 0,
                'recovery_count': 0,
                'challenge_count': 0,
                'challenge_success_count': 0
            }
        }
        
        self.save_ledgers()
        logger.info(f"Created session ledger for {session_id}")
    
    def record_trust_score(self, session_id, trust_score, components=None, context=None):
        """
        Record a trust score measurement
        
        Args:
            session_id: Session identifier
            trust_score: Trust score value (0-1)
            components: Dict of component scores
            context: Additional context
        """
        if session_id not in self.ledgers:
            logger.warning(f"Session {session_id} not found in ledger")
            return
        
        ledger = self.ledgers[session_id]
        
        # Record trust score
        entry = {
            'timestamp': time.time(),
            'trust_score': trust_score,
            'components': components or {},
            'context': context or {}
        }
        
        ledger['trust_history'].append(entry)
        ledger['last_updated'] = time.time()
        
        # Update statistics
        stats = ledger['statistics']
        stats['total_checks'] += 1
        stats['min_trust_score'] = min(stats['min_trust_score'], trust_score)
        stats['max_trust_score'] = max(stats['max_trust_score'], trust_score)
        
        # Calculate running average
        total_trust = sum(e['trust_score'] for e in ledger['trust_history'])
        stats['avg_trust_score'] = total_trust / len(ledger['trust_history'])
        
        # Keep only last 1000 entries
        if len(ledger['trust_history']) > 1000:
            ledger['trust_history'] = ledger['trust_history'][-1000:]
        
        # Save periodically
        if stats['total_checks'] % 10 == 0:
            self.save_ledgers()
    
    def record_anomaly(self, session_id, anomaly_type, severity, details=None):
        """
        Record a detected anomaly
        
        Args:
            session_id: Session identifier
            anomaly_type: Type of anomaly (e.g., 'trust_drop', 'new_device', etc.)
            severity: Severity level (low, medium, high, critical)
            details: Additional details
        """
        if session_id not in self.ledgers:
            return
        
        ledger = self.ledgers[session_id]
        
        anomaly = {
            'timestamp': time.time(),
            'type': anomaly_type,
            'severity': severity,
            'details': details or {},
            'resolved': False
        }
        
        ledger['anomalies'].append(anomaly)
        ledger['statistics']['anomaly_count'] += 1
        ledger['last_updated'] = time.time()
        
        self.save_ledgers()
        logger.warning(f"Anomaly recorded for session {session_id}: {anomaly_type} ({severity})")
    
    def record_recovery_attempt(self, session_id, recovery_type, success, details=None):
        """
        Record a recovery attempt (e.g., re-authentication, challenge)
        
        Args:
            session_id: Session identifier
            recovery_type: Type of recovery (e.g., 'reauth', 'challenge', 'manual')
            success: Boolean indicating if recovery succeeded
            details: Additional details
        """
        if session_id not in self.ledgers:
            return
        
        ledger = self.ledgers[session_id]
        
        recovery = {
            'timestamp': time.time(),
            'type': recovery_type,
            'success': success,
            'details': details or {}
        }
        
        ledger['recovery_attempts'].append(recovery)
        ledger['statistics']['recovery_count'] += 1
        ledger['last_updated'] = time.time()
        
        self.save_ledgers()
        logger.info(f"Recovery attempt recorded for session {session_id}: {recovery_type} ({'success' if success else 'failed'})")
    
    def record_escalation(self, session_id, from_level, to_level, reason):
        """
        Record a security escalation
        
        Args:
            session_id: Session identifier
            from_level: Previous threat level
            to_level: New threat level
            reason: Reason for escalation
        """
        if session_id not in self.ledgers:
            return
        
        ledger = self.ledgers[session_id]
        
        escalation = {
            'timestamp': time.time(),
            'from_level': from_level,
            'to_level': to_level,
            'reason': reason
        }
        
        ledger['escalations'].append(escalation)
        ledger['last_updated'] = time.time()
        
        self.save_ledgers()
    
    def record_challenge(self, session_id, challenge_type, success, response_time=None):
        """
        Record a security challenge
        
        Args:
            session_id: Session identifier
            challenge_type: Type of challenge (liveness, voice, otp, etc.)
            success: Boolean indicating if challenge passed
            response_time: Time taken to complete challenge (seconds)
        """
        if session_id not in self.ledgers:
            return
        
        ledger = self.ledgers[session_id]
        
        challenge = {
            'timestamp': time.time(),
            'type': challenge_type,
            'success': success,
            'response_time': response_time
        }
        
        ledger['challenges'].append(challenge)
        ledger['statistics']['challenge_count'] += 1
        if success:
            ledger['statistics']['challenge_success_count'] += 1
        ledger['last_updated'] = time.time()
        
        self.save_ledgers()
    
    def get_session_summary(self, session_id):
        """
        Get comprehensive session summary
        
        Returns:
            Dict with session analytics
        """
        if session_id not in self.ledgers:
            return None
        
        ledger = self.ledgers[session_id]
        stats = ledger['statistics']
        
        # Calculate session duration
        duration = time.time() - ledger['created_at']
        
        # Get recent trust scores
        recent_scores = [e['trust_score'] for e in ledger['trust_history'][-10:]]
        
        # Count unresolved anomalies
        unresolved_anomalies = sum(1 for a in ledger['anomalies'] if not a.get('resolved', False))
        
        # Calculate challenge success rate
        challenge_success_rate = 0
        if stats['challenge_count'] > 0:
            challenge_success_rate = stats['challenge_success_count'] / stats['challenge_count']
        
        return {
            'session_id': session_id,
            'user_id': ledger['user_id'],
            'duration_seconds': int(duration),
            'duration_formatted': self._format_duration(duration),
            'statistics': {
                'total_checks': stats['total_checks'],
                'avg_trust_score': round(stats['avg_trust_score'], 3),
                'min_trust_score': round(stats['min_trust_score'], 3),
                'max_trust_score': round(stats['max_trust_score'], 3),
                'current_trust_score': recent_scores[-1] if recent_scores else 0,
                'anomaly_count': stats['anomaly_count'],
                'unresolved_anomalies': unresolved_anomalies,
                'recovery_count': stats['recovery_count'],
                'challenge_count': stats['challenge_count'],
                'challenge_success_rate': round(challenge_success_rate, 2)
            },
            'recent_trust_scores': recent_scores,
            'recent_anomalies': ledger['anomalies'][-5:],
            'recent_escalations': ledger['escalations'][-5:]
        }
    
    def get_risk_assessment(self, session_id):
        """
        Get current risk assessment for session
        
        Returns:
            Dict with risk level and factors
        """
        summary = self.get_session_summary(session_id)
        if not summary:
            return None
        
        stats = summary['statistics']
        
        # Calculate risk score (0-1, higher = more risky)
        risk_factors = []
        risk_score = 0.0
        
        # Low average trust
        if stats['avg_trust_score'] < 0.5:
            risk_score += 0.3
            risk_factors.append("Low average trust score")
        
        # Recent trust drop
        if stats['current_trust_score'] < 0.4:
            risk_score += 0.3
            risk_factors.append("Current trust score critical")
        
        # High anomaly count
        if stats['anomaly_count'] > 5:
            risk_score += 0.2
            risk_factors.append("Multiple anomalies detected")
        
        # Unresolved anomalies
        if stats['unresolved_anomalies'] > 0:
            risk_score += 0.1
            risk_factors.append(f"{stats['unresolved_anomalies']} unresolved anomalies")
        
        # Failed challenges
        if stats['challenge_success_rate'] < 0.5 and stats['challenge_count'] > 0:
            risk_score += 0.2
            risk_factors.append("Low challenge success rate")
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "critical"
        elif risk_score >= 0.5:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'session_id': session_id,
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level):
        """Get recommendation based on risk level"""
        recommendations = {
            'low': "Session appears normal. Continue monitoring.",
            'medium': "Increased vigilance recommended. Monitor closely.",
            'high': "Consider requiring re-authentication or additional verification.",
            'critical': "Immediate action required. Terminate session or force re-authentication."
        }
        return recommendations.get(risk_level, "Unknown risk level")
    
    def _format_duration(self, seconds):
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def close_session(self, session_id, reason="normal"):
        """
        Close a session and finalize ledger
        
        Args:
            session_id: Session identifier
            reason: Reason for closure (normal, timeout, security, etc.)
        """
        if session_id not in self.ledgers:
            return
        
        ledger = self.ledgers[session_id]
        ledger['closed_at'] = time.time()
        ledger['close_reason'] = reason
        ledger['last_updated'] = time.time()
        
        self.save_ledgers()
        logger.info(f"Session {session_id} closed: {reason}")
    
    def get_user_session_history(self, user_id, limit=10):
        """Get session history for a user"""
        user_sessions = [
            ledger for ledger in self.ledgers.values()
            if ledger['user_id'] == user_id
        ]
        
        # Sort by creation time (most recent first)
        user_sessions.sort(key=lambda x: x['created_at'], reverse=True)
        
        return user_sessions[:limit]
