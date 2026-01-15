# Graduated Response System
# Implements multi-level security responses instead of instant logout

import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ResponseAction(Enum):
    """Possible security responses"""
    ALLOW = "allow"
    MONITOR = "monitor"
    WARN = "warn"
    CHALLENGE = "challenge"
    RESTRICT = "restrict"
    LOGOUT = "logout"

class GraduatedResponseSystem:
    """
    Implements graduated security responses based on threat level.
    Prevents instant logout, uses progressive escalation.
    """
    
    def __init__(self):
        # Trust score thresholds for each threat level
        self.threat_thresholds = {
            ThreatLevel.NONE: 0.8,      # Trust >= 0.8
            ThreatLevel.LOW: 0.6,       # 0.6 <= Trust < 0.8
            ThreatLevel.MEDIUM: 0.4,    # 0.4 <= Trust < 0.6
            ThreatLevel.HIGH: 0.2,      # 0.2 <= Trust < 0.4
            ThreatLevel.CRITICAL: 0.0   # Trust < 0.2
        }
        
        # Response actions for each threat level
        self.threat_responses = {
            ThreatLevel.NONE: ResponseAction.ALLOW,
            ThreatLevel.LOW: ResponseAction.MONITOR,
            ThreatLevel.MEDIUM: ResponseAction.WARN,
            ThreatLevel.HIGH: ResponseAction.CHALLENGE,
            ThreatLevel.CRITICAL: ResponseAction.LOGOUT
        }
        
        # Escalation tracking
        self.user_escalations = {}
        
        # Escalation parameters
        self.escalation_window = 300  # 5 minutes
        self.max_warnings = 3
        self.max_challenges = 2
    
    def evaluate_response(self, user_id, trust_score, context=None):
        """
        Determine appropriate response based on trust score and history
        
        Args:
            user_id: User identifier
            trust_score: Current trust score (0-1)
            context: Additional context (action_type, resource_sensitivity, etc.)
        
        Returns:
            Dict with response action and details
        """
        context = context or {}
        
        # Determine base threat level from trust score
        threat_level = self._calculate_threat_level(trust_score, context)
        
        # Get escalation history
        if user_id not in self.user_escalations:
            self.user_escalations[user_id] = {
                'warnings': 0,
                'challenges': 0,
                'last_warning': 0,
                'last_challenge': 0,
                'escalation_start': time.time()
            }
        
        escalation = self.user_escalations[user_id]
        
        # Check if escalation window has expired
        if time.time() - escalation['escalation_start'] > self.escalation_window:
            # Reset escalation
            self._reset_escalation(user_id)
            escalation = self.user_escalations[user_id]
        
        # Determine response action with escalation
        response_action = self._determine_action(
            threat_level,
            escalation,
            context
        )
        
        # Update escalation state
        self._update_escalation(user_id, response_action)
        
        # Generate response details
        response = self._generate_response(
            user_id,
            trust_score,
            threat_level,
            response_action,
            escalation,
            context
        )
        
        return response
    
    def _calculate_threat_level(self, trust_score, context):
        """Calculate threat level from trust score and context"""
        # Base threat level from trust score
        if trust_score >= self.threat_thresholds[ThreatLevel.NONE]:
            base_level = ThreatLevel.NONE
        elif trust_score >= self.threat_thresholds[ThreatLevel.LOW]:
            base_level = ThreatLevel.LOW
        elif trust_score >= self.threat_thresholds[ThreatLevel.MEDIUM]:
            base_level = ThreatLevel.MEDIUM
        elif trust_score >= self.threat_thresholds[ThreatLevel.HIGH]:
            base_level = ThreatLevel.HIGH
        else:
            base_level = ThreatLevel.CRITICAL
        
        # Adjust based on context
        resource_sensitivity = context.get('resource_sensitivity', 'normal')
        if resource_sensitivity == 'high' and base_level.value < ThreatLevel.CRITICAL.value:
            # Escalate one level for sensitive resources
            base_level = ThreatLevel(base_level.value + 1)
        
        action_type = context.get('action_type', 'read')
        if action_type in ['delete', 'admin', 'financial'] and base_level.value < ThreatLevel.CRITICAL.value:
            # Escalate for risky actions
            base_level = ThreatLevel(base_level.value + 1)
        
        return base_level
    
    def _determine_action(self, threat_level, escalation, context):
        """Determine response action with escalation logic"""
        # Get base action for threat level
        base_action = self.threat_responses[threat_level]
        
        # Apply escalation logic
        if base_action == ResponseAction.WARN:
            if escalation['warnings'] >= self.max_warnings:
                # Escalate to challenge after too many warnings
                return ResponseAction.CHALLENGE
        
        elif base_action == ResponseAction.CHALLENGE:
            if escalation['challenges'] >= self.max_challenges:
                # Escalate to logout after too many challenges
                return ResponseAction.LOGOUT
        
        return base_action
    
    def _update_escalation(self, user_id, action):
        """Update escalation state"""
        escalation = self.user_escalations[user_id]
        current_time = time.time()
        
        if action == ResponseAction.WARN:
            escalation['warnings'] += 1
            escalation['last_warning'] = current_time
        
        elif action == ResponseAction.CHALLENGE:
            escalation['challenges'] += 1
            escalation['last_challenge'] = current_time
        
        elif action == ResponseAction.LOGOUT:
            # Reset on logout
            self._reset_escalation(user_id)
    
    def _reset_escalation(self, user_id):
        """Reset escalation state"""
        self.user_escalations[user_id] = {
            'warnings': 0,
            'challenges': 0,
            'last_warning': 0,
            'last_challenge': 0,
            'escalation_start': time.time()
        }
    
    def _generate_response(self, user_id, trust_score, threat_level, action, escalation, context):
        """Generate detailed response"""
        response = {
            'user_id': user_id,
            'trust_score': trust_score,
            'threat_level': threat_level.name,
            'action': action.value,
            'timestamp': time.time()
        }
        
        # Add action-specific details
        if action == ResponseAction.ALLOW:
            response['message'] = "Access granted - trust level normal"
            response['ui_action'] = None
        
        elif action == ResponseAction.MONITOR:
            response['message'] = "Access granted - increased monitoring"
            response['ui_action'] = "show_monitoring_indicator"
        
        elif action == ResponseAction.WARN:
            response['message'] = f"⚠️ Security Warning: Trust score is {int(trust_score * 100)}%"
            response['ui_action'] = "show_warning_banner"
            response['warning_count'] = escalation['warnings']
            response['max_warnings'] = self.max_warnings
            response['details'] = "Your behavior appears unusual. Please ensure you're in a secure environment."
        
        elif action == ResponseAction.CHALLENGE:
            response['message'] = "🔐 Additional verification required"
            response['ui_action'] = "show_challenge_modal"
            response['challenge_type'] = self._select_challenge_type(context)
            response['challenge_count'] = escalation['challenges']
            response['max_challenges'] = self.max_challenges
            response['details'] = "Please complete additional authentication to continue."
        
        elif action == ResponseAction.LOGOUT:
            response['message'] = "🚨 Session terminated due to security concerns"
            response['ui_action'] = "force_logout"
            response['reason'] = "Sustained low trust score"
            response['details'] = "Your session has been terminated for security reasons. Please log in again."
        
        # Add escalation info
        response['escalation'] = {
            'warnings_issued': escalation['warnings'],
            'challenges_issued': escalation['challenges'],
            'time_in_escalation': int(time.time() - escalation['escalation_start'])
        }
        
        return response
    
    def _select_challenge_type(self, context):
        """Select appropriate challenge type based on context"""
        # Could be: 'liveness', 'voice', 'otp', 'security_question'
        
        if context.get('camera_available'):
            return 'liveness'
        elif context.get('microphone_available'):
            return 'voice'
        else:
            return 'otp'
    
    def handle_challenge_response(self, user_id, challenge_success):
        """
        Handle result of challenge
        
        Args:
            user_id: User identifier
            challenge_success: Boolean indicating if challenge passed
        
        Returns:
            Dict with next action
        """
        if challenge_success:
            # Reset escalation on successful challenge
            self._reset_escalation(user_id)
            
            return {
                'action': 'allow',
                'message': '✅ Challenge passed - access restored',
                'trust_boost': 0.2  # Temporary trust boost
            }
        else:
            # Failed challenge - escalate
            escalation = self.user_escalations[user_id]
            escalation['challenges'] += 1
            
            if escalation['challenges'] >= self.max_challenges:
                return {
                    'action': 'logout',
                    'message': '🚨 Challenge failed - session terminated',
                    'reason': 'Failed authentication challenge'
                }
            else:
                return {
                    'action': 'retry_challenge',
                    'message': '❌ Challenge failed - please try again',
                    'attempts_remaining': self.max_challenges - escalation['challenges']
                }
    
    def get_user_escalation_status(self, user_id):
        """Get current escalation status for user"""
        if user_id not in self.user_escalations:
            return None
        
        escalation = self.user_escalations[user_id]
        
        return {
            'user_id': user_id,
            'warnings': escalation['warnings'],
            'challenges': escalation['challenges'],
            'time_in_escalation': int(time.time() - escalation['escalation_start']),
            'window_remaining': int(self.escalation_window - (time.time() - escalation['escalation_start']))
        }
