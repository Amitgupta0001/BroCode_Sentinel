# Use-Case Expansion Module
# Configures BroCode Sentinel for different use cases

import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class UseCaseManager:
    """
    Manages different use-case configurations for BroCode Sentinel.
    Supports: exam proctoring, corporate zero-trust, high-security, insider threat detection.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.config_file = os.path.join(storage_dir, "usecase_config.json")
        
        # Predefined use-case configurations
        self.use_cases = {
            'exam_proctoring': {
                'name': 'Online Exam Proctoring',
                'description': 'Monitor students during online exams',
                'trust_threshold': 0.85,
                'monitoring_interval': 3,  # seconds
                'features': {
                    'liveness_detection': {'enabled': True, 'frequency': 'high'},
                    'face_tracking': {'enabled': True, 'continuous': True},
                    'gaze_tracking': {'enabled': True, 'alert_on_deviation': True},
                    'audio_monitoring': {'enabled': True, 'detect_voices': True},
                    'screen_recording': {'enabled': True},
                    'tab_switching_detection': {'enabled': True, 'max_switches': 3},
                    'copy_paste_blocking': {'enabled': True},
                    'multiple_faces_detection': {'enabled': True, 'max_faces': 1}
                },
                'alerts': {
                    'face_not_detected': 'immediate',
                    'multiple_faces': 'immediate',
                    'tab_switch': 'warning',
                    'audio_detected': 'warning',
                    'gaze_deviation': 'warning'
                },
                'actions': {
                    'on_violation': 'flag_for_review',
                    'on_repeated_violation': 'terminate_exam',
                    'max_warnings': 3
                }
            },
            
            'corporate_zero_trust': {
                'name': 'Corporate Zero-Trust Login',
                'description': 'Continuous authentication for corporate environments',
                'trust_threshold': 0.7,
                'monitoring_interval': 10,
                'features': {
                    'device_fingerprinting': {'enabled': True, 'strict': True},
                    'location_verification': {'enabled': True, 'allowed_locations': []},
                    'time_based_access': {'enabled': True, 'work_hours_only': True},
                    'behavioral_analysis': {'enabled': True, 'adaptive': True},
                    'risk_scoring': {'enabled': True, 'context_aware': True},
                    'session_timeout': {'enabled': True, 'idle_timeout': 900}  # 15 min
                },
                'alerts': {
                    'new_device': 'require_verification',
                    'unusual_location': 'require_mfa',
                    'after_hours_access': 'notify_admin',
                    'trust_drop': 'progressive_challenge'
                },
                'actions': {
                    'on_new_device': 'require_admin_approval',
                    'on_suspicious_activity': 'step_up_auth',
                    'on_critical_action': 'require_mfa'
                }
            },
            
            'high_security_dashboard': {
                'name': 'High-Security Dashboard Access',
                'description': 'Maximum security for sensitive dashboards',
                'trust_threshold': 0.9,
                'monitoring_interval': 5,
                'features': {
                    'multi_factor_auth': {'enabled': True, 'factors': 3},
                    'continuous_liveness': {'enabled': True, 'interval': 30},
                    'keystroke_dynamics': {'enabled': True, 'strict': True},
                    'voice_verification': {'enabled': True, 'periodic': True},
                    'anti_spoofing': {'enabled': True, 'all_methods': True},
                    'session_recording': {'enabled': True, 'full_audit': True},
                    'ip_whitelisting': {'enabled': True},
                    'time_limited_access': {'enabled': True, 'max_duration': 3600}
                },
                'alerts': {
                    'any_anomaly': 'immediate_escalation',
                    'trust_below_threshold': 'force_reauth',
                    'suspicious_action': 'admin_notification'
                },
                'actions': {
                    'on_anomaly': 'immediate_logout',
                    'on_critical_action': 'require_supervisor_approval',
                    'log_everything': True
                }
            },
            
            'insider_threat_detection': {
                'name': 'Insider Threat Detection',
                'description': 'Detect and prevent insider threats',
                'trust_threshold': 0.6,
                'monitoring_interval': 5,
                'features': {
                    'behavioral_profiling': {'enabled': True, 'deep_learning': True},
                    'anomaly_detection': {'enabled': True, 'sensitivity': 'high'},
                    'data_access_monitoring': {'enabled': True, 'track_all': True},
                    'unusual_pattern_detection': {'enabled': True},
                    'peer_comparison': {'enabled': True},
                    'time_pattern_analysis': {'enabled': True},
                    'resource_access_tracking': {'enabled': True}
                },
                'alerts': {
                    'unusual_data_access': 'security_team_alert',
                    'after_hours_activity': 'flag_and_monitor',
                    'bulk_download': 'immediate_block',
                    'privilege_escalation': 'security_incident'
                },
                'actions': {
                    'on_high_risk': 'restrict_access',
                    'on_data_exfiltration_attempt': 'block_and_alert',
                    'on_repeated_anomalies': 'suspend_account'
                }
            },
            
            'standard': {
                'name': 'Standard Authentication',
                'description': 'Balanced security for general use',
                'trust_threshold': 0.5,
                'monitoring_interval': 10,
                'features': {
                    'keystroke_dynamics': {'enabled': True},
                    'face_recognition': {'enabled': True},
                    'device_fingerprinting': {'enabled': True},
                    'behavioral_analysis': {'enabled': True}
                },
                'alerts': {
                    'trust_drop': 'warning',
                    'new_device': 'notification'
                },
                'actions': {
                    'on_low_trust': 'progressive_reauth'
                }
            }
        }
        
        # Current active use case
        self.active_use_case = 'standard'
        
        # Load configuration
        self.load_config()
    
    def load_config(self):
        """Load use-case configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.active_use_case = data.get('active_use_case', 'standard')
            except Exception as e:
                logger.error(f"Error loading use-case config: {e}")
    
    def save_config(self):
        """Save use-case configuration"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump({
                    'active_use_case': self.active_use_case,
                    'last_updated': time.time()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving use-case config: {e}")
    
    def set_use_case(self, use_case_name):
        """
        Set active use case
        
        Args:
            use_case_name: Name of use case to activate
        
        Returns:
            Dict with configuration
        """
        if use_case_name not in self.use_cases:
            return {
                'success': False,
                'message': f'Use case {use_case_name} not found',
                'available_use_cases': list(self.use_cases.keys())
            }
        
        self.active_use_case = use_case_name
        self.save_config()
        
        logger.info(f"Activated use case: {use_case_name}")
        
        return {
            'success': True,
            'use_case': use_case_name,
            'configuration': self.use_cases[use_case_name]
        }
    
    def get_active_config(self):
        """Get active use-case configuration"""
        return {
            'use_case': self.active_use_case,
            'configuration': self.use_cases[self.active_use_case]
        }
    
    def get_all_use_cases(self):
        """Get all available use cases"""
        return {
            use_case: {
                'name': config['name'],
                'description': config['description'],
                'trust_threshold': config['trust_threshold']
            }
            for use_case, config in self.use_cases.items()
        }
    
    def create_custom_use_case(self, use_case_name, configuration):
        """Create a custom use case"""
        self.use_cases[use_case_name] = configuration
        
        logger.info(f"Created custom use case: {use_case_name}")
        
        return {
            'success': True,
            'use_case': use_case_name
        }
    
    def get_feature_config(self, feature_name):
        """Get configuration for a specific feature"""
        config = self.use_cases[self.active_use_case]
        
        if feature_name in config['features']:
            return config['features'][feature_name]
        
        return {'enabled': False}
    
    def should_enable_feature(self, feature_name):
        """Check if a feature should be enabled"""
        feature_config = self.get_feature_config(feature_name)
        return feature_config.get('enabled', False)
    
    def get_trust_threshold(self):
        """Get trust threshold for active use case"""
        return self.use_cases[self.active_use_case]['trust_threshold']
    
    def get_monitoring_interval(self):
        """Get monitoring interval for active use case"""
        return self.use_cases[self.active_use_case]['monitoring_interval']
    
    def get_alert_config(self, alert_type):
        """Get alert configuration"""
        config = self.use_cases[self.active_use_case]
        return config['alerts'].get(alert_type, 'log')
    
    def get_action_config(self, action_type):
        """Get action configuration"""
        config = self.use_cases[self.active_use_case]
        return config['actions'].get(action_type)


# Import time for save_config
import time
