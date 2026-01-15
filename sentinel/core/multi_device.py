# Multi-Device Support Module
# Tracks and manages multiple devices per user with trust levels

import json
import os
import time
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MultiDeviceManager:
    """
    Manages multiple devices per user with:
    - Device registration and tracking
    - Per-device trust levels
    - Cross-device synchronization
    - Risky device isolation
    - Device verification
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.devices_file = os.path.join(storage_dir, "user_devices.json")
        
        # User devices: {user_id: {device_id: device_info}}
        self.user_devices = {}
        
        # Device trust levels
        self.trust_levels = {
            'trusted': 1.0,
            'verified': 0.8,
            'recognized': 0.6,
            'new': 0.4,
            'suspicious': 0.2,
            'blocked': 0.0
        }
        
        # Load existing devices
        self.load_devices()
    
    def load_devices(self):
        """Load device data from disk"""
        if os.path.exists(self.devices_file):
            try:
                with open(self.devices_file, 'r') as f:
                    self.user_devices = json.load(f)
            except Exception as e:
                logger.error(f"Error loading devices: {e}")
                self.user_devices = {}
    
    def save_devices(self):
        """Save device data to disk"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.devices_file, 'w') as f:
                json.dump(self.user_devices, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving devices: {e}")
    
    def register_device(self, user_id, device_fingerprint, device_info=None):
        """
        Register a new device for user
        
        Args:
            user_id: User identifier
            device_fingerprint: Unique device fingerprint
            device_info: Additional device information
        
        Returns:
            Dict with registration result
        """
        # Generate device ID
        device_id = self._generate_device_id(user_id, device_fingerprint)
        
        # Initialize user devices if needed
        if user_id not in self.user_devices:
            self.user_devices[user_id] = {}
        
        # Check if device already exists
        if device_id in self.user_devices[user_id]:
            # Update last seen
            self.user_devices[user_id][device_id]['last_seen'] = time.time()
            self.user_devices[user_id][device_id]['access_count'] += 1
            self.save_devices()
            
            return {
                'device_id': device_id,
                'status': 'existing',
                'trust_level': self.user_devices[user_id][device_id]['trust_level'],
                'message': 'Device already registered'
            }
        
        # Register new device
        device_data = {
            'device_id': device_id,
            'user_id': user_id,
            'fingerprint': device_fingerprint,
            'registered_at': time.time(),
            'last_seen': time.time(),
            'trust_level': 'new',
            'trust_score': self.trust_levels['new'],
            'access_count': 1,
            'verification_status': 'pending',
            'device_info': device_info or {},
            'security_events': [],
            'location_history': []
        }
        
        self.user_devices[user_id][device_id] = device_data
        self.save_devices()
        
        logger.info(f"New device registered for user {user_id}: {device_id}")
        
        return {
            'device_id': device_id,
            'status': 'new',
            'trust_level': 'new',
            'trust_score': self.trust_levels['new'],
            'message': 'New device registered - verification required'
        }
    
    def verify_device(self, user_id, device_id, verification_method='email'):
        """
        Verify a device
        
        Args:
            user_id: User identifier
            device_id: Device identifier
            verification_method: Method used for verification
        
        Returns:
            Boolean indicating success
        """
        if user_id not in self.user_devices or device_id not in self.user_devices[user_id]:
            return False
        
        device = self.user_devices[user_id][device_id]
        
        # Update verification status
        device['verification_status'] = 'verified'
        device['verified_at'] = time.time()
        device['verification_method'] = verification_method
        
        # Upgrade trust level
        device['trust_level'] = 'verified'
        device['trust_score'] = self.trust_levels['verified']
        
        self.save_devices()
        
        logger.info(f"Device verified for user {user_id}: {device_id}")
        
        return True
    
    def get_device_trust(self, user_id, device_id):
        """Get trust level for a device"""
        if user_id not in self.user_devices or device_id not in self.user_devices[user_id]:
            return {
                'trust_level': 'unknown',
                'trust_score': 0.0,
                'reason': 'Device not found'
            }
        
        device = self.user_devices[user_id][device_id]
        
        # Calculate dynamic trust score based on usage
        base_trust = self.trust_levels[device['trust_level']]
        
        # Factors that increase trust
        age_days = (time.time() - device['registered_at']) / 86400
        age_bonus = min(0.1, age_days / 30 * 0.1)  # Up to 0.1 for 30+ days
        
        usage_bonus = min(0.05, device['access_count'] / 100 * 0.05)  # Up to 0.05 for 100+ accesses
        
        # Factors that decrease trust
        security_penalty = len(device['security_events']) * 0.05
        
        # Calculate final trust
        trust_score = base_trust + age_bonus + usage_bonus - security_penalty
        trust_score = max(0.0, min(1.0, trust_score))
        
        return {
            'trust_level': device['trust_level'],
            'trust_score': round(trust_score, 3),
            'base_trust': base_trust,
            'age_bonus': round(age_bonus, 3),
            'usage_bonus': round(usage_bonus, 3),
            'security_penalty': round(security_penalty, 3),
            'access_count': device['access_count'],
            'age_days': round(age_days, 1)
        }
    
    def update_device_trust(self, user_id, device_id, new_level):
        """Update device trust level"""
        if user_id not in self.user_devices or device_id not in self.user_devices[user_id]:
            return False
        
        if new_level not in self.trust_levels:
            return False
        
        device = self.user_devices[user_id][device_id]
        old_level = device['trust_level']
        
        device['trust_level'] = new_level
        device['trust_score'] = self.trust_levels[new_level]
        device['trust_updated_at'] = time.time()
        
        self.save_devices()
        
        logger.info(f"Device trust updated for {user_id}/{device_id}: {old_level} → {new_level}")
        
        return True
    
    def record_security_event(self, user_id, device_id, event_type, details=None):
        """Record a security event for a device"""
        if user_id not in self.user_devices or device_id not in self.user_devices[user_id]:
            return False
        
        device = self.user_devices[user_id][device_id]
        
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details or {}
        }
        
        device['security_events'].append(event)
        
        # Auto-downgrade trust on security events
        if event_type in ['failed_auth', 'suspicious_activity', 'anomaly_detected']:
            if device['trust_level'] == 'trusted':
                self.update_device_trust(user_id, device_id, 'verified')
            elif device['trust_level'] == 'verified':
                self.update_device_trust(user_id, device_id, 'recognized')
            elif device['trust_level'] == 'recognized':
                self.update_device_trust(user_id, device_id, 'suspicious')
        
        self.save_devices()
        
        return True
    
    def get_user_devices(self, user_id):
        """Get all devices for a user"""
        if user_id not in self.user_devices:
            return []
        
        devices = []
        for device_id, device in self.user_devices[user_id].items():
            trust_info = self.get_device_trust(user_id, device_id)
            
            devices.append({
                'device_id': device_id,
                'trust_level': device['trust_level'],
                'trust_score': trust_info['trust_score'],
                'registered_at': device['registered_at'],
                'last_seen': device['last_seen'],
                'access_count': device['access_count'],
                'verification_status': device['verification_status'],
                'device_info': device['device_info'],
                'security_events_count': len(device['security_events'])
            })
        
        # Sort by last seen (most recent first)
        devices.sort(key=lambda d: d['last_seen'], reverse=True)
        
        return devices
    
    def block_device(self, user_id, device_id, reason='manual'):
        """Block a device"""
        if self.update_device_trust(user_id, device_id, 'blocked'):
            self.record_security_event(user_id, device_id, 'device_blocked', {'reason': reason})
            return True
        return False
    
    def unblock_device(self, user_id, device_id):
        """Unblock a device"""
        return self.update_device_trust(user_id, device_id, 'recognized')
    
    def remove_device(self, user_id, device_id):
        """Remove a device"""
        if user_id not in self.user_devices or device_id not in self.user_devices[user_id]:
            return False
        
        del self.user_devices[user_id][device_id]
        self.save_devices()
        
        logger.info(f"Device removed for user {user_id}: {device_id}")
        
        return True
    
    def sync_device_data(self, user_id, device_id, data_type, data):
        """
        Sync data across devices
        
        Args:
            user_id: User identifier
            device_id: Source device
            data_type: Type of data to sync
            data: Data to sync
        """
        # This would implement cross-device synchronization
        # For now, just log the sync request
        logger.info(f"Sync request from {user_id}/{device_id}: {data_type}")
        
        return {
            'synced': True,
            'timestamp': time.time(),
            'devices_synced': len(self.user_devices.get(user_id, {}))
        }
    
    def get_risky_devices(self, user_id):
        """Get list of risky devices for user"""
        if user_id not in self.user_devices:
            return []
        
        risky = []
        for device_id, device in self.user_devices[user_id].items():
            if device['trust_level'] in ['suspicious', 'blocked']:
                risky.append({
                    'device_id': device_id,
                    'trust_level': device['trust_level'],
                    'security_events': len(device['security_events']),
                    'last_seen': device['last_seen']
                })
        
        return risky
    
    def _generate_device_id(self, user_id, fingerprint):
        """Generate unique device ID"""
        combined = f"{user_id}:{fingerprint}:{time.time()}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get_statistics(self):
        """Get multi-device statistics"""
        total_users = len(self.user_devices)
        total_devices = sum(len(devices) for devices in self.user_devices.values())
        
        # Count by trust level
        trust_counts = {level: 0 for level in self.trust_levels.keys()}
        for devices in self.user_devices.values():
            for device in devices.values():
                trust_counts[device['trust_level']] += 1
        
        # Average devices per user
        avg_devices = total_devices / total_users if total_users > 0 else 0
        
        return {
            'total_users': total_users,
            'total_devices': total_devices,
            'avg_devices_per_user': round(avg_devices, 2),
            'trust_distribution': trust_counts,
            'risky_devices': trust_counts.get('suspicious', 0) + trust_counts.get('blocked', 0)
        }
