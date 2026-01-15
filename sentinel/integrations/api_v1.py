# API Module for Third-Party Integration
# Provides RESTful API endpoints for external systems

import secrets
import hashlib
import time
import json
import os
import logging
from functools import wraps
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages API keys for third-party integrations"""
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.keys_file = os.path.join(storage_dir, "api_keys.json")
        self.api_keys = self.load_keys()
        self.rate_limits = {}  # Track API usage
    
    def load_keys(self):
        """Load API keys from storage"""
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading API keys: {e}")
                return {}
        return {}
    
    def save_keys(self):
        """Save API keys to storage"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.keys_file, 'w') as f:
                json.dump(self.api_keys, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
    
    def generate_api_key(self, client_name, permissions=None):
        """
        Generate new API key for a client
        
        Args:
            client_name: Name of the client/application
            permissions: List of allowed endpoints/actions
        
        Returns:
            API key string
        """
        # Generate secure random key
        api_key = f"bk_{secrets.token_urlsafe(32)}"
        
        # Hash for storage (never store plain keys)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.api_keys[key_hash] = {
            "client_name": client_name,
            "permissions": permissions or ["read"],
            "created_at": time.time(),
            "last_used": None,
            "usage_count": 0,
            "rate_limit": 1000,  # requests per hour
            "active": True
        }
        
        self.save_keys()
        
        logger.info(f"Generated API key for {client_name}")
        
        # Return the plain key (only time it's visible)
        return api_key
    
    def validate_api_key(self, api_key):
        """
        Validate API key and return client info
        
        Returns:
            Dict with client info or None if invalid
        """
        if not api_key or not api_key.startswith("bk_"):
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash not in self.api_keys:
            return None
        
        key_info = self.api_keys[key_hash]
        
        if not key_info.get("active", True):
            return None
        
        # Check rate limit
        if not self.check_rate_limit(key_hash):
            return None
        
        # Update usage
        key_info["last_used"] = time.time()
        key_info["usage_count"] = key_info.get("usage_count", 0) + 1
        self.save_keys()
        
        return key_info
    
    def check_rate_limit(self, key_hash):
        """Check if API key is within rate limit"""
        current_time = time.time()
        
        if key_hash not in self.rate_limits:
            self.rate_limits[key_hash] = {
                "window_start": current_time,
                "request_count": 0
            }
        
        rate_info = self.rate_limits[key_hash]
        key_info = self.api_keys[key_hash]
        rate_limit = key_info.get("rate_limit", 1000)
        
        # Reset window if hour has passed
        if current_time - rate_info["window_start"] > 3600:
            rate_info["window_start"] = current_time
            rate_info["request_count"] = 0
        
        # Check limit
        if rate_info["request_count"] >= rate_limit:
            logger.warning(f"Rate limit exceeded for {key_info['client_name']}")
            return False
        
        rate_info["request_count"] += 1
        return True
    
    def revoke_api_key(self, api_key):
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            self.api_keys[key_hash]["active"] = False
            self.save_keys()
            logger.info(f"Revoked API key for {self.api_keys[key_hash]['client_name']}")
            return True
        
        return False
    
    def list_api_keys(self):
        """List all API keys (without revealing actual keys)"""
        keys_list = []
        
        for key_hash, info in self.api_keys.items():
            keys_list.append({
                "client_name": info["client_name"],
                "permissions": info["permissions"],
                "created_at": info["created_at"],
                "last_used": info["last_used"],
                "usage_count": info["usage_count"],
                "active": info["active"],
                "key_hash": key_hash[:16] + "..."  # Partial hash for identification
            })
        
        return keys_list


def require_api_key(permissions=None):
    """Decorator to require API key authentication"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            
            if not api_key:
                return jsonify({
                    "error": "API key required",
                    "message": "Include X-API-Key header"
                }), 401
            
            # Get API key manager from app context
            from flask import current_app
            api_manager = current_app.config.get('API_KEY_MANAGER')
            
            if not api_manager:
                return jsonify({"error": "API not configured"}), 500
            
            client_info = api_manager.validate_api_key(api_key)
            
            if not client_info:
                return jsonify({
                    "error": "Invalid API key",
                    "message": "API key is invalid or revoked"
                }), 401
            
            # Check permissions if specified
            if permissions:
                client_permissions = client_info.get("permissions", [])
                if not any(p in client_permissions for p in permissions):
                    return jsonify({
                        "error": "Insufficient permissions",
                        "message": f"Required permissions: {permissions}"
                    }), 403
            
            # Add client info to request context
            request.api_client = client_info
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# Create API Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Public health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time()
    })


@api_bp.route('/auth/register', methods=['POST'])
@require_api_key(permissions=['write', 'admin'])
def api_register_user():
    """
    Register a new user via API
    
    POST /api/v1/auth/register
    Headers: X-API-Key: <api_key>
    Body: {
        "username": "user123",
        "language": "english",
        "keystrokes": [...],
        "email": "user@example.com"
    }
    """
    try:
        data = request.get_json()
        
        username = data.get('username')
        language = data.get('language', 'english')
        keystrokes = data.get('keystrokes', [])
        email = data.get('email')
        
        if not username or not keystrokes:
            return jsonify({
                "error": "Missing required fields",
                "required": ["username", "keystrokes"]
            }), 400
        
        # Here you would integrate with your existing registration logic
        # For now, return success
        
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "username": username,
            "client": request.api_client['client_name']
        }), 201
        
    except Exception as e:
        logger.error(f"API registration error: {e}")
        return jsonify({
            "error": "Registration failed",
            "message": str(e)
        }), 500


@api_bp.route('/auth/verify', methods=['POST'])
@require_api_key(permissions=['read', 'write', 'admin'])
def api_verify_user():
    """
    Verify user authentication via API
    
    POST /api/v1/auth/verify
    Headers: X-API-Key: <api_key>
    Body: {
        "username": "user123",
        "language": "english",
        "keystrokes": [...]
    }
    """
    try:
        data = request.get_json()
        
        username = data.get('username')
        language = data.get('language', 'english')
        keystrokes = data.get('keystrokes', [])
        
        if not username or not keystrokes:
            return jsonify({
                "error": "Missing required fields",
                "required": ["username", "keystrokes"]
            }), 400
        
        # Here you would integrate with your existing auth logic
        # For now, return mock response
        
        return jsonify({
            "verified": True,
            "username": username,
            "trust_score": 0.85,
            "timestamp": time.time()
        }), 200
        
    except Exception as e:
        logger.error(f"API verification error: {e}")
        return jsonify({
            "error": "Verification failed",
            "message": str(e)
        }), 500


@api_bp.route('/users/<username>/trust', methods=['GET'])
@require_api_key(permissions=['read', 'admin'])
def api_get_trust_score(username):
    """
    Get current trust score for a user
    
    GET /api/v1/users/<username>/trust
    Headers: X-API-Key: <api_key>
    """
    try:
        # Here you would get actual trust score
        # For now, return mock data
        
        return jsonify({
            "username": username,
            "trust_score": 0.85,
            "components": {
                "keystroke": 0.8,
                "face": 0.9,
                "behavior": 0.85,
                "liveness": 0.9
            },
            "timestamp": time.time()
        }), 200
        
    except Exception as e:
        logger.error(f"API trust score error: {e}")
        return jsonify({
            "error": "Failed to get trust score",
            "message": str(e)
        }), 500


@api_bp.route('/users/<username>/sessions', methods=['GET'])
@require_api_key(permissions=['read', 'admin'])
def api_get_user_sessions(username):
    """
    Get user's session history
    
    GET /api/v1/users/<username>/sessions
    Headers: X-API-Key: <api_key>
    Query: ?limit=10&offset=0
    """
    try:
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        
        # Here you would get actual session data
        # For now, return mock data
        
        return jsonify({
            "username": username,
            "sessions": [
                {
                    "session_id": "sess_123",
                    "start_time": time.time() - 3600,
                    "end_time": time.time(),
                    "device": "Chrome/Windows",
                    "location": "New York, US",
                    "trust_score": 0.85
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        logger.error(f"API sessions error: {e}")
        return jsonify({
            "error": "Failed to get sessions",
            "message": str(e)
        }), 500


@api_bp.route('/webhooks', methods=['POST'])
@require_api_key(permissions=['admin'])
def api_register_webhook():
    """
    Register a webhook for events
    
    POST /api/v1/webhooks
    Headers: X-API-Key: <api_key>
    Body: {
        "url": "https://example.com/webhook",
        "events": ["trust_drop", "new_device", "anomaly"],
        "secret": "webhook_secret"
    }
    """
    try:
        data = request.get_json()
        
        url = data.get('url')
        events = data.get('events', [])
        secret = data.get('secret')
        
        if not url or not events:
            return jsonify({
                "error": "Missing required fields",
                "required": ["url", "events"]
            }), 400
        
        # Here you would store webhook configuration
        # For now, return success
        
        return jsonify({
            "success": True,
            "webhook_id": f"wh_{secrets.token_urlsafe(16)}",
            "url": url,
            "events": events,
            "created_at": time.time()
        }), 201
        
    except Exception as e:
        logger.error(f"API webhook registration error: {e}")
        return jsonify({
            "error": "Webhook registration failed",
            "message": str(e)
        }), 500


@api_bp.route('/stats', methods=['GET'])
@require_api_key(permissions=['read', 'admin'])
def api_get_stats():
    """
    Get system statistics
    
    GET /api/v1/stats
    Headers: X-API-Key: <api_key>
    """
    try:
        return jsonify({
            "total_users": 100,
            "active_sessions": 15,
            "total_authentications": 5000,
            "average_trust_score": 0.85,
            "anomalies_detected": 12,
            "timestamp": time.time()
        }), 200
        
    except Exception as e:
        logger.error(f"API stats error: {e}")
        return jsonify({
            "error": "Failed to get stats",
            "message": str(e)
        }), 500
