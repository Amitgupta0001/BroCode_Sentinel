# Real-Time Streaming with WebSockets
# Replaces HTTP polling with efficient WebSocket connections

from flask_socketio import SocketIO, emit, join_room, leave_room
import logging

logger = logging.getLogger(__name__)

class RealTimeStreaming:
    """
    Manages real-time bidirectional communication using WebSockets.
    Replaces polling with push-based updates.
    """
    
    def __init__(self, app):
        self.socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
        self.active_connections = {}
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid if 'request' in dir() else 'unknown'}")
            emit('connection_established', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected")
            # Clean up user room if needed
            for user_id, sid in list(self.active_connections.items()):
                if sid == request.sid if 'request' in dir() else None:
                    del self.active_connections[user_id]
        
        @self.socketio.on('join_user_room')
        def handle_join_room(data):
            """Join user-specific room for targeted updates"""
            user_id = data.get('user_id')
            if user_id:
                join_room(f"user_{user_id}")
                self.active_connections[user_id] = request.sid if 'request' in dir() else None
                logger.info(f"User {user_id} joined their room")
                emit('room_joined', {'user_id': user_id, 'room': f"user_{user_id}"})
        
        @self.socketio.on('leave_user_room')
        def handle_leave_room(data):
            """Leave user-specific room"""
            user_id = data.get('user_id')
            if user_id:
                leave_room(f"user_{user_id}")
                if user_id in self.active_connections:
                    del self.active_connections[user_id]
                logger.info(f"User {user_id} left their room")
        
        @self.socketio.on('heartbeat')
        def handle_heartbeat():
            """Handle client heartbeat"""
            emit('heartbeat_ack', {'timestamp': time.time()})
    
    def push_trust_update(self, user_id, trust_data):
        """
        Push trust score update to specific user
        
        trust_data: {
            'trust_score': float,
            'risk': str,
            'auth_state': str,
            'components': dict,
            'explanation': list
        }
        """
        self.socketio.emit(
            'trust_update',
            trust_data,
            room=f"user_{user_id}"
        )
        logger.debug(f"Pushed trust update to user {user_id}: {trust_data.get('trust_score')}")
    
    def push_notification(self, user_id, notification):
        """
        Push notification to specific user
        
        notification: {
            'type': str,
            'title': str,
            'message': str,
            'priority': str,
            'timestamp': float
        }
        """
        self.socketio.emit(
            'notification',
            notification,
            room=f"user_{user_id}"
        )
        logger.info(f"Pushed notification to user {user_id}: {notification.get('type')}")
    
    def push_security_alert(self, user_id, alert):
        """
        Push security alert to specific user
        
        alert: {
            'type': 'new_device' | 'impossible_travel' | 'anomaly',
            'severity': 'low' | 'medium' | 'high' | 'critical',
            'message': str,
            'details': dict
        }
        """
        self.socketio.emit(
            'security_alert',
            alert,
            room=f"user_{user_id}"
        )
        logger.warning(f"Pushed security alert to user {user_id}: {alert.get('type')}")
    
    def push_reauth_required(self, user_id, reauth_data):
        """
        Push re-authentication requirement to user
        
        reauth_data: {
            'reason': str,
            'deadline': float,
            'methods': list
        }
        """
        self.socketio.emit(
            'reauth_required',
            reauth_data,
            room=f"user_{user_id}"
        )
        logger.info(f"Pushed re-auth requirement to user {user_id}")
    
    def broadcast_system_message(self, message):
        """Broadcast message to all connected clients"""
        self.socketio.emit('system_message', message)
        logger.info(f"Broadcasted system message: {message.get('type')}")
    
    def get_active_connections(self):
        """Get count of active WebSocket connections"""
        return len(self.active_connections)
    
    def is_user_connected(self, user_id):
        """Check if user has active WebSocket connection"""
        return user_id in self.active_connections
    
    def run(self, app, **kwargs):
        """Run the SocketIO server"""
        self.socketio.run(app, **kwargs)
    
    def get_socketio(self):
        """Get SocketIO instance for integration with Flask app"""
        return self.socketio
