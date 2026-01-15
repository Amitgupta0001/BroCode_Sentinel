# Notification System
# Handles email and SMS alerts for security events

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class NotificationService:
    """
    Centralized notification service for security alerts.
    Supports email, SMS, and in-app notifications.
    """
    
    def __init__(self):
        self.email_enabled = False
        self.sms_enabled = False
        self.notification_history = []
        
        # Check for email configuration
        self.sendgrid_api_key = os.environ.get("SENDGRID_API_KEY")
        self.from_email = os.environ.get("NOTIFICATION_FROM_EMAIL", "security@brocode.ai")
        
        # Check for SMS configuration
        self.twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_phone = os.environ.get("TWILIO_PHONE_NUMBER")
        
        if self.sendgrid_api_key:
            self.email_enabled = True
            logger.info("Email notifications enabled (SendGrid)")
        else:
            logger.warning("Email notifications disabled (no SENDGRID_API_KEY)")
            
        if all([self.twilio_account_sid, self.twilio_auth_token, self.twilio_phone]):
            self.sms_enabled = True
            logger.info("SMS notifications enabled (Twilio)")
        else:
            logger.warning("SMS notifications disabled (missing Twilio credentials)")
    
    def send_new_device_alert(self, username: str, device_info: Dict, user_email: Optional[str] = None):
        """Alert user about new device login"""
        subject = "🔐 New Device Detected - BroCode Sentinel"
        
        device_name = device_info.get("device_name", "Unknown Device")
        timestamp = datetime.fromtimestamp(device_info.get("registered_at", 0)).strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
        Hello {username},
        
        A new device was detected logging into your BroCode Sentinel account:
        
        Device: {device_name}
        Time: {timestamp}
        Device ID: {device_info.get("hash", "unknown")[:16]}...
        
        If this was you, no action is needed. If you don't recognize this device,
        please secure your account immediately.
        
        - BroCode Sentinel Security Team
        """
        
        self._send_notification(
            username=username,
            subject=subject,
            message=message,
            priority="high",
            notification_type="new_device",
            user_email=user_email
        )
    
    def send_impossible_travel_alert(self, username: str, travel_info: Dict, user_email: Optional[str] = None):
        """Alert user about impossible travel detection"""
        subject = "🚨 SECURITY ALERT: Impossible Travel Detected"
        
        distance = travel_info.get("distance_km", 0)
        time_hours = travel_info.get("time_hours", 0)
        from_location = travel_info.get("from_location", "Unknown")
        to_location = travel_info.get("to_location", "Unknown")
        
        message = f"""
        URGENT SECURITY ALERT
        
        Hello {username},
        
        We detected impossible travel on your account:
        
        From: {from_location}
        To: {to_location}
        Distance: {distance} km
        Time: {time_hours} hours
        
        This indicates potential unauthorized access to your account.
        
        IMMEDIATE ACTION REQUIRED:
        1. Change your password
        2. Review trusted devices
        3. Enable two-factor authentication
        
        - BroCode Sentinel Security Team
        """
        
        self._send_notification(
            username=username,
            subject=subject,
            message=message,
            priority="critical",
            notification_type="impossible_travel",
            user_email=user_email
        )
    
    def send_trust_drop_alert(self, username: str, trust_score: float, user_email: Optional[str] = None):
        """Alert user about significant trust score drop"""
        subject = "⚠️ Trust Score Alert - BroCode Sentinel"
        
        message = f"""
        Hello {username},
        
        Your trust score has dropped significantly:
        
        Current Score: {trust_score:.2f}
        Threshold: 0.50
        
        This may indicate:
        - Unusual behavior patterns
        - Different typing rhythm
        - Face not detected
        - Environmental changes
        
        If this is you, the system will adapt. If not, your session may be terminated.
        
        - BroCode Sentinel Security Team
        """
        
        self._send_notification(
            username=username,
            subject=subject,
            message=message,
            priority="medium",
            notification_type="trust_drop",
            user_email=user_email
        )
    
    def send_weekly_report(self, username: str, report_data: Dict, user_email: Optional[str] = None):
        """Send weekly security summary"""
        subject = "📊 Weekly Security Report - BroCode Sentinel"
        
        total_logins = report_data.get("total_logins", 0)
        new_devices = report_data.get("new_devices", 0)
        anomalies = report_data.get("anomalies", 0)
        avg_trust = report_data.get("avg_trust_score", 0)
        
        message = f"""
        Hello {username},
        
        Your weekly security summary:
        
        📊 Statistics:
        - Total Logins: {total_logins}
        - New Devices: {new_devices}
        - Anomalies Detected: {anomalies}
        - Average Trust Score: {avg_trust:.2f}
        
        🔐 Security Status: {"✅ Good" if anomalies == 0 else "⚠️ Review Needed"}
        
        Stay secure!
        - BroCode Sentinel Security Team
        """
        
        self._send_notification(
            username=username,
            subject=subject,
            message=message,
            priority="low",
            notification_type="weekly_report",
            user_email=user_email
        )
    
    def _send_notification(self, username: str, subject: str, message: str, 
                          priority: str, notification_type: str, user_email: Optional[str] = None):
        """Internal method to send notifications via all enabled channels"""
        
        notification = {
            "username": username,
            "subject": subject,
            "message": message,
            "priority": priority,
            "type": notification_type,
            "timestamp": datetime.now().isoformat(),
            "sent_via": []
        }
        
        # Send email if enabled and user email provided
        if self.email_enabled and user_email:
            success = self._send_email(user_email, subject, message)
            if success:
                notification["sent_via"].append("email")
        
        # Send SMS for critical alerts if enabled
        if self.sms_enabled and priority == "critical":
            # In production, you'd get user's phone number from database
            # For now, we'll just log it
            logger.info(f"SMS would be sent for critical alert: {subject}")
            notification["sent_via"].append("sms_simulated")
        
        # Always log the notification
        logger.info(f"Notification sent to {username}: {notification_type} ({priority})")
        
        # Store in history
        self.notification_history.append(notification)
        
        # Keep only last 100 notifications
        if len(self.notification_history) > 100:
            self.notification_history = self.notification_history[-100:]
        
        return notification
    
    def _send_email(self, to_email: str, subject: str, message: str) -> bool:
        """Send email via SendGrid"""
        if not self.email_enabled:
            logger.warning("Email not sent: SendGrid not configured")
            return False
        
        try:
            # Import SendGrid only if configured
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            
            email_message = Mail(
                from_email=self.from_email,
                to_emails=to_email,
                subject=subject,
                plain_text_content=message
            )
            
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(email_message)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Email sent successfully to {to_email}")
                return True
            else:
                logger.error(f"Email failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Email error: {e}")
            return False
    
    def _send_sms(self, to_phone: str, message: str) -> bool:
        """Send SMS via Twilio"""
        if not self.sms_enabled:
            logger.warning("SMS not sent: Twilio not configured")
            return False
        
        try:
            # Import Twilio only if configured
            from twilio.rest import Client
            
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            sms_message = client.messages.create(
                body=message,
                from_=self.twilio_phone,
                to=to_phone
            )
            
            logger.info(f"SMS sent successfully to {to_phone}: {sms_message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"SMS error: {e}")
            return False
    
    def get_notification_history(self, username: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get notification history for a user"""
        if username:
            user_notifications = [n for n in self.notification_history if n["username"] == username]
            return user_notifications[-limit:]
        return self.notification_history[-limit:]
    
    def get_stats(self) -> Dict:
        """Get notification system statistics"""
        return {
            "email_enabled": self.email_enabled,
            "sms_enabled": self.sms_enabled,
            "total_notifications": len(self.notification_history),
            "notification_types": {
                "new_device": len([n for n in self.notification_history if n["type"] == "new_device"]),
                "impossible_travel": len([n for n in self.notification_history if n["type"] == "impossible_travel"]),
                "trust_drop": len([n for n in self.notification_history if n["type"] == "trust_drop"]),
                "weekly_report": len([n for n in self.notification_history if n["type"] == "weekly_report"])
            }
        }
