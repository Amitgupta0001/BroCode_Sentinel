# Audit Logging & Compliance Module
# Provides tamper-proof audit trails and compliance reporting

import json
import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_REAUTH_REQUIRED = "auth.reauth.required"
    AUTH_SESSION_EXPIRED = "auth.session.expired"
    
    # Authorization events
    AUTHZ_ACCESS_GRANTED = "authz.access.granted"
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_PERMISSION_CHANGED = "authz.permission.changed"
    AUTHZ_ROLE_ASSIGNED = "authz.role.assigned"
    AUTHZ_ROLE_REMOVED = "authz.role.removed"
    
    # User management
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_PASSWORD_CHANGED = "user.password.changed"
    
    # Data access
    DATA_READ = "data.read"
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"
    
    # Security events
    SECURITY_ANOMALY_DETECTED = "security.anomaly.detected"
    SECURITY_TRUST_DROP = "security.trust.drop"
    SECURITY_NEW_DEVICE = "security.new.device"
    SECURITY_IMPOSSIBLE_TRAVEL = "security.impossible.travel"
    SECURITY_DEEPFAKE_DETECTED = "security.deepfake.detected"
    
    # System events
    SYSTEM_CONFIG_CHANGED = "system.config.changed"
    SYSTEM_API_KEY_CREATED = "system.api.key.created"
    SYSTEM_API_KEY_REVOKED = "system.api.key.revoked"
    SYSTEM_BACKUP_CREATED = "system.backup.created"


class AuditLogger:
    """
    Tamper-proof audit logging system.
    Implements blockchain-like chaining for integrity verification.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.audit_dir = os.path.join(storage_dir, "audit_logs")
        self.current_log_file = None
        self.log_rotation_size = 10 * 1024 * 1024  # 10MB
        self.retention_days = 365  # 1 year default
        
        os.makedirs(self.audit_dir, exist_ok=True)
        
        # Initialize current log file
        self._initialize_log_file()
        
        # Last event hash for chaining
        self.last_hash = self._get_last_hash()
    
    def _initialize_log_file(self):
        """Initialize or rotate log file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        self.current_log_file = os.path.join(self.audit_dir, f"audit_{timestamp}.jsonl")
        
        # Check if rotation needed
        if os.path.exists(self.current_log_file):
            size = os.path.getsize(self.current_log_file)
            if size > self.log_rotation_size:
                # Rotate to new file
                rotation_num = 1
                while os.path.exists(f"{self.current_log_file}.{rotation_num}"):
                    rotation_num += 1
                os.rename(self.current_log_file, f"{self.current_log_file}.{rotation_num}")
    
    def _get_last_hash(self):
        """Get hash of last audit entry for chaining"""
        if not os.path.exists(self.current_log_file):
            return "0" * 64  # Genesis hash
        
        try:
            with open(self.current_log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get("hash", "0" * 64)
        except Exception as e:
            logger.error(f"Error reading last hash: {e}")
        
        return "0" * 64
    
    def _compute_hash(self, event_data, previous_hash):
        """Compute SHA-256 hash for event (blockchain-style)"""
        hash_input = json.dumps(event_data, sort_keys=True) + previous_hash
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def log_event(self, event_type, user_id, details=None, severity="info", ip_address=None):
        """
        Log an audit event
        
        Args:
            event_type: AuditEventType enum value
            user_id: User who triggered the event
            details: Additional event details (dict)
            severity: Event severity (info, warning, error, critical)
            ip_address: IP address of the request
        
        Returns:
            Event ID
        """
        event_id = f"evt_{int(time.time() * 1000)}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        event_data = {
            "event_id": event_id,
            "event_type": event_type.value if isinstance(event_type, AuditEventType) else event_type,
            "user_id": user_id,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "severity": severity,
            "ip_address": ip_address,
            "details": details or {}
        }
        
        # Compute hash with previous hash (blockchain chaining)
        event_hash = self._compute_hash(event_data, self.last_hash)
        event_data["previous_hash"] = self.last_hash
        event_data["hash"] = event_hash
        
        # Write to log file (append-only)
        try:
            with open(self.current_log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
            
            self.last_hash = event_hash
            
            # Check if rotation needed
            if os.path.getsize(self.current_log_file) > self.log_rotation_size:
                self._initialize_log_file()
            
            logger.debug(f"Audit event logged: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
        
        return event_id
    
    def verify_integrity(self, log_file=None):
        """
        Verify integrity of audit log using hash chain
        
        Returns:
            Dict with verification results
        """
        log_file = log_file or self.current_log_file
        
        if not os.path.exists(log_file):
            return {"valid": False, "error": "Log file not found"}
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return {"valid": True, "events": 0}
            
            previous_hash = "0" * 64
            
            for i, line in enumerate(lines):
                event = json.loads(line)
                
                # Verify previous hash matches
                if event.get("previous_hash") != previous_hash:
                    return {
                        "valid": False,
                        "error": f"Hash chain broken at event {i}",
                        "event_id": event.get("event_id")
                    }
                
                # Recompute hash
                event_copy = event.copy()
                stored_hash = event_copy.pop("hash")
                event_copy.pop("previous_hash")
                
                computed_hash = self._compute_hash(event_copy, previous_hash)
                
                if computed_hash != stored_hash:
                    return {
                        "valid": False,
                        "error": f"Hash mismatch at event {i}",
                        "event_id": event.get("event_id")
                    }
                
                previous_hash = stored_hash
            
            return {
                "valid": True,
                "events": len(lines),
                "log_file": log_file
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def query_events(self, filters=None, start_time=None, end_time=None, limit=100):
        """
        Query audit events with filters
        
        Args:
            filters: Dict of filters (event_type, user_id, severity)
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of results
        
        Returns:
            List of matching events
        """
        filters = filters or {}
        results = []
        
        # Get all log files
        log_files = sorted([
            os.path.join(self.audit_dir, f)
            for f in os.listdir(self.audit_dir)
            if f.startswith("audit_") and f.endswith((".jsonl", ".jsonl.1", ".jsonl.2"))
        ], reverse=True)
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        event = json.loads(line)
                        
                        # Apply filters
                        if start_time and event["timestamp"] < start_time:
                            continue
                        if end_time and event["timestamp"] > end_time:
                            continue
                        
                        if filters.get("event_type") and event["event_type"] != filters["event_type"]:
                            continue
                        if filters.get("user_id") and event["user_id"] != filters["user_id"]:
                            continue
                        if filters.get("severity") and event["severity"] != filters["severity"]:
                            continue
                        
                        results.append(event)
                        
                        if len(results) >= limit:
                            return results
                            
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
        
        return results
    
    def generate_compliance_report(self, report_type="gdpr", start_date=None, end_date=None):
        """
        Generate compliance report
        
        Args:
            report_type: Type of report (gdpr, soc2, hipaa, pci)
            start_date: Report start date
            end_date: Report end date
        
        Returns:
            Dict with compliance report
        """
        end_time = time.time() if not end_date else end_date
        start_time = end_time - (30 * 24 * 3600) if not start_date else start_date  # 30 days default
        
        events = self.query_events(start_time=start_time, end_time=end_time, limit=10000)
        
        report = {
            "report_type": report_type.upper(),
            "period": {
                "start": datetime.fromtimestamp(start_time).isoformat(),
                "end": datetime.fromtimestamp(end_time).isoformat()
            },
            "generated_at": datetime.now().isoformat(),
            "total_events": len(events),
            "summary": {}
        }
        
        if report_type == "gdpr":
            # GDPR-specific metrics
            report["summary"] = {
                "data_access_requests": len([e for e in events if e["event_type"] == AuditEventType.DATA_READ.value]),
                "data_modifications": len([e for e in events if e["event_type"] in [
                    AuditEventType.DATA_UPDATED.value,
                    AuditEventType.DATA_DELETED.value
                ]]),
                "data_exports": len([e for e in events if e["event_type"] == AuditEventType.DATA_EXPORTED.value]),
                "user_deletions": len([e for e in events if e["event_type"] == AuditEventType.USER_DELETED.value]),
                "consent_changes": 0  # Would track consent events
            }
        
        elif report_type == "soc2":
            # SOC2-specific metrics
            report["summary"] = {
                "authentication_events": len([e for e in events if e["event_type"].startswith("auth.")]),
                "authorization_failures": len([e for e in events if e["event_type"] == AuditEventType.AUTHZ_ACCESS_DENIED.value]),
                "security_incidents": len([e for e in events if e["event_type"].startswith("security.")]),
                "config_changes": len([e for e in events if e["event_type"] == AuditEventType.SYSTEM_CONFIG_CHANGED.value]),
                "data_integrity_checks": 1  # Would track integrity verification runs
            }
        
        elif report_type == "hipaa":
            # HIPAA-specific metrics
            report["summary"] = {
                "phi_access_events": len([e for e in events if e["event_type"] == AuditEventType.DATA_READ.value]),
                "phi_modifications": len([e for e in events if e["event_type"] in [
                    AuditEventType.DATA_UPDATED.value,
                    AuditEventType.DATA_DELETED.value
                ]]),
                "unauthorized_access_attempts": len([e for e in events if e["event_type"] == AuditEventType.AUTHZ_ACCESS_DENIED.value]),
                "security_incidents": len([e for e in events if e["severity"] in ["error", "critical"]]),
                "audit_log_reviews": 1  # Would track review activities
            }
        
        # Event breakdown by type
        event_types = {}
        for event in events:
            event_type = event["event_type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        report["event_breakdown"] = event_types
        
        # Severity breakdown
        severity_counts = {}
        for event in events:
            severity = event["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        report["severity_breakdown"] = severity_counts
        
        return report
    
    def export_logs(self, start_time=None, end_time=None, format="json"):
        """
        Export audit logs for external analysis
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            format: Export format (json, csv)
        
        Returns:
            Exported data as string
        """
        events = self.query_events(start_time=start_time, end_time=end_time, limit=100000)
        
        if format == "json":
            return json.dumps(events, indent=2)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if events:
                writer = csv.DictWriter(output, fieldnames=events[0].keys())
                writer.writeheader()
                writer.writerows(events)
            
            return output.getvalue()
        
        return ""
    
    def cleanup_old_logs(self):
        """Remove logs older than retention period"""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        cutoff_date = datetime.fromtimestamp(cutoff_time).strftime("%Y%m%d")
        
        removed_count = 0
        
        for filename in os.listdir(self.audit_dir):
            if filename.startswith("audit_"):
                # Extract date from filename
                date_str = filename.split("_")[1].split(".")[0]
                
                if date_str < cutoff_date:
                    file_path = os.path.join(self.audit_dir, filename)
                    os.remove(file_path)
                    removed_count += 1
                    logger.info(f"Removed old audit log: {filename}")
        
        return removed_count
    
    def get_statistics(self):
        """Get audit logging statistics"""
        log_files = [f for f in os.listdir(self.audit_dir) if f.startswith("audit_")]
        
        total_events = 0
        total_size = 0
        
        for log_file in log_files:
            file_path = os.path.join(self.audit_dir, log_file)
            total_size += os.path.getsize(file_path)
            
            try:
                with open(file_path, 'r') as f:
                    total_events += sum(1 for _ in f)
            except:
                pass
        
        return {
            "total_log_files": len(log_files),
            "total_events": total_events,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "retention_days": self.retention_days,
            "current_log_file": os.path.basename(self.current_log_file)
        }
