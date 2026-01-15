# Role-Based Access Control (RBAC) Module
# Manages user roles and permissions

import json
import os
import time
import logging
from functools import wraps
from flask import session, jsonify, redirect, url_for, flash

logger = logging.getLogger(__name__)

class RBACManager:
    """
    Role-Based Access Control system.
    Manages roles, permissions, and access control.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.roles_file = os.path.join(storage_dir, "rbac_roles.json")
        self.user_roles_file = os.path.join(storage_dir, "rbac_user_roles.json")
        
        # Define default roles and permissions
        self.default_roles = {
            "admin": {
                "name": "Administrator",
                "description": "Full system access",
                "permissions": [
                    "user.create", "user.read", "user.update", "user.delete",
                    "role.create", "role.read", "role.update", "role.delete",
                    "system.configure", "system.monitor", "system.audit",
                    "api.manage", "data.export", "data.delete"
                ],
                "priority": 100
            },
            "manager": {
                "name": "Manager",
                "description": "Manage users and view reports",
                "permissions": [
                    "user.create", "user.read", "user.update",
                    "role.read", "system.monitor",
                    "data.export"
                ],
                "priority": 50
            },
            "user": {
                "name": "User",
                "description": "Standard user access",
                "permissions": [
                    "user.read", "profile.update",
                    "session.manage"
                ],
                "priority": 10
            },
            "guest": {
                "name": "Guest",
                "description": "Limited read-only access",
                "permissions": [
                    "user.read"
                ],
                "priority": 1
            }
        }
        
        self.roles = self.load_roles()
        self.user_roles = self.load_user_roles()
    
    def load_roles(self):
        """Load role definitions"""
        if os.path.exists(self.roles_file):
            try:
                with open(self.roles_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading roles: {e}")
                return self.default_roles.copy()
        return self.default_roles.copy()
    
    def save_roles(self):
        """Save role definitions"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.roles_file, 'w') as f:
                json.dump(self.roles, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving roles: {e}")
    
    def load_user_roles(self):
        """Load user role assignments"""
        if os.path.exists(self.user_roles_file):
            try:
                with open(self.user_roles_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading user roles: {e}")
                return {}
        return {}
    
    def save_user_roles(self):
        """Save user role assignments"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.user_roles_file, 'w') as f:
                json.dump(self.user_roles, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user roles: {e}")
    
    def assign_role(self, user_id, role_name):
        """
        Assign a role to a user
        
        Args:
            user_id: User identifier
            role_name: Name of the role to assign
        
        Returns:
            bool: Success status
        """
        if role_name not in self.roles:
            logger.error(f"Role {role_name} does not exist")
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = {
                "roles": [],
                "assigned_at": time.time(),
                "updated_at": time.time()
            }
        
        if role_name not in self.user_roles[user_id]["roles"]:
            self.user_roles[user_id]["roles"].append(role_name)
            self.user_roles[user_id]["updated_at"] = time.time()
            self.save_user_roles()
            logger.info(f"Assigned role {role_name} to user {user_id}")
            return True
        
        return False
    
    def remove_role(self, user_id, role_name):
        """Remove a role from a user"""
        if user_id in self.user_roles and role_name in self.user_roles[user_id]["roles"]:
            self.user_roles[user_id]["roles"].remove(role_name)
            self.user_roles[user_id]["updated_at"] = time.time()
            self.save_user_roles()
            logger.info(f"Removed role {role_name} from user {user_id}")
            return True
        
        return False
    
    def get_user_roles(self, user_id):
        """Get all roles assigned to a user"""
        if user_id in self.user_roles:
            return self.user_roles[user_id]["roles"]
        return ["guest"]  # Default role
    
    def get_user_permissions(self, user_id):
        """Get all permissions for a user (from all their roles)"""
        user_roles = self.get_user_roles(user_id)
        permissions = set()
        
        for role_name in user_roles:
            if role_name in self.roles:
                role_permissions = self.roles[role_name].get("permissions", [])
                permissions.update(role_permissions)
        
        return list(permissions)
    
    def has_permission(self, user_id, permission):
        """
        Check if user has a specific permission
        
        Args:
            user_id: User identifier
            permission: Permission string (e.g., "user.create")
        
        Returns:
            bool: True if user has permission
        """
        user_permissions = self.get_user_permissions(user_id)
        
        # Check exact match
        if permission in user_permissions:
            return True
        
        # Check wildcard permissions (e.g., "user.*" matches "user.create")
        permission_parts = permission.split('.')
        for user_perm in user_permissions:
            if user_perm.endswith('.*'):
                if permission.startswith(user_perm[:-2]):
                    return True
        
        return False
    
    def has_role(self, user_id, role_name):
        """Check if user has a specific role"""
        user_roles = self.get_user_roles(user_id)
        return role_name in user_roles
    
    def get_highest_role(self, user_id):
        """Get user's highest priority role"""
        user_roles = self.get_user_roles(user_id)
        
        highest_role = None
        highest_priority = -1
        
        for role_name in user_roles:
            if role_name in self.roles:
                priority = self.roles[role_name].get("priority", 0)
                if priority > highest_priority:
                    highest_priority = priority
                    highest_role = role_name
        
        return highest_role or "guest"
    
    def create_role(self, role_name, name, description, permissions, priority=10):
        """Create a new custom role"""
        if role_name in self.roles:
            logger.error(f"Role {role_name} already exists")
            return False
        
        self.roles[role_name] = {
            "name": name,
            "description": description,
            "permissions": permissions,
            "priority": priority,
            "custom": True,
            "created_at": time.time()
        }
        
        self.save_roles()
        logger.info(f"Created new role: {role_name}")
        return True
    
    def update_role(self, role_name, updates):
        """Update an existing role"""
        if role_name not in self.roles:
            logger.error(f"Role {role_name} does not exist")
            return False
        
        # Don't allow updating default roles
        if not self.roles[role_name].get("custom", False):
            logger.error(f"Cannot update default role: {role_name}")
            return False
        
        self.roles[role_name].update(updates)
        self.roles[role_name]["updated_at"] = time.time()
        self.save_roles()
        logger.info(f"Updated role: {role_name}")
        return True
    
    def delete_role(self, role_name):
        """Delete a custom role"""
        if role_name not in self.roles:
            return False
        
        # Don't allow deleting default roles
        if not self.roles[role_name].get("custom", False):
            logger.error(f"Cannot delete default role: {role_name}")
            return False
        
        # Remove role from all users
        for user_id in self.user_roles:
            if role_name in self.user_roles[user_id]["roles"]:
                self.user_roles[user_id]["roles"].remove(role_name)
        
        del self.roles[role_name]
        self.save_roles()
        self.save_user_roles()
        logger.info(f"Deleted role: {role_name}")
        return True
    
    def get_all_roles(self):
        """Get all role definitions"""
        return self.roles
    
    def get_statistics(self):
        """Get RBAC statistics"""
        total_roles = len(self.roles)
        custom_roles = sum(1 for r in self.roles.values() if r.get("custom", False))
        total_users = len(self.user_roles)
        
        role_distribution = {}
        for user_data in self.user_roles.values():
            for role in user_data["roles"]:
                role_distribution[role] = role_distribution.get(role, 0) + 1
        
        return {
            "total_roles": total_roles,
            "default_roles": total_roles - custom_roles,
            "custom_roles": custom_roles,
            "total_users": total_users,
            "role_distribution": role_distribution
        }


# Decorators for route protection

def require_permission(permission):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get("logged_in"):
                flash("Please login to access this resource", "error")
                return redirect(url_for("authenticate"))
            
            user_id = session.get("user_id")
            
            # Get RBAC manager from app context
            from flask import current_app
            rbac = current_app.config.get('RBAC_MANAGER')
            
            if not rbac:
                flash("Access control not configured", "error")
                return redirect(url_for("index"))
            
            if not rbac.has_permission(user_id, permission):
                flash(f"Insufficient permissions. Required: {permission}", "error")
                return redirect(url_for("dashboard"))
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_role(role_name):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get("logged_in"):
                flash("Please login to access this resource", "error")
                return redirect(url_for("authenticate"))
            
            user_id = session.get("user_id")
            
            # Get RBAC manager from app context
            from flask import current_app
            rbac = current_app.config.get('RBAC_MANAGER')
            
            if not rbac:
                flash("Access control not configured", "error")
                return redirect(url_for("index"))
            
            if not rbac.has_role(user_id, role_name):
                flash(f"Insufficient permissions. Required role: {role_name}", "error")
                return redirect(url_for("dashboard"))
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator
