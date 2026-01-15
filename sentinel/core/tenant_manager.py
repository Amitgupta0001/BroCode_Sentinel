# BroCode Sentinel - Multi-Tenant Enterprise Manager
# Provides logical and data isolation for multiple client organizations

import os
import json
import logging
import uuid
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TenantManager:
    """
    Manages tenant registration, configuration, and data isolation paths.
    Each tenant has its own security policy, user base, and audit trails.
    """
    
    def __init__(self, base_data_dir: str = "tenant_data"):
        self.base_data_dir = base_data_dir
        self.tenants_registry_file = os.path.join(base_data_dir, "tenants.json")
        self.tenants: Dict[str, Dict] = {}
        
        if not os.path.exists(self.base_data_dir):
            os.makedirs(self.base_data_dir)
            
        self.load_tenants()

    def load_tenants(self):
        """Loads tenant registry from persistent storage."""
        if os.path.exists(self.tenants_registry_file):
            try:
                with open(self.tenants_registry_file, 'r') as f:
                    self.tenants = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load tenant registry: {e}")

    def save_tenants(self):
        """Persists tenant registry."""
        with open(self.tenants_registry_file, 'w') as f:
            json.dump(self.tenants, f, indent=4)

    def create_tenant(self, name: str, industry: str = "general") -> str:
        """
        Registers a new tenant with unique ID and dedicated storage.
        """
        tenant_id = f"tnt_{uuid.uuid4().hex[:8]}"
        storage_path = os.path.join(self.base_data_dir, tenant_id)
        
        # Create dedicated storage directories
        os.makedirs(os.path.join(storage_path, "users"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "data/models"), exist_ok=True)
        
        self.tenants[tenant_id] = {
            "name": name,
            "id": tenant_id,
            "created_at": datetime.now().isoformat(),
            "industry": industry,
            "storage_path": storage_path,
            "settings": {
                "base_trust_threshold": 0.6,
                "max_failed_attempts": 3,
                "require_blink": True,
                "reauth_interval_minutes": 60
            },
            "status": "active"
        }
        
        self.save_tenants()
        logger.info(f"Tenant created: {name} (ID: {tenant_id})")
        return tenant_id

    def get_tenant_config(self, tenant_id: str) -> Optional[Dict]:
        """Retrieves specific configuration for a tenant."""
        return self.tenants.get(tenant_id)

    def update_tenant_policy(self, tenant_id: str, new_settings: Dict):
        """Updates the security policy for a specific tenant."""
        if tenant_id in self.tenants:
            self.tenants[tenant_id]["settings"].update(new_settings)
            self.save_tenants()
            return True
        return False

    def get_isolated_path(self, tenant_id: str, sub_dir: str) -> Optional[str]:
        """Provides a safe path for data storage within a tenant's silo."""
        if tenant_id in self.tenants:
            return os.path.join(self.tenants[tenant_id]["storage_path"], sub_dir)
        return None

# Singleton usage for application persistence
tenant_engine = TenantManager()

# Example: Initializing Default Tenants
if not tenant_engine.tenants:
    t1 = tenant_engine.create_tenant("BroCode Global", "Technology")
    t2 = tenant_engine.create_tenant("Sentix Medical", "Healthcare")
