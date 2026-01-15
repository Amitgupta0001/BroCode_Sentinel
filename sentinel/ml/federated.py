# BroCode Sentinel - Federated Learning Engine
# Securely aggregates model improvements without exposing raw biometric data

import numpy as np
import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedLearningEngine:
    """
    Implements a Federated Averaging (FedAvg) system for keystroke and behavioral models.
    Enables collective intelligence while ensuring data residency within device/tenant.
    """
    
    def __init__(self, model_save_path: str = "models/global_model.json"):
        self.model_save_path = model_save_path
        self.global_weights = {} # Represents the global state of the model
        self.client_weights_buffer: List[Dict] = []
        self.min_clients_for_update = 3 # Minimum updates needed before aggregation
        
        self.load_global_model()

    def load_global_model(self):
        """Loads the current global model state."""
        if os.path.exists(self.model_save_path):
            with open(self.model_save_path, 'r') as f:
                self.global_weights = json.load(f)
        else:
            # Initialize with random/default weights if no model exists
            self.global_weights = {"w1": 0.5, "w2": 0.3, "bias": 0.1} 

    def receive_local_update(self, tenant_id: str, local_weights: Dict, sample_size: int):
        """
        Receives model weights from a local client/tenant.
        Weights are weighted by the number of samples used during local training.
        """
        update = {
            "tenant": tenant_id,
            "weights": local_weights,
            "samples": sample_size,
            "timestamp": datetime.now().isoformat()
        }
        self.client_weights_buffer.append(update)
        logger.info(f"Received local model update from {tenant_id} (Samples: {sample_size})")

        # Trigger aggregation if threshold met
        if len(self.client_weights_buffer) >= self.min_clients_for_update:
            self.aggregate_global_model()

    def aggregate_global_model(self):
        """
        Performs Federated Averaging to update the global model.
        New Global Weight = Sum(Local_Weight * Sample_Size) / Total_Samples
        """
        logger.info("Triggering Global Model Aggregation...")
        
        total_samples = sum(u["samples"] for u in self.client_weights_buffer)
        new_weights = {}

        # Aggregate each parameter in the model
        for key in self.global_weights.keys():
            weighted_sum = sum(u["weights"].get(key, 0) * u["samples"] for u in self.client_weights_buffer)
            new_weights[key] = weighted_sum / total_samples

        self.global_weights = new_weights
        self.client_weights_buffer = [] # Clear buffer after aggregation
        self.save_global_model()
        logger.info("Global model successfully updated via FedAvg.")

    def save_global_model(self):
        """Persists the aggregated global model."""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        with open(self.model_save_path, 'w') as f:
            json.dump(self.global_weights, f, indent=4)

    def get_latest_global_model(self) -> Dict:
        """Allows clients to download the latest global model for local inference/tuning."""
        return self.global_weights

# Implementation for a local client simulator
class FederatedClient:
    def __init__(self, tenant_id: str, local_data: np.ndarray):
        self.tenant_id = tenant_id
        self.local_data = local_data
        
    def train_locally(self, current_global_weights: Dict) -> Dict:
        """
        Simulates local training process.
        Adjusts global weights based on local biometric patterns.
        """
        # (Simplified) Add local 'noise' or 'learning' based on data
        learning_rate = 0.01
        local_improvement = np.mean(self.local_data) * learning_rate
        
        updated_weights = {}
        for k, v in current_global_weights.items():
            updated_weights[k] = v + local_improvement
            
        return updated_weights
