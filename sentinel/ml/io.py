import os
import time
import joblib
from typing import Any, Dict, Tuple

def save_model(path: str, model: Any, metadata: Dict = None) -> None:
    """
    Save model and metadata together as a single joblib payload.
    Ensures parent directory exists.
    """
    payload = {
        "model": model,
        "metadata": dict(metadata or {}),
    }
    payload["metadata"].setdefault("saved_at", time.time())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(payload, path)

def load_model(path: str) -> Tuple[Any, Dict]:
    """
    Load a model payload saved with save_model.
    Returns (model, metadata).
    """
    payload = joblib.load(path)
    return payload.get("model"), payload.get("metadata", {})
