import os
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sentinel.ml.io import save_model, load_model

@dataclass
class TrainResult:
    success: bool
    accuracy: float

class KeystrokeAuthenticationModel:
    """
    Machine Learning Model for Keystroke Dynamics Authentication
    Based on the BKSD dataset research - Multilingual Version
    """
    
    def __init__(self, user_id, language):
        self.user_id = user_id
        self.language = language
        self.model = None
        self.is_trained = False
        self.feature_info = {}
        
    def extract_features(self, raw_keystrokes):
        """
        Extract features from raw keystroke timing data
        Based on BKSD dataset feature extraction methodology
        Enhanced for multilingual support
        """
        if len(raw_keystrokes) < 4:
            # Return zero features if insufficient data
            return np.zeros(17)
            
        features = []
        
        # Hold times (dwell times)
        hold_times = []
        for i in range(len(raw_keystrokes)):
            if i % 2 == 0:  # keydown events
                if i + 1 < len(raw_keystrokes):  # corresponding keyup exists
                    hold_time = raw_keystrokes[i + 1] - raw_keystrokes[i]
                    hold_times.append(hold_time)
        
        # Flight times (keyup-keydown)
        flight_times = []
        for i in range(1, len(raw_keystrokes) - 2, 2):
            if i + 2 < len(raw_keystrokes):
                flight_time = raw_keystrokes[i + 2] - raw_keystrokes[i]
                flight_times.append(flight_time)
        
        # Keydown-keydown latencies
        keydown_latencies = []
        for i in range(0, len(raw_keystrokes) - 2, 2):
            if i + 2 < len(raw_keystrokes):
                latency = raw_keystrokes[i + 2] - raw_keystrokes[i]
                keydown_latencies.append(latency)
        
        # Statistical features for hold times
        if hold_times:
            features.extend([
                np.mean(hold_times), np.std(hold_times), np.min(hold_times), 
                np.max(hold_times), np.median(hold_times)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Statistical features for flight times
        if flight_times:
            features.extend([
                np.mean(flight_times), np.std(flight_times), np.min(flight_times),
                np.max(flight_times), np.median(flight_times)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Statistical features for keydown latencies
        if keydown_latencies:
            features.extend([
                np.mean(keydown_latencies), np.std(keydown_latencies),
                np.min(keydown_latencies), np.max(keydown_latencies),
                np.median(keydown_latencies)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Total typing duration
        if len(raw_keystrokes) >= 2:
            total_duration = raw_keystrokes[-1] - raw_keystrokes[0]
            features.append(total_duration)
        else:
            features.append(0)
        
        # Typing speed (keys per second)
        num_keys = len(hold_times)
        if total_duration > 0 and num_keys > 0:
            typing_speed = num_keys / (total_duration / 1000)  # keys per second
            features.append(typing_speed)
        else:
            features.append(0)
        
        self.feature_names = [
            'hold_time_mean', 'hold_time_std', 'hold_time_min', 'hold_time_max', 'hold_time_median',
            'flight_time_mean', 'flight_time_std', 'flight_time_min', 'flight_time_max', 'flight_time_median',
            'keydown_latency_mean', 'keydown_latency_std', 'keydown_latency_min', 'keydown_latency_max', 'keydown_latency_median',
            'total_duration', 'typing_speed'
        ]
        
        return np.array(features)
    
    def create_synthetic_imposters(self, genuine_features, n_imposters=100):
        """
        Create synthetic imposter data using Safe-Level SMOTE inspired approach
        """
        if len(genuine_features) == 0:
            return np.array([])
            
        n_genuine = len(genuine_features)
        imposters = []
        
        for _ in range(n_imposters):
            # Select two random genuine samples
            idx1, idx2 = np.random.choice(n_genuine, 2, replace=False)
            sample1 = genuine_features[idx1]
            sample2 = genuine_features[idx2]
            
            # Create synthetic sample with some variation
            alpha = np.random.uniform(0.1, 0.9)
            synthetic = alpha * sample1 + (1 - alpha) * sample2
            
            # Add some noise
            noise = np.random.normal(0, 0.1, synthetic.shape)
            synthetic += noise
            
            imposters.append(synthetic)
        
        return np.array(imposters)
    
    def prepare_dataset(self, genuine_samples, imposter_samples=None):
        """
        Prepare dataset for training with genuine and imposter samples
        """
        X_genuine = np.array([self.extract_features(sample) for sample in genuine_samples])
        y_genuine = np.ones(len(X_genuine))  # 1 for genuine user
        
        if imposter_samples is None:
            # Create synthetic imposters if not provided
            X_imposter = self.create_synthetic_imposters(X_genuine)
        else:
            X_imposter = np.array([self.extract_features(sample) for sample in imposter_samples])
        
        y_imposter = np.zeros(len(X_imposter))  # 0 for imposters
        
        # Combine datasets
        X = np.vstack([X_genuine, X_imposter])
        y = np.hstack([y_genuine, y_imposter])
        
        return X, y
    
    def train(self, samples, labels, save_path: Optional[str] = None, test_size: float = 0.2, random_state: int = 42, **kwargs) -> TrainResult:
        """
        Train a simple classifier over keystroke-derived features.

        Accepts:
          - samples: list of samples; each sample can be either
             * a list/sequence of event dicts (each with 't' timestamp), or
             * a numeric feature sequence (list/ndarray).
          - labels: list/ndarray of labels (int/str)
        Returns:
          TrainResult(success, accuracy)
        """
        try:
            X_list: List[np.ndarray] = []
            for s in samples:
                if s is None:
                    continue
                if isinstance(s, (list, tuple)) and s and isinstance(s[0], dict):
                    feat = _extract_features_from_events(s)
                else:
                    arr = np.asarray(s, dtype=float)
                    if arr.size >= 5:
                        # reduce to 5 by mean/std + length
                        feat = np.array([float(np.mean(arr)), float(np.std(arr)), 0.0, 0.0, float(arr.size)])
                    else:
                        pad = np.zeros(5 - arr.size)
                        feat = np.concatenate([arr.flatten(), pad])
                X_list.append(feat)
            if len(X_list) == 0:
                raise ValueError("No valid samples provided")

            X = np.vstack(X_list)
            y = np.asarray(labels)

            # ensure at least two classes; create synthetic negatives if necessary
            if len(np.unique(y)) == 1:
                neg_count = max(2, int(0.5 * X.shape[0]))
                neg = X[:neg_count] + np.random.normal(scale=0.01, size=(neg_count, X.shape[1]))
                X = np.vstack([X, neg])
                y = np.concatenate([y, np.array(["__other__"] * neg_count)])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )

            clf = LogisticRegression(max_iter=400)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            accuracy = float(accuracy_score(y_test, preds))

            model_path = save_path or f"{self.model_dir}/{self.user_id}_{self.language}.joblib"
            payload = {"sk_model": clf, "feature_info": {"type": "iki_hold_basic", "dim": X.shape[1]}}
            save_model(model_path, payload, metadata={"user_id": self.user_id, "language": self.language, "accuracy": accuracy})

            self.model = clf
            self.feature_info = payload["feature_info"]
            return TrainResult(success=True, accuracy=accuracy)
        except Exception:
            return TrainResult(success=False, accuracy=0.0)

    def save_model(self, path: str, metadata: Optional[dict] = None) -> None:
        save_model(path, getattr(self, "model", {"user_id": self.user_id, "language": self.language}), metadata={"user_id": self.user_id, "language": self.language, **(metadata or {})})

    @classmethod
    def load_model(cls, path: str):
        model_obj, metadata = load_model(path)
        if model_obj is None:
            return None
        inst = cls(user_id=metadata.get("user_id", "unknown"), language=metadata.get("language", "unknown"), model_dir=os.path.dirname(path))
        # rehydrate sklearn model or payload
        if isinstance(model_obj, dict) and "sk_model" in model_obj:
            inst.model = model_obj["sk_model"]
            inst.feature_info = model_obj.get("feature_info")
        else:
            inst.model = model_obj
        return inst

    def authenticate(self, sample_events) -> Tuple[bool, float]:
        """
        Return (accepted:bool, score:float) using the trained internal model.
        """
        if not hasattr(self, "model") or self.model is None:
            return False, 0.0
        if isinstance(sample_events, (list, tuple)) and sample_events and isinstance(sample_events[0], dict):
            feat = _extract_features_from_events(sample_events).reshape(1, -1)
        else:
            feat = np.asarray(sample_events, dtype=float).reshape(1, -1)
            # pad to match expected dim 5 if needed
            if feat.shape[1] < 5:
                pad = np.zeros((1, 5 - feat.shape[1]))
                feat = np.hstack([feat, pad])
        try:
            prob = max(self.model.predict_proba(feat).max(), 0.0) if hasattr(self.model, "predict_proba") else 0.0
            pred = self.model.predict(feat)[0]
            accepted = bool(pred == getattr(self, "user_id", pred))
            return accepted, float(prob)
        except Exception:
            return False, 0.0

def _extract_features_from_events(events: Sequence[dict]) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a sequence of key events.

    Expected event format: {'key': 'a', 't': <ms>, 'type': 'keydown'|'keyup'}
    Features (5):
      - mean inter-key-interval (IKI) between consecutive keydown events
      - std of IKI
      - mean key hold time (keydown->keyup) across matched keys
      - std of hold times
      - number of unique keys / event count
    """
    if not events:
        return np.zeros(5, dtype=float)

    # collect keydown times in order
    keydown_times = []
    # map of last keydown time per key to compute hold times
    last_keydown = {}
    hold_times = []
    for ev in events:
        try:
            t = float(ev.get("t", 0.0))
        except Exception:
            continue
        typ = ev.get("type", "").lower()
        key = ev.get("key")
        if typ == "keydown":
            keydown_times.append(t)
            last_keydown[key] = t
        elif typ == "keyup":
            if key in last_keydown:
                ht = t - last_keydown.pop(key)
                if ht >= 0:
                    hold_times.append(ht)

    # IKIs from keydown times
    if len(keydown_times) >= 2:
        sorted_kd = np.asarray(keydown_times, dtype=float)
        ikis = np.diff(sorted_kd)
        mean_iki = float(np.mean(ikis))
        std_iki = float(np.std(ikis))
    else:
        mean_iki = 0.0
        std_iki = 0.0

    if len(hold_times) >= 1:
        ht_arr = np.asarray(hold_times, dtype=float)
        mean_hold = float(np.mean(ht_arr))
        std_hold = float(np.std(ht_arr))
    else:
        mean_hold = 0.0
        std_hold = 0.0

    unique_keys = len({ev.get("key") for ev in events if ev.get("key") is not None})
    count = float(unique_keys or len(events))

    return np.array([mean_iki, std_iki, mean_hold, std_hold, count], dtype=float)

class BilingualAuthenticationSystem:
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        self.user_models = {}  # cache user-language -> model

    def _model_path(self, user_id: str, language: str) -> str:
        return f"{self.model_dir}/{user_id}_{language}.joblib"

    def register_user(self, user_id: str, language: str, samples, labels, **kwargs) -> bool:
        model = KeystrokeAuthenticationModel(user_id=user_id, language=language, model_dir=self.model_dir)
        result = model.train(samples, labels, save_path=self._model_path(user_id, language), **kwargs)
        if isinstance(result, (int, float)):
            return float(result) > 0
        if isinstance(result, TrainResult):
            return result.success
        return False

    def authenticate_user(self, user_id: str, language: str, sample_events) -> Tuple[bool, float]:
        key = f"{user_id}_{language}"
        if key in self.user_models:
            model = self.user_models[key]
        else:
            path = self._model_path(user_id, language)
            model = KeystrokeAuthenticationModel.load_model(path)
            if model is None:
                return False, 0.0
            self.user_models[key] = model
        return model.authenticate(sample_events)

# Example usage and demonstration
def demo_multilingual_authentication():
    """
    Demonstration of the multilingual keystroke authentication system
    """
    print("=== Multilingual Keystroke Dynamics Authentication System ===\n")
    
    # Create multilingual authentication system
    auth_system = BilingualAuthenticationSystem()
    
    # Simulate sample data
    def simulate_keystroke_samples(n_samples=50, user_pattern=True):
        """Simulate keystroke timing data for demonstration"""
        samples = []
        for _ in range(n_samples):
            if user_pattern:
                base_times = np.cumsum(np.random.normal(100, 20, 10))
            else:
                base_times = np.cumsum(np.random.normal(150, 30, 10))
            
            keystroke_sequence = []
            for time in base_times:
                keystroke_sequence.extend([time, time + np.random.normal(50, 10)])
            
            samples.append(keystroke_sequence)
        return samples
    
    # Official UN languages to demonstrate
    demo_languages = ['english', 'arabic', 'french', 'spanish']
    
    # Register user for multiple languages
    user_id = "multilingual_user"
    
    for language in demo_languages:
        try:
            samples = simulate_keystroke_samples(10, True)
            auth_system.register_user(user_id, language, samples)
            print(f"✓ Registered for {language}")
        except Exception as e:
            print(f"✗ Failed to register for {language}: {e}")
    
    # Test authentication in different languages
    print("\n--- Testing Multilingual Authentication ---")
    
    for language in demo_languages:
        try:
            test_sample = simulate_keystroke_samples(1, True)[0]
            is_auth, confidence = auth_system.authenticate_user(user_id, language, test_sample)
            print(f"{language.capitalize()}: Authenticated={is_auth}, Confidence={confidence:.4f}")
        except Exception as e:
            print(f"{language.capitalize()}: Error - {e}")
    
    # Show user statistics
    print(f"\n--- User Language Statistics ---")
    stats = auth_system.get_user_language_stats(user_id)
    print(f"User: {stats['user_id']}")
    print(f"Languages registered: {stats['total_languages']}")
    print(f"Languages: {', '.join(stats['languages'])}")
    
    # Show all users
    all_users = auth_system.get_all_users()
    print(f"\nAll registered users: {all_users}")

if __name__ == "__main__":
    # Run demonstration
    demo_multilingual_authentication()
