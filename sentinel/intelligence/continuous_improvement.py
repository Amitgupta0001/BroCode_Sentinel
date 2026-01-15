# Continuous Model Improvement System
# Automatically monitors and improves ML models over time

import json
import os
import time
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ContinuousImprovement:
    """
    Monitors model performance and triggers retraining when needed.
    Implements A/B testing for model versions.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.metrics_file = os.path.join(storage_dir, "model_metrics.json")
        self.versions_file = os.path.join(storage_dir, "model_versions.json")
        
        self.metrics = self.load_metrics()
        self.versions = self.load_versions()
        
        # Performance thresholds for retraining
        self.thresholds = {
            "accuracy_drop": 0.05,  # 5% drop triggers retraining
            "false_positive_rate": 0.1,  # 10% FPR is too high
            "false_negative_rate": 0.1,  # 10% FNR is too high
            "min_samples": 100  # Minimum samples before retraining
        }
    
    def load_metrics(self):
        """Load model performance metrics"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
                return {}
        return {}
    
    def save_metrics(self):
        """Save performance metrics"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def load_versions(self):
        """Load model version history"""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading versions: {e}")
                return {}
        return {}
    
    def save_versions(self):
        """Save model versions"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.versions_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving versions: {e}")
    
    def record_prediction(self, model_name, prediction, actual=None, confidence=None):
        """
        Record a model prediction for performance tracking
        
        Args:
            model_name: Name of the model
            prediction: Predicted value
            actual: Actual value (if known)
            confidence: Prediction confidence
        """
        if model_name not in self.metrics:
            self.metrics[model_name] = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "confidence_scores": [],
                "created_at": time.time(),
                "last_updated": time.time()
            }
        
        metrics = self.metrics[model_name]
        metrics["total_predictions"] += 1
        metrics["last_updated"] = time.time()
        
        if confidence is not None:
            metrics["confidence_scores"].append(confidence)
            # Keep only last 1000 scores
            if len(metrics["confidence_scores"]) > 1000:
                metrics["confidence_scores"] = metrics["confidence_scores"][-1000:]
        
        if actual is not None:
            if prediction == actual:
                metrics["correct_predictions"] += 1
            else:
                if prediction == 1:  # Predicted positive
                    metrics["false_positives"] += 1
                else:  # Predicted negative
                    metrics["false_negatives"] += 1
        
        self.save_metrics()
        
        # Check if retraining needed
        if self.should_retrain(model_name):
            logger.warning(f"Model {model_name} needs retraining!")
            return {"retrain_needed": True}
        
        return {"retrain_needed": False}
    
    def should_retrain(self, model_name):
        """Determine if model should be retrained"""
        if model_name not in self.metrics:
            return False
        
        metrics = self.metrics[model_name]
        total = metrics["total_predictions"]
        
        if total < self.thresholds["min_samples"]:
            return False
        
        # Calculate current accuracy
        correct = metrics["correct_predictions"]
        accuracy = correct / total if total > 0 else 0
        
        # Get baseline accuracy
        baseline = self.get_baseline_accuracy(model_name)
        
        # Check accuracy drop
        if baseline - accuracy > self.thresholds["accuracy_drop"]:
            logger.info(f"Accuracy dropped from {baseline:.2f} to {accuracy:.2f}")
            return True
        
        # Check false positive rate
        fp_rate = metrics["false_positives"] / total if total > 0 else 0
        if fp_rate > self.thresholds["false_positive_rate"]:
            logger.info(f"False positive rate too high: {fp_rate:.2f}")
            return True
        
        # Check false negative rate
        fn_rate = metrics["false_negatives"] / total if total > 0 else 0
        if fn_rate > self.thresholds["false_negative_rate"]:
            logger.info(f"False negative rate too high: {fn_rate:.2f}")
            return True
        
        return False
    
    def get_baseline_accuracy(self, model_name):
        """Get baseline accuracy for model"""
        if model_name in self.versions:
            current_version = self.versions[model_name].get("current_version")
            if current_version:
                version_data = self.versions[model_name]["versions"].get(current_version, {})
                return version_data.get("accuracy", 0.8)
        return 0.8  # Default baseline
    
    def create_model_version(self, model_name, version_id, accuracy, metadata=None):
        """Create a new model version"""
        if model_name not in self.versions:
            self.versions[model_name] = {
                "current_version": None,
                "versions": {},
                "ab_test": None
            }
        
        self.versions[model_name]["versions"][version_id] = {
            "version_id": version_id,
            "accuracy": accuracy,
            "created_at": time.time(),
            "metadata": metadata or {},
            "deployed": False
        }
        
        self.save_versions()
        logger.info(f"Created model version {version_id} for {model_name}")
    
    def deploy_version(self, model_name, version_id):
        """Deploy a model version to production"""
        if model_name not in self.versions:
            raise ValueError(f"Model {model_name} not found")
        
        if version_id not in self.versions[model_name]["versions"]:
            raise ValueError(f"Version {version_id} not found")
        
        # Mark old version as not deployed
        old_version = self.versions[model_name].get("current_version")
        if old_version:
            self.versions[model_name]["versions"][old_version]["deployed"] = False
        
        # Deploy new version
        self.versions[model_name]["current_version"] = version_id
        self.versions[model_name]["versions"][version_id]["deployed"] = True
        self.versions[model_name]["versions"][version_id]["deployed_at"] = time.time()
        
        self.save_versions()
        logger.info(f"Deployed version {version_id} for {model_name}")
    
    def start_ab_test(self, model_name, version_a, version_b, traffic_split=0.5):
        """
        Start A/B test between two model versions
        
        Args:
            model_name: Name of the model
            version_a: First version ID
            version_b: Second version ID
            traffic_split: Percentage of traffic for version_b (0-1)
        """
        if model_name not in self.versions:
            raise ValueError(f"Model {model_name} not found")
        
        self.versions[model_name]["ab_test"] = {
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "started_at": time.time(),
            "metrics_a": {"predictions": 0, "correct": 0},
            "metrics_b": {"predictions": 0, "correct": 0}
        }
        
        self.save_versions()
        logger.info(f"Started A/B test for {model_name}: {version_a} vs {version_b}")
    
    def get_ab_test_version(self, model_name):
        """Get which version to use for A/B test"""
        if model_name not in self.versions:
            return None
        
        ab_test = self.versions[model_name].get("ab_test")
        if not ab_test:
            return self.versions[model_name].get("current_version")
        
        # Random selection based on traffic split
        import random
        if random.random() < ab_test["traffic_split"]:
            return ab_test["version_b"]
        else:
            return ab_test["version_a"]
    
    def record_ab_test_result(self, model_name, version_id, correct):
        """Record A/B test result"""
        if model_name not in self.versions:
            return
        
        ab_test = self.versions[model_name].get("ab_test")
        if not ab_test:
            return
        
        if version_id == ab_test["version_a"]:
            ab_test["metrics_a"]["predictions"] += 1
            if correct:
                ab_test["metrics_a"]["correct"] += 1
        elif version_id == ab_test["version_b"]:
            ab_test["metrics_b"]["predictions"] += 1
            if correct:
                ab_test["metrics_b"]["correct"] += 1
        
        self.save_versions()
    
    def get_ab_test_results(self, model_name):
        """Get A/B test results"""
        if model_name not in self.versions:
            return None
        
        ab_test = self.versions[model_name].get("ab_test")
        if not ab_test:
            return None
        
        metrics_a = ab_test["metrics_a"]
        metrics_b = ab_test["metrics_b"]
        
        accuracy_a = metrics_a["correct"] / metrics_a["predictions"] if metrics_a["predictions"] > 0 else 0
        accuracy_b = metrics_b["correct"] / metrics_b["predictions"] if metrics_b["predictions"] > 0 else 0
        
        return {
            "version_a": ab_test["version_a"],
            "version_b": ab_test["version_b"],
            "accuracy_a": accuracy_a,
            "accuracy_b": accuracy_b,
            "predictions_a": metrics_a["predictions"],
            "predictions_b": metrics_b["predictions"],
            "winner": ab_test["version_b"] if accuracy_b > accuracy_a else ab_test["version_a"],
            "improvement": abs(accuracy_b - accuracy_a)
        }
    
    def end_ab_test(self, model_name, deploy_winner=True):
        """End A/B test and optionally deploy winner"""
        results = self.get_ab_test_results(model_name)
        
        if not results:
            return None
        
        if deploy_winner:
            self.deploy_version(model_name, results["winner"])
        
        # Archive test results
        if model_name not in self.versions:
            return results
        
        if "ab_test_history" not in self.versions[model_name]:
            self.versions[model_name]["ab_test_history"] = []
        
        test_result = self.versions[model_name]["ab_test"].copy()
        test_result["ended_at"] = time.time()
        test_result["winner"] = results["winner"]
        
        self.versions[model_name]["ab_test_history"].append(test_result)
        self.versions[model_name]["ab_test"] = None
        
        self.save_versions()
        logger.info(f"Ended A/B test for {model_name}. Winner: {results['winner']}")
        
        return results
    
    def get_model_performance(self, model_name):
        """Get current model performance metrics"""
        if model_name not in self.metrics:
            return None
        
        metrics = self.metrics[model_name]
        total = metrics["total_predictions"]
        
        if total == 0:
            return None
        
        accuracy = metrics["correct_predictions"] / total
        fp_rate = metrics["false_positives"] / total
        fn_rate = metrics["false_negatives"] / total
        
        confidence_scores = metrics.get("confidence_scores", [])
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            "model_name": model_name,
            "total_predictions": total,
            "accuracy": round(accuracy, 3),
            "false_positive_rate": round(fp_rate, 3),
            "false_negative_rate": round(fn_rate, 3),
            "average_confidence": round(avg_confidence, 3),
            "last_updated": metrics["last_updated"],
            "retrain_needed": self.should_retrain(model_name)
        }
    
    def get_statistics(self):
        """Get overall improvement system statistics"""
        total_models = len(self.metrics)
        models_needing_retrain = sum(1 for m in self.metrics if self.should_retrain(m))
        active_ab_tests = sum(1 for v in self.versions.values() if v.get("ab_test"))
        
        return {
            "total_models": total_models,
            "models_needing_retrain": models_needing_retrain,
            "active_ab_tests": active_ab_tests,
            "total_versions": sum(len(v.get("versions", {})) for v in self.versions.values()),
            "thresholds": self.thresholds
        }
