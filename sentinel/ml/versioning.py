# Model Versioning & A/B Testing Module
# Manages ML model versions, performance tracking, and A/B testing

import json
import os
import time
import logging
import hashlib
from typing import Dict, Any, List
import shutil

logger = logging.getLogger(__name__)

class ModelVersionManager:
    """
    Comprehensive model versioning and A/B testing system.
    Tracks model performance, manages versions, and runs A/B tests.
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.versions_file = os.path.join(storage_dir, "model_versions_registry.json")
        self.ab_tests_file = os.path.join(storage_dir, "ab_tests.json")
        
        # Model registry
        self.model_registry = {}
        
        # Active A/B tests
        self.ab_tests = {}
        
        # Load existing data
        self.load_registry()
        self.load_ab_tests()
    
    def load_registry(self):
        """Load model version registry"""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r') as f:
                    self.model_registry = json.load(f)
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
                self.model_registry = {}
    
    def save_registry(self):
        """Save model version registry"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.versions_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def load_ab_tests(self):
        """Load A/B tests"""
        if os.path.exists(self.ab_tests_file):
            try:
                with open(self.ab_tests_file, 'r') as f:
                    self.ab_tests = json.load(f)
            except Exception as e:
                logger.error(f"Error loading A/B tests: {e}")
                self.ab_tests = {}
    
    def save_ab_tests(self):
        """Save A/B tests"""
        try:
            with open(self.ab_tests_file, 'w') as f:
                json.dump(self.ab_tests, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving A/B tests: {e}")
    
    def register_model(self, model_name, version, model_path, metadata=None):
        """
        Register a new model version
        
        Args:
            model_name: Name of the model (e.g., 'keystroke_classifier')
            version: Version string (e.g., 'v1.2.0')
            model_path: Path to model file
            metadata: Additional metadata
        
        Returns:
            Dict with registration result
        """
        # Initialize model entry if needed
        if model_name not in self.model_registry:
            self.model_registry[model_name] = {
                'model_name': model_name,
                'versions': {},
                'production_version': None,
                'created_at': time.time()
            }
        
        model_entry = self.model_registry[model_name]
        
        # Check if version already exists
        if version in model_entry['versions']:
            return {
                'success': False,
                'message': f'Version {version} already exists'
            }
        
        # Calculate model hash
        model_hash = self._calculate_file_hash(model_path)
        
        # Register version
        version_data = {
            'version': version,
            'model_path': model_path,
            'model_hash': model_hash,
            'registered_at': time.time(),
            'metadata': metadata or {},
            'status': 'registered',
            'performance_metrics': {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'false_positive_rate': None,
                'false_negative_rate': None
            },
            'deployment_history': [],
            'test_results': []
        }
        
        model_entry['versions'][version] = version_data
        
        # Set as production if first version
        if model_entry['production_version'] is None:
            model_entry['production_version'] = version
            version_data['status'] = 'production'
        
        self.save_registry()
        
        logger.info(f"Registered model {model_name} version {version}")
        
        return {
            'success': True,
            'model_name': model_name,
            'version': version,
            'model_hash': model_hash,
            'status': version_data['status']
        }
    
    def update_performance_metrics(self, model_name, version, metrics):
        """Update performance metrics for a model version"""
        if model_name not in self.model_registry:
            return False
        
        if version not in self.model_registry[model_name]['versions']:
            return False
        
        version_data = self.model_registry[model_name]['versions'][version]
        version_data['performance_metrics'].update(metrics)
        version_data['last_metrics_update'] = time.time()
        
        self.save_registry()
        
        return True
    
    def deploy_to_production(self, model_name, version):
        """Deploy a model version to production"""
        if model_name not in self.model_registry:
            return {'success': False, 'message': 'Model not found'}
        
        model_entry = self.model_registry[model_name]
        
        if version not in model_entry['versions']:
            return {'success': False, 'message': 'Version not found'}
        
        # Get current production version
        old_version = model_entry['production_version']
        
        # Update statuses
        if old_version and old_version in model_entry['versions']:
            model_entry['versions'][old_version]['status'] = 'archived'
        
        model_entry['versions'][version]['status'] = 'production'
        model_entry['production_version'] = version
        
        # Record deployment
        deployment = {
            'timestamp': time.time(),
            'from_version': old_version,
            'to_version': version,
            'deployed_by': 'system'
        }
        
        model_entry['versions'][version]['deployment_history'].append(deployment)
        
        self.save_registry()
        
        logger.info(f"Deployed {model_name} version {version} to production")
        
        return {
            'success': True,
            'model_name': model_name,
            'old_version': old_version,
            'new_version': version
        }
    
    def create_ab_test(self, test_name, model_name, version_a, version_b, traffic_split=0.5):
        """
        Create an A/B test between two model versions
        
        Args:
            test_name: Name of the test
            model_name: Model to test
            version_a: Control version
            version_b: Treatment version
            traffic_split: Fraction of traffic to version_b (0-1)
        
        Returns:
            Dict with test configuration
        """
        if model_name not in self.model_registry:
            return {'success': False, 'message': 'Model not found'}
        
        model_entry = self.model_registry[model_name]
        
        if version_a not in model_entry['versions'] or version_b not in model_entry['versions']:
            return {'success': False, 'message': 'One or both versions not found'}
        
        # Create test
        test_data = {
            'test_name': test_name,
            'model_name': model_name,
            'version_a': version_a,
            'version_b': version_b,
            'traffic_split': traffic_split,
            'created_at': time.time(),
            'status': 'active',
            'results': {
                'version_a': {
                    'requests': 0,
                    'successes': 0,
                    'failures': 0,
                    'avg_latency': 0,
                    'metrics': {}
                },
                'version_b': {
                    'requests': 0,
                    'successes': 0,
                    'failures': 0,
                    'avg_latency': 0,
                    'metrics': {}
                }
            }
        }
        
        self.ab_tests[test_name] = test_data
        self.save_ab_tests()
        
        logger.info(f"Created A/B test: {test_name}")
        
        return {
            'success': True,
            'test_name': test_name,
            'test_data': test_data
        }
    
    def record_ab_test_result(self, test_name, version, success, latency=None, metrics=None):
        """Record result from A/B test"""
        if test_name not in self.ab_tests:
            return False
        
        test = self.ab_tests[test_name]
        
        if version not in [test['version_a'], test['version_b']]:
            return False
        
        # Determine which variant
        variant = 'version_a' if version == test['version_a'] else 'version_b'
        results = test['results'][variant]
        
        # Update counts
        results['requests'] += 1
        if success:
            results['successes'] += 1
        else:
            results['failures'] += 1
        
        # Update latency
        if latency is not None:
            current_avg = results['avg_latency']
            n = results['requests']
            results['avg_latency'] = (current_avg * (n - 1) + latency) / n
        
        # Update metrics
        if metrics:
            for key, value in metrics.items():
                if key not in results['metrics']:
                    results['metrics'][key] = []
                results['metrics'][key].append(value)
        
        self.save_ab_tests()
        
        return True
    
    def get_ab_test_winner(self, test_name, metric='success_rate'):
        """Determine winner of A/B test"""
        if test_name not in self.ab_tests:
            return None
        
        test = self.ab_tests[test_name]
        results_a = test['results']['version_a']
        results_b = test['results']['version_b']
        
        # Calculate success rates
        success_rate_a = results_a['successes'] / results_a['requests'] if results_a['requests'] > 0 else 0
        success_rate_b = results_b['successes'] / results_b['requests'] if results_b['requests'] > 0 else 0
        
        # Determine winner
        if success_rate_b > success_rate_a * 1.05:  # 5% improvement threshold
            winner = test['version_b']
            confidence = (success_rate_b - success_rate_a) / success_rate_a if success_rate_a > 0 else 1.0
        elif success_rate_a > success_rate_b * 1.05:
            winner = test['version_a']
            confidence = (success_rate_a - success_rate_b) / success_rate_b if success_rate_b > 0 else 1.0
        else:
            winner = None
            confidence = 0.0
        
        return {
            'test_name': test_name,
            'winner': winner,
            'confidence': round(confidence, 3),
            'version_a_success_rate': round(success_rate_a, 3),
            'version_b_success_rate': round(success_rate_b, 3),
            'version_a_requests': results_a['requests'],
            'version_b_requests': results_b['requests']
        }
    
    def close_ab_test(self, test_name, deploy_winner=False):
        """Close an A/B test"""
        if test_name not in self.ab_tests:
            return {'success': False, 'message': 'Test not found'}
        
        test = self.ab_tests[test_name]
        test['status'] = 'completed'
        test['completed_at'] = time.time()
        
        # Get winner
        winner_info = self.get_ab_test_winner(test_name)
        test['winner'] = winner_info
        
        # Deploy winner if requested
        if deploy_winner and winner_info['winner']:
            self.deploy_to_production(test['model_name'], winner_info['winner'])
        
        self.save_ab_tests()
        
        return {
            'success': True,
            'test_name': test_name,
            'winner': winner_info
        }
    
    def get_model_versions(self, model_name):
        """Get all versions of a model"""
        if model_name not in self.model_registry:
            return None
        
        model_entry = self.model_registry[model_name]
        
        versions = []
        for version, data in model_entry['versions'].items():
            versions.append({
                'version': version,
                'status': data['status'],
                'registered_at': data['registered_at'],
                'performance_metrics': data['performance_metrics']
            })
        
        # Sort by registration time (newest first)
        versions.sort(key=lambda v: v['registered_at'], reverse=True)
        
        return {
            'model_name': model_name,
            'production_version': model_entry['production_version'],
            'versions': versions
        }
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of file"""
        if not os.path.exists(file_path):
            return None
        
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def get_statistics(self):
        """Get versioning statistics"""
        total_models = len(self.model_registry)
        total_versions = sum(len(m['versions']) for m in self.model_registry.values())
        active_tests = sum(1 for t in self.ab_tests.values() if t['status'] == 'active')
        
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'active_ab_tests': active_tests,
            'completed_ab_tests': len(self.ab_tests) - active_tests
        }
