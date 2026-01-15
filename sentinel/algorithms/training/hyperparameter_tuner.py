import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from dataclasses import dataclass
import warnings

@dataclass
class TuningConfig:
    tuning_method: str  # 'grid', 'random', 'bayesian'
    param_distributions: Dict[str, List[Any]]
    scoring: str
    cv: int
    n_iter: int
    n_trials: int
    random_state: int

@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    best_estimator: Any
    cv_results: Dict[str, Any]
    tuning_time: float

class HyperparameterTuner:
    def __init__(self, config: TuningConfig):
        self.config = config
        self.study = None
        
    def tune_hyperparameters(self, 
                           model: Any,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None) -> TuningResult:
        """Perform hyperparameter tuning using specified method"""
        import time
        
        start_time = time.time()
        
        if self.config.tuning_method == 'grid':
            result = self._grid_search(model, X_train, y_train)
        elif self.config.tuning_method == 'random':
            result = self._random_search(model, X_train, y_train)
        elif self.config.tuning_method == 'bayesian':
            result = self._bayesian_optimization(model, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unsupported tuning method: {self.config.tuning_method}")
        
        result.tuning_time = time.time() - start_time
        
        print(f"Hyperparameter tuning completed in {result.tuning_time:.2f} seconds")
        print(f"Best score: {result.best_score:.4f}")
        print(f"Best parameters: {result.best_params}")
        
        return result
    
    def _grid_search(self, model: Any, X: np.ndarray, y: np.ndarray) -> TuningResult:
        """Perform grid search hyperparameter tuning"""
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.config.param_distributions,
            scoring=self.config.scoring,
            cv=self.config.cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        return TuningResult(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            best_estimator=grid_search.best_estimator_,
            cv_results=grid_search.cv_results_,
            tuning_time=0.0
        )
    
    def _random_search(self, model: Any, X: np.ndarray, y: np.ndarray) -> TuningResult:
        """Perform random search hyperparameter tuning"""
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.config.param_distributions,
            n_iter=self.config.n_iter,
            scoring=self.config.scoring,
            cv=self.config.cv,
            n_jobs=-1,
            verbose=1,
            random_state=self.config.random_state,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        return TuningResult(
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            best_estimator=random_search.best_estimator_,
            cv_results=random_search.cv_results_,
            tuning_time=0.0
        )
    
    def _bayesian_optimization(self, 
                             model: Any,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: Optional[np.ndarray],
                             y_val: Optional[np.ndarray]) -> TuningResult:
        """Perform Bayesian optimization for hyperparameter tuning"""
        def objective(trial):
            # Suggest hyperparameters
            params = {}
            for param_name, param_config in self.config.param_distributions.items():
                if isinstance(param_config[0], (int, np.integer)):
                    params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[-1])
                elif isinstance(param_config[0], (float, np.floating)):
                    params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[-1])
                elif isinstance(param_config[0], str):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                else:
                    warnings.warn(f"Unsupported parameter type for {param_name}, using first value")
                    params[param_name] = param_config[0]
            
            # Set model parameters
            model.set_params(**params)
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=self.config.cv, scoring=self.config.scoring)
            
            return scores.mean()
        
        # Create and run study
        self.study = optuna.create_study(direction='maximize', 
                                       sampler=optuna.samplers.TPESampler(seed=self.config.random_state))
        self.study.optimize(objective, n_trials=self.config.n_trials)
        
        # Get best parameters and retrain model
        best_params = self.study.best_params
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        
        # Calculate best score
        if X_val is not None and y_val is not None:
            from sklearn.metrics import get_scorer
            scorer = get_scorer(self.config.scoring)
            best_score = scorer(model, X_val, y_val)
        else:
            best_score = self.study.best_value
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_estimator=model,
            cv_results={'study': self.study.trials_dataframe().to_dict()},
            tuning_time=0.0
        )
    
    def get_optimization_history(self) -> Optional[Any]:
        """Get the optimization history for Bayesian optimization"""
        if self.study is None:
            return None
        
        return self.study.trials_dataframe()
    
    def plot_optimization_history(self):
        """Plot optimization history (for Bayesian optimization)"""
        if self.study is None:
            print("No optimization history available. Run Bayesian optimization first.")
            return
        
        try:
            import plotly.express as px
            
            history_df = self.study.trials_dataframe()
            fig = px.line(history_df, x='number', y='value', 
                         title='Hyperparameter Optimization History',
                         labels={'number': 'Trial', 'value': 'Score'})
            fig.show()
            
        except ImportError:
            print("Plotly is required for visualization. Install with: pip install plotly")
    
    def parameter_importance_analysis(self) -> Optional[Dict[str, float]]:
        """Analyze parameter importance (for Bayesian optimization)"""
        if self.study is None:
            return None
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            warnings.warn(f"Parameter importance analysis failed: {e}")
            return None
    
    def create_parameter_grid(self, 
                            model_type: str,
                            complexity: str = 'medium') -> Dict[str, List[Any]]:
        """Create parameter grid based on model type and desired complexity"""
        base_grids = {
            'random_forest': {
                'low': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5]
                },
                'medium': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'high': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'xgboost': {
                'low': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 6],
                    'learning_rate': [0.1, 0.3]
                },
                'medium': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0]
                },
                'high': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [3, 6, 9, 12],
                    'learning_rate': [0.001, 0.01, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 1.5, 2]
                }
            },
            'svm': {
                'low': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                'medium': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                'high': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],
                    'degree': [2, 3, 4]
                }
            }
        }
        
        if model_type not in base_grids:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if complexity not in ['low', 'medium', 'high']:
            raise ValueError(f"Unsupported complexity level: {complexity}")
        
        return base_grids[model_type][complexity]
    
    def compare_tuning_methods(self,
                             model: Any,
                             X: np.ndarray,
                             y: np.ndarray) -> Dict[str, TuningResult]:
        """Compare different tuning methods"""
        methods = ['grid', 'random', 'bayesian']
        results = {}
        
        original_config = self.config
        
        for method in methods:
            print(f"\n=== Tuning with {method} search ===")
            
            # Update config for current method
            self.config.tuning_method = method
            
            # Adjust parameters based on method
            if method == 'bayesian':
                self.config.n_trials = 20  # Reduced for comparison
            elif method == 'random':
                self.config.n_iter = 10  # Reduced for comparison
            
            try:
                result = self.tune_hyperparameters(model, X, y)
                results[method] = result
            except Exception as e:
                print(f"Tuning with {method} failed: {e}")
                results[method] = None
        
        # Restore original config
        self.config = original_config
        
        return results
    
    def get_tuning_recommendations(self, model_type: str, dataset_size: int) -> Dict[str, Any]:
        """Get recommendations for tuning based on model type and dataset size"""
        recommendations = {
            'preferred_method': 'random' if dataset_size > 10000 else 'grid',
            'cv_folds': min(5, dataset_size // 100),
            'max_iterations': min(100, dataset_size // 100),
            'notes': []
        }
        
        if dataset_size < 1000:
            recommendations['notes'].append("Consider using Bayesian optimization for small datasets")
            recommendations['preferred_method'] = 'bayesian'
        
        if model_type == 'neural_network':
            recommendations['notes'].append("For neural networks, consider using learning rate finder")
            recommendations['preferred_method'] = 'bayesian'
        
        return recommendations
