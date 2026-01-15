import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
from dataclasses import dataclass
import warnings
import os
from ...model_io import save_model, load_model

@dataclass
class TrainingConfig:
    model_type: str
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    early_stopping_patience: int
    validation_metric: str

@dataclass
class TrainingResult:
    model: Any
    history: Optional[Dict[str, List[float]]]
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    training_time: float

class BehavioralModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = None
        
    def initialize_model(self) -> Any:
        """Initialize the model based on configuration"""
        model_type = self.config.model_type.lower()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.config.model_params)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**self.config.model_params)
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**self.config.model_params)
        elif model_type == 'svm':
            self.model = SVC(**self.config.model_params)
        elif model_type == 'neural_network':
            self.model = self._build_neural_network()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return self.model
    
    def _build_neural_network(self) -> tf.keras.Model:
        """Build a neural network model for behavioral analysis"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.config.model_params.get('input_dim', 100),)))
        
        # Hidden layers
        for units in self.config.model_params.get('hidden_layers', [128, 64, 32]):
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(self.config.model_params.get('dropout_rate', 0.3)))
        
        # Output layer
        output_units = self.config.model_params.get('output_units', 1)
        output_activation = self.config.model_params.get('output_activation', 'sigmoid')
        model.add(tf.keras.layers.Dense(output_units, activation=output_activation))
        
        # Compile model
        optimizer = self.config.model_params.get('optimizer', 'adam')
        loss = self.config.model_params.get('loss', 'binary_crossentropy')
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None) -> TrainingResult:
        """Train the model on the provided data"""
        import time
        
        start_time = time.time()
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        print(f"Training {self.config.model_type} model...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape if X_val is not None else 'None'}")
        
        training_history = None
        
        if self.config.model_type == 'neural_network':
            training_result = self._train_neural_network(X_train, y_train, X_val, y_val)
        else:
            training_result = self._train_sklearn_model(X_train, y_train, X_val, y_val)
        
        training_time = time.time() - start_time
        training_result.training_time = training_time
        
        self.is_trained = True
        self.training_history = training_result.history
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return training_result
    
    def _train_neural_network(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: Optional[np.ndarray],
                            y_val: Optional[np.ndarray]) -> TrainingResult:
        """Train neural network model"""
        callbacks = []
        
        # Early stopping
        if X_val is not None and y_val is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.config.training_params.get('epochs', 100),
            batch_size=self.config.training_params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Extract metrics
        training_metrics = {
            'final_train_loss': history.history['loss'][-1],
            'final_train_accuracy': history.history['accuracy'][-1] if 'accuracy' in history.history else None
        }
        
        if X_val is not None:
            training_metrics['final_val_loss'] = history.history['val_loss'][-1]
            if 'val_accuracy' in history.history:
                training_metrics['final_val_accuracy'] = history.history['val_accuracy'][-1]
        
        return TrainingResult(
            model=self.model,
            history=history.history,
            metrics=training_metrics,
            feature_importance=None,  # Neural networks don't have direct feature importance
            training_time=0.0  # Will be set by parent method
        )
    
    def _train_sklearn_model(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: Optional[np.ndarray],
                           y_val: Optional[np.ndarray]) -> TrainingResult:
        """Train scikit-learn model"""
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        metrics = {
            'train_accuracy': train_accuracy
        }
        
        # Calculate validation metrics if available
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            metrics['val_accuracy'] = val_accuracy
        
        # Extract feature importance if available
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(range(len(self.model.feature_importances_)), 
                                        self.model.feature_importances_))
        
        return TrainingResult(
            model=self.model,
            history=None,
            metrics=metrics,
            feature_importance=feature_importance,
            training_time=0.0  # Will be set by parent method
        )
    
    def evaluate_model(self, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate the trained model on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        if self.config.model_type == 'neural_network':
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
        
        # Feature importance analysis
        if feature_names is not None and hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
            evaluation_results['feature_importance'] = importance_dict
        
        return evaluation_results
    
    def cross_validate(self, 
                      X: np.ndarray, 
                      y: np.ndarray, 
                      cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation"""
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        if self.model is None:
            self.initialize_model()
        
        # Use stratified k-fold for classification
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.config.training_params.get('random_state', 42))
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')
        
        cv_results = {
            'cv_accuracy_mean': np.mean(cv_scores),
            'cv_accuracy_std': np.std(cv_scores),
            'cv_accuracy_scores': cv_scores.tolist(),
            'cv_folds': cv
        }
        
        return cv_results
    
    def hyperparameter_tuning(self, 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Perform hyperparameter tuning using grid search"""
        from sklearn.model_selection import GridSearchCV
        
        if self.model is None:
            self.initialize_model()
        
        print("Performing hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        return tuning_results
    
    def save_model(self, model_obj, path: str, metadata: dict = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_model(path, model_obj, metadata=metadata)

    def load_model(self, path: str):
        model_obj, meta = load_model(path)
        return model_obj, meta
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the trained model"""
        if not self.is_trained:
            return {'status': 'Model not trained'}
        
        summary = {
            'model_type': self.config.model_type,
            'is_trained': self.is_trained,
            'training_config': self.config.training_params
        }
        
        if self.training_history:
            summary['training_history_keys'] = list(self.training_history.keys())
        
        return summary
