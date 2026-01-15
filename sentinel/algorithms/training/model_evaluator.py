import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings

@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]

@dataclass
class ModelComparison:
    model_names: List[str]
    metrics_comparison: pd.DataFrame
    best_model: str
    statistical_significance: Dict[str, float]

class ModelEvaluator:
    def __init__(self, target_names: Optional[List[str]] = None):
        self.target_names = target_names
        self.evaluation_history = []
        
    def comprehensive_evaluation(self,
                               model: Any,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               X_train: Optional[np.ndarray] = None,
                               y_train: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """Perform comprehensive model evaluation"""
        
        # Predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate AUC scores if probabilities are available
        roc_auc = None
        pr_auc = None
        
        if y_pred_proba is not None:
            if len(np.unique(y_test)) == 2:  # Binary classification
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
                pr_auc = np.trapz(recall_vals, precision_vals)
            else:  # Multiclass classification
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=self.target_names,
                                           output_dict=True,
                                           zero_division=0)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            confusion_matrix=cm,
            classification_report=class_report
        )
    
    def plot_confusion_matrix(self, 
                            cm: np.ndarray,
                            title: str = "Confusion Matrix",
                            figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix"""
        plt.figure(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_pred_proba: np.ndarray,
                      model_name: str = "Model"):
        """Plot ROC curve"""
        if len(np.unique(y_true)) != 2:
            print("ROC curve is only for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curve(self,
                                  y_true: np.ndarray,
                                  y_pred_proba: np.ndarray,
                                  model_name: str = "Model"):
        """Plot precision-recall curve"""
        if len(np.unique(y_true)) != 2:
            print("Precision-recall curve is only for binary classification")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        pr_auc = np.trapz(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()
    
    def plot_calibration_curve(self,
                             y_true: np.ndarray,
                             y_pred_proba: np.ndarray,
                             model_name: str = "Model"):
        """Plot calibration curve"""
        if len(np.unique(y_true)) != 2:
            print("Calibration curve is only for binary classification")
            return
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba[:, 1], n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=model_name)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel('Mean predicted value')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def feature_importance_analysis(self, 
                                  model: Any,
                                  feature_names: List[str],
                                  top_n: int = 15) -> pd.DataFrame:
        """Analyze and plot feature importance"""
        importance_df = pd.DataFrame()
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
            
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) == 1:
                # Binary classification
                coefficients = model.coef_
            else:
                # Multiclass classification - use mean absolute coefficient
                coefficients = np.mean(np.abs(model.coef_), axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coefficients)
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot coefficients
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Coefficients (Absolute Values)')
            plt.tight_layout()
            plt.show()
        
        return importance_df
    
    def learning_curve_analysis(self,
                              model: Any,
                              X: np.ndarray,
                              y: np.ndarray,
                              cv: int = 5):
        """Plot learning curve"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
    
    def compare_models(self,
                     models: Dict[str, Any],
                     X_test: np.ndarray,
                     y_test: np.ndarray) -> ModelComparison:
        """Compare multiple models"""
        comparison_results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            metrics = self.comprehensive_evaluation(model, X_test, y_test)
            
            comparison_results.append({
                'model': model_name,
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'roc_auc': metrics.roc_auc if metrics.roc_auc else np.nan,
                'pr_auc': metrics.pr_auc if metrics.pr_auc else np.nan
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        # Determine best model
        best_model = comparison_df.iloc[0]['model']
        
        # Statistical significance testing (simplified)
        statistical_significance = self._calculate_statistical_significance(models, X_test, y_test)
        
        return ModelComparison(
            model_names=list(models.keys()),
            metrics_comparison=comparison_df,
            best_model=best_model,
            statistical_significance=statistical_significance
        )
    
    def _calculate_statistical_significance(self,
                                          models: Dict[str, Any],
                                          X_test: np.ndarray,
                                          y_test: np.ndarray) -> Dict[str, float]:
        """Calculate statistical significance between models (simplified)"""
        # This is a simplified implementation
        # In practice, you might use McNemar's test or paired t-test
        
        scores = {}
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            scores[model_name] = accuracy_score(y_test, y_pred)
        
        # Simple difference-based significance
        significance = {}
        model_names = list(models.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    key = f"{model1}_vs_{model2}"
                    diff = abs(scores[model1] - scores[model2])
                    # Simple heuristic: difference > 0.02 is significant
                    significance[key] = 1.0 if diff > 0.02 else 0.0
        
        return significance
    
    def generate_evaluation_report(self,
                                 metrics: EvaluationMetrics,
                                 model_name: str = "Model") -> str:
        """Generate a comprehensive evaluation report"""
        report = f"""
# Model Evaluation Report: {model_name}

## Performance Metrics
- Accuracy: {metrics.accuracy:.4f}
- Precision: {metrics.precision:.4f}
- Recall: {metrics.recall:.4f}
- F1-Score: {metrics.f1_score:.4f}
- ROC AUC: {metrics.roc_auc if metrics.roc_auc else 'N/A':.4f}
- PR AUC: {metrics.pr_auc if metrics.pr_auc else 'N/A':.4f}

## Detailed Classification Report
"""
        
        # Add classification report details
        for class_name, class_metrics in metrics.classification_report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                report += f"\n**{class_name.title()}**:\n"
                report += f"  Precision: {class_metrics.get('precision', 0):.4f}\n"
                report += f"  Recall: {class_metrics.get('recall', 0):.4f}\n"
                report += f"  F1-Score: {class_metrics.get('f1-score', 0):.4f}\n"
                report += f"  Support: {class_metrics.get('support', 0)}\n"
        
        return report
    
    def save_evaluation_results(self,
                              metrics: EvaluationMetrics,
                              filepath: str,
                              model_name: str = "Model"):
        """Save evaluation results to file"""
        import json
        
        results = {
            'model_name': model_name,
            'metrics': {
                'accuracy': float(metrics.accuracy),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1_score': float(metrics.f1_score),
                'roc_auc': float(metrics.roc_auc) if metrics.roc_auc else None,
                'pr_auc': float(metrics.pr_auc) if metrics.pr_auc else None
            },
            'confusion_matrix': metrics.confusion_matrix.tolist(),
            'classification_report': metrics.classification_report
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
