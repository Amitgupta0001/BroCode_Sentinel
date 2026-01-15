import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from dataclasses import dataclass
import warnings

@dataclass
class PreprocessingConfig:
    feature_columns: List[str]
    target_column: str
    categorical_columns: List[str]
    numerical_columns: List[str]
    test_size: float
    random_state: int
    scaling_method: str

class BehavioralDataPreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = []
        self.is_fitted = False
        
    def load_and_validate_data(self, filepath: str) -> pd.DataFrame:
        """Load behavioral data from file and validate structure"""
        try:
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
            elif filepath.endswith('.parquet'):
                data = pd.read_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # Validate required columns
            self._validate_data_columns(data)
            
            # Check for missing values
            self._check_missing_values(data)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error loading data from {filepath}: {e}")
    
    def _validate_data_columns(self, data: pd.DataFrame):
        """Validate that required columns are present"""
        missing_columns = []
        
        for column in self.config.feature_columns + [self.config.target_column]:
            if column not in data.columns:
                missing_columns.append(column)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _check_missing_values(self, data: pd.DataFrame):
        """Check for and handle missing values"""
        missing_count = data.isnull().sum()
        missing_columns = missing_count[missing_count > 0]
        
        if len(missing_columns) > 0:
            print(f"Warning: Found missing values in columns: {list(missing_columns.index)}")
            print("Missing value counts:")
            for col, count in missing_columns.items():
                print(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
    
    def preprocess_dataset(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess the entire dataset"""
        # Handle missing values
        data_clean = self._handle_missing_values(data)
        
        # Encode categorical variables
        data_encoded = self._encode_categorical_features(data_clean)
        
        # Scale numerical features
        data_scaled = self._scale_numerical_features(data_encoded)
        
        # Extract features and target
        X, y, feature_names = self._extract_features_target(data_scaled)
        
        self.is_fitted = True
        self.feature_names = feature_names
        
        return X, y, feature_names
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        data_clean = data.copy()
        
        for column in data_clean.columns:
            if data_clean[column].isnull().sum() > 0:
                if column in self.config.categorical_columns:
                    # For categorical, use mode
                    mode_value = data_clean[column].mode()
                    if len(mode_value) > 0:
                        data_clean[column].fillna(mode_value[0], inplace=True)
                    else:
                        data_clean[column].fillna('unknown', inplace=True)
                else:
                    # For numerical, use median
                    data_clean[column].fillna(data_clean[column].median(), inplace=True)
        
        return data_clean
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding"""
        data_encoded = data.copy()
        
        for column in self.config.categorical_columns:
            if column in data_encoded.columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    data_encoded[column] = self.label_encoders[column].fit_transform(data_encoded[column])
                else:
                    # Transform using fitted encoder
                    data_encoded[column] = self.label_encoders[column].transform(data_encoded[column])
        
        return data_encoded
    
    def _scale_numerical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        data_scaled = data.copy()
        
        for column in self.config.numerical_columns:
            if column in data_scaled.columns:
                if column not in self.scalers:
                    if self.config.scaling_method == 'standard':
                        self.scalers[column] = StandardScaler()
                    else:
                        raise ValueError(f"Unsupported scaling method: {self.config.scaling_method}")
                    
                    data_scaled[column] = self.scalers[column].fit_transform(
                        data_scaled[column].values.reshape(-1, 1)
                    ).flatten()
                else:
                    # Transform using fitted scaler
                    data_scaled[column] = self.scalers[column].transform(
                        data_scaled[column].values.reshape(-1, 1)
                    ).flatten()
        
        return data_scaled
    
    def _extract_features_target(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features and target variables"""
        X = data[self.config.feature_columns].values
        y = data[self.config.target_column].values
        feature_names = self.config.feature_columns
        
        return X, y, feature_names
    
    def split_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Split dataset into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Second split: separate validation set from temp
        val_size = self.config.test_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.random_state,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_sequences(self, 
                       data: np.ndarray, 
                       sequence_length: int, 
                       target: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for temporal models"""
        sequences = []
        targets = [] if target is not None else None
        
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
            if target is not None:
                # Use the target of the last element in the sequence
                targets.append(target[i + sequence_length - 1])
        
        sequences_array = np.array(sequences)
        if targets is not None:
            targets_array = np.array(targets)
            return sequences_array, targets_array
        else:
            return sequences_array, None
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray, method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """Balance imbalanced dataset"""
        from collections import Counter
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        
        class_counts = Counter(y)
        print(f"Original class distribution: {class_counts}")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.config.random_state)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=self.config.random_state)
            X_balanced, y_balanced = rus.fit_resample(X, y)
        elif method == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=self.config.random_state)
            X_balanced, y_balanced = ros.fit_resample(X, y)
        else:
            raise ValueError(f"Unsupported balancing method: {method}")
        
        balanced_counts = Counter(y_balanced)
        print(f"Balanced class distribution: {balanced_counts}")
        
        return X_balanced, y_balanced
    
    def save_preprocessor(self, filepath: str):
        """Save fitted preprocessor to file"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        save_data = {
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor from file"""
        load_data = joblib.load(filepath)
        
        self.label_encoders = load_data['label_encoders']
        self.scalers = load_data['scalers']
        self.feature_names = load_data['feature_names']
        self.config = load_data['config']
        self.is_fitted = load_data['is_fitted']
    
    def transform_new_data(self, new_data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        # Apply the same preprocessing steps
        data_clean = self._handle_missing_values(new_data)
        data_encoded = self._encode_categorical_features(data_clean)
        data_scaled = self._scale_numerical_features(data_encoded)
        
        # Extract features
        X_new = data_scaled[self.config.feature_columns].values
        
        return X_new
    
    def get_feature_importance_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform basic feature importance analysis"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
        rf.fit(X, y)
        rf_importance = dict(zip(self.feature_names, rf.feature_importances_))
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=self.config.random_state)
        mi_importance = dict(zip(self.feature_names, mi_scores))
        
        return {
            'random_forest_importance': rf_importance,
            'mutual_information': mi_importance
        }
