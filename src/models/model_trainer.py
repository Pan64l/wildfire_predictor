"""
Model training module for FireSight.

This module handles training of classification and regression models
for wildfire prediction and severity estimation.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

from ..utils.logger import LoggerMixin
from ..feature_engineering.feature_engineer import FeatureEngineer


class ModelTrainer(LoggerMixin):
    """Train and evaluate wildfire prediction models."""
    
    def __init__(self, config: Dict):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config['models']
        self.eval_config = config['evaluation']
        self.paths = config['paths']
        
        # Create model directories
        self._create_directories()
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(config)
        
        # Initialize models
        self.classification_models = {}
        self.regression_models = {}
        self.scalers = {}
        
        # Load latest data
        self.data = self._load_latest_data()
    
    def _create_directories(self) -> None:
        """Create necessary model directories."""
        directories = [
            self.paths['model_files']['classification'],
            self.paths['model_files']['regression'],
            self.paths['results_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_latest_data(self) -> pd.DataFrame:
        """Load the latest processed data."""
        # Try to load latest features first
        features_file = Path(self.paths['features_data']) / "latest_Bay Area.csv"
        if features_file.exists():
            self.logger.info(f"Loading latest features from {features_file}")
            return pd.read_csv(features_file)
        
        # Fall back to processed data
        processed_file = Path(self.paths['processed_data']) / "latest_Bay Area.csv"
        if processed_file.exists():
            self.logger.info(f"Loading latest processed data from {processed_file}")
            data = pd.read_csv(processed_file)
            # Engineer features
            return self.feature_engineer.engineer_features(data, "Bay Area")
        
        self.logger.warning("No data found. Please run data collection first.")
        return pd.DataFrame()
    
    def train_classification_models(self) -> None:
        """Train classification models for wildfire occurrence prediction."""
        self.logger.info("Training classification models...")
        
        if self.data.empty:
            self.logger.error("No data available for training")
            return
        
        # Prepare data
        X, y = self._prepare_classification_data()
        
        if X.empty or y.empty:
            self.logger.error("Failed to prepare classification data")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.eval_config['test_size'],
            random_state=self.eval_config['random_state'],
            stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['classification'] = scaler
        
        # Train models
        models_config = self.model_config['classification']['algorithms']
        
        for model_name, params in models_config.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                model = self._create_classification_model(model_name, params)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                metrics = self._evaluate_classification_model(y_test, y_pred, y_pred_proba)
                
                # Store model and metrics
                self.classification_models[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_names': X.columns.tolist()
                }
                
                # Save model
                self._save_model(model, scaler, model_name, 'classification')
                
                self.logger.info(f"{model_name} training completed. F1 Score: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
    
    def train_regression_models(self) -> None:
        """Train regression models for fire severity prediction."""
        self.logger.info("Training regression models...")
        
        if self.data.empty:
            self.logger.error("No data available for training")
            return
        
        # Prepare data
        X, y = self._prepare_regression_data()
        
        if X.empty or y.empty:
            self.logger.error("Failed to prepare regression data")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.eval_config['test_size'],
            random_state=self.eval_config['random_state']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['regression'] = scaler
        
        # Train models
        models_config = self.model_config['regression']['algorithms']
        
        for model_name, params in models_config.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                model = self._create_regression_model(model_name, params)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                metrics = self._evaluate_regression_model(y_test, y_pred)
                
                # Store model and metrics
                self.regression_models[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_names': X.columns.tolist()
                }
                
                # Save model
                self._save_model(model, scaler, model_name, 'regression')
                
                self.logger.info(f"{model_name} training completed. R² Score: {metrics['r2_score']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
    
    def _prepare_classification_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for classification training."""
        target_col = self.model_config['classification']['target_column']
        
        if target_col not in self.data.columns:
            self.logger.error(f"Target column '{target_col}' not found in data")
            return pd.DataFrame(), pd.Series()
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns(self.data)
        
        if not feature_cols:
            self.logger.error("No feature columns found")
            return pd.DataFrame(), pd.Series()
        
        # Remove rows with missing target values
        data_clean = self.data.dropna(subset=[target_col])
        
        X = data_clean[feature_cols]
        y = data_clean[target_col]
        
        self.logger.info(f"Classification data prepared: {X.shape}, target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _prepare_regression_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for regression training."""
        target_col = self.model_config['regression']['target_column']
        
        if target_col not in self.data.columns:
            self.logger.error(f"Target column '{target_col}' not found in data")
            return pd.DataFrame(), pd.Series()
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns(self.data)
        
        if not feature_cols:
            self.logger.error("No feature columns found")
            return pd.DataFrame(), pd.Series()
        
        # Remove rows with missing target values
        data_clean = self.data.dropna(subset=[target_col])
        
        X = data_clean[feature_cols]
        y = data_clean[target_col]
        
        self.logger.info(f"Regression data prepared: {X.shape}, target stats: mean={y.mean():.2f}, std={y.std():.2f}")
        
        return X, y
    
    def _create_classification_model(self, model_name: str, params: Dict) -> Any:
        """Create a classification model instance."""
        if model_name == 'logistic_regression':
            return LogisticRegression(**params, random_state=self.eval_config['random_state'])
        
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params, random_state=self.eval_config['random_state'])
        
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(**params, random_state=self.eval_config['random_state'])
        
        elif model_name == 'adaboost':
            return AdaBoostClassifier(**params, random_state=self.eval_config['random_state'])
        
        elif model_name == 'tabnet':
            # TabNet requires special handling
            from pytorch_tabnet.tab_model import TabNetClassifier
            return TabNetClassifier(**params)
        
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
    
    def _create_regression_model(self, model_name: str, params: Dict) -> Any:
        """Create a regression model instance."""
        if model_name == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)
        
        elif model_name == 'svr':
            return SVR(**params)
        
        elif model_name == 'random_forest_regressor':
            return RandomForestRegressor(**params, random_state=self.eval_config['random_state'])
        
        elif model_name == 'xgboost_regressor':
            return xgb.XGBRegressor(**params, random_state=self.eval_config['random_state'])
        
        else:
            raise ValueError(f"Unknown regression model: {model_name}")
    
    def _evaluate_classification_model(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict:
        """Evaluate classification model performance."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def _evaluate_regression_model(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Evaluate regression model performance."""
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'mean_squared_error': mean_squared_error(y_true, y_pred),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        return metrics
    
    def _save_model(self, model: Any, scaler: StandardScaler, model_name: str, model_type: str) -> None:
        """Save trained model and scaler."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(self.paths['model_files'][model_type])
        
        # Save model
        model_file = model_dir / f"{model_name}_{timestamp}.pkl"
        joblib.dump(model, model_file)
        
        # Save scaler
        scaler_file = model_dir / f"{model_name}_scaler_{timestamp}.pkl"
        joblib.dump(scaler, scaler_file)
        
        # Save as latest
        latest_model_file = model_dir / f"latest_{model_name}.pkl"
        latest_scaler_file = model_dir / f"latest_{model_name}_scaler.pkl"
        
        joblib.dump(model, latest_model_file)
        joblib.dump(scaler, latest_scaler_file)
        
        self.logger.info(f"Saved {model_name} model to {model_file}")
    
    def evaluate_classification_models(self) -> pd.DataFrame:
        """Evaluate all trained classification models."""
        if not self.classification_models:
            self.logger.warning("No classification models trained yet")
            return pd.DataFrame()
        
        results = []
        for model_name, model_info in self.classification_models.items():
            metrics = model_info['metrics']
            results.append({
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics.get('roc_auc', np.nan)
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        return results_df
    
    def evaluate_regression_models(self) -> pd.DataFrame:
        """Evaluate all trained regression models."""
        if not self.regression_models:
            self.logger.warning("No regression models trained yet")
            return pd.DataFrame()
        
        results = []
        for model_name, model_info in self.regression_models.items():
            metrics = model_info['metrics']
            results.append({
                'model': model_name,
                'r2_score': metrics['r2_score'],
                'mae': metrics['mean_absolute_error'],
                'mse': metrics['mean_squared_error'],
                'rmse': metrics['root_mean_squared_error']
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('r2_score', ascending=False)
        
        return results_df 