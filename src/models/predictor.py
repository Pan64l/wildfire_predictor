"""
Prediction module for FireSight.

This module handles loading trained models and making predictions
for wildfire occurrence and severity.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import joblib

from ..utils.logger import LoggerMixin


class WildfirePredictor(LoggerMixin):
    """Make wildfire predictions using trained models."""
    
    def __init__(self, config: Dict):
        """
        Initialize the wildfire predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = config['paths']
        self.data_sources = config['data_sources']
        
        # Load trained models
        self.classification_model = None
        self.regression_model = None
        self.classification_scaler = None
        self.regression_scaler = None
        self.feature_names = None
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load the best trained models."""
        classification_dir = Path(self.paths['model_files']['classification'])
        regression_dir = Path(self.paths['model_files']['regression'])
        
        # Find the best classification model
        best_classification_model = self._find_best_model(classification_dir, 'classification')
        if best_classification_model:
            self.classification_model = joblib.load(best_classification_model)
            scaler_file = best_classification_model.parent / f"{best_classification_model.stem}_scaler.pkl"
            if scaler_file.exists():
                self.classification_scaler = joblib.load(scaler_file)
            self.logger.info(f"Loaded classification model: {best_classification_model.name}")
        
        # Find the best regression model
        best_regression_model = self._find_best_model(regression_dir, 'regression')
        if best_regression_model:
            self.regression_model = joblib.load(best_regression_model)
            scaler_file = best_regression_model.parent / f"{best_regression_model.stem}_scaler.pkl"
            if scaler_file.exists():
                self.regression_scaler = joblib.load(scaler_file)
            self.logger.info(f"Loaded regression model: {best_regression_model.name}")
        
        if not self.classification_model and not self.regression_model:
            self.logger.warning("No trained models found. Please train models first.")
    
    def _find_best_model(self, model_dir: Path, model_type: str) -> Optional[Path]:
        """Find the best performing model based on metrics."""
        if not model_dir.exists():
            return None
        
        # Look for latest models first
        latest_models = list(model_dir.glob("latest_*.pkl"))
        if latest_models:
            return latest_models[0]
        
        # Fall back to any available model
        all_models = list(model_dir.glob("*.pkl"))
        if all_models:
            return all_models[0]
        
        return None
    
    def predict(self, latitude: float, longitude: float, date: Optional[str] = None) -> Dict:
        """
        Make wildfire prediction for a specific location and date.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            date: Date for prediction (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.classification_model and not self.regression_model:
            raise ValueError("No trained models available for prediction")
        
        # Set default date to today if not provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Making prediction for location ({latitude}, {longitude}) on {date}")
        
        try:
            # Get weather data for the location and date
            weather_data = self._get_weather_data(latitude, longitude, date)
            
            if weather_data.empty:
                raise ValueError("Could not retrieve weather data for the specified location and date")
            
            # Engineer features
            features = self._engineer_features_for_prediction(weather_data)
            
            # Make predictions
            prediction = {
                'date': date,
                'latitude': latitude,
                'longitude': longitude,
                'wildfire_risk': 0.0,
                'fire_severity': 0.0,
                'confidence': 0.0
            }
            
            # Classification prediction
            if self.classification_model and self.classification_scaler:
                wildfire_risk = self._predict_wildfire_occurrence(features)
                prediction['wildfire_risk'] = wildfire_risk
            
            # Regression prediction
            if self.regression_model and self.regression_scaler:
                fire_severity = self._predict_fire_severity(features)
                prediction['fire_severity'] = fire_severity
            
            # Calculate confidence based on model agreement
            prediction['confidence'] = self._calculate_confidence(features)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def _get_weather_data(self, latitude: float, longitude: float, date: str) -> pd.DataFrame:
        """
        Get weather data for a specific location and date.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            date: Date for weather data
            
        Returns:
            DataFrame containing weather data
        """
        # Calculate date range (7 days before to capture lag features)
        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = date
        
        # OpenMeteo API parameters
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'timezone': self.data_sources['openmeteo']['default_timezone'],
            'daily': ','.join(self.config['data_collection']['weather_features'])
        }
        
        url = self.data_sources['openmeteo']['base_url']
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'daily' not in data:
                raise ValueError("No weather data found in API response")
            
            # Convert to DataFrame
            weather_df = pd.DataFrame(data['daily'])
            
            # Convert date column
            weather_df['date'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.drop('time', axis=1)
            
            # Add location information
            weather_df['latitude'] = latitude
            weather_df['longitude'] = longitude
            weather_df['region'] = 'Prediction'
            
            return weather_df
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch weather data: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing weather data: {e}")
            raise
    
    def _engineer_features_for_prediction(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for prediction from weather data.
        
        Args:
            weather_data: Raw weather data
            
        Returns:
            DataFrame with engineered features
        """
        from ..feature_engineering.feature_engineer import FeatureEngineer
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(self.config)
        
        # Engineer features
        features = feature_engineer.engineer_features(weather_data, "Prediction")
        
        # Get the most recent row (for current prediction)
        if not features.empty:
            features = features.iloc[-1:].copy()
        
        return features
    
    def _predict_wildfire_occurrence(self, features: pd.DataFrame) -> float:
        """
        Predict wildfire occurrence probability.
        
        Args:
            features: Engineered features
            
        Returns:
            Probability of wildfire occurrence (0-1)
        """
        if self.classification_model is None or self.classification_scaler is None:
            return 0.0
        
        try:
            # Get feature columns that the model expects
            model_features = self._get_model_features(features, 'classification')
            
            if model_features.empty:
                self.logger.warning("No matching features found for classification model")
                return 0.0
            
            # Scale features
            features_scaled = self.classification_scaler.transform(model_features)
            
            # Make prediction
            if hasattr(self.classification_model, 'predict_proba'):
                proba = self.classification_model.predict_proba(features_scaled)
                return proba[0, 1]  # Probability of positive class
            else:
                prediction = self.classification_model.predict(features_scaled)
                return float(prediction[0])
                
        except Exception as e:
            self.logger.error(f"Classification prediction failed: {e}")
            return 0.0
    
    def _predict_fire_severity(self, features: pd.DataFrame) -> float:
        """
        Predict fire severity.
        
        Args:
            features: Engineered features
            
        Returns:
            Predicted fire severity
        """
        if self.regression_model is None or self.regression_scaler is None:
            return 0.0
        
        try:
            # Get feature columns that the model expects
            model_features = self._get_model_features(features, 'regression')
            
            if model_features.empty:
                self.logger.warning("No matching features found for regression model")
                return 0.0
            
            # Scale features
            features_scaled = self.regression_scaler.transform(model_features)
            
            # Make prediction
            prediction = self.regression_model.predict(features_scaled)
            return float(prediction[0])
            
        except Exception as e:
            self.logger.error(f"Regression prediction failed: {e}")
            return 0.0
    
    def _get_model_features(self, features: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """
        Get features that match what the model expects.
        
        Args:
            features: Available features
            model_type: Type of model ('classification' or 'regression')
            
        Returns:
            DataFrame with matching features
        """
        # For now, we'll use all available features
        # In a production system, you'd want to store and use the exact feature names
        # that were used during training
        
        # Remove non-feature columns
        exclude_cols = [
            'date', 'latitude', 'longitude', 'region',
            'wildfire_occurred', 'fire_severity', 'confidence', 'frp'
        ]
        
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        if not feature_cols:
            return pd.DataFrame()
        
        return features[feature_cols]
    
    def _calculate_confidence(self, features: pd.DataFrame) -> float:
        """
        Calculate prediction confidence based on data quality and model agreement.
        
        Args:
            features: Engineered features
            
        Returns:
            Confidence score (0-1)
        """
        # Simple confidence calculation based on data completeness
        # In a more sophisticated system, you might consider:
        # - Model uncertainty estimates
        # - Data quality metrics
        # - Historical prediction accuracy
        
        if features.empty:
            return 0.0
        
        # Calculate data completeness
        completeness = 1 - features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
        
        # Base confidence on data completeness
        confidence = min(completeness, 0.95)  # Cap at 95%
        
        return confidence
    
    def predict_batch(self, locations: List[Tuple[float, float]], date: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions for multiple locations.
        
        Args:
            locations: List of (latitude, longitude) tuples
            date: Date for prediction (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with predictions for all locations
        """
        predictions = []
        
        for lat, lon in locations:
            try:
                prediction = self.predict(lat, lon, date)
                predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"Failed to predict for location ({lat}, {lon}): {e}")
                # Add default prediction
                predictions.append({
                    'date': date or datetime.now().strftime("%Y-%m-%d"),
                    'latitude': lat,
                    'longitude': lon,
                    'wildfire_risk': 0.0,
                    'fire_severity': 0.0,
                    'confidence': 0.0
                })
        
        return pd.DataFrame(predictions) 