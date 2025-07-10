"""
Feature engineering module for FireSight.

This module creates features from raw weather and wildfire data,
including temporal features, rolling statistics, and fire risk indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

from ..utils.logger import LoggerMixin


class FeatureEngineer(LoggerMixin):
    """Engineer features for wildfire prediction models."""
    
    def __init__(self, config: Dict):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config['feature_engineering']
        self.paths = config['paths']
        
        # Create features directory
        Path(self.paths['features_data']).mkdir(parents=True, exist_ok=True)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def engineer_features(self, data: pd.DataFrame, region: str = "Bay Area") -> pd.DataFrame:
        """
        Engineer features from raw data.
        
        Args:
            data: Raw processed data
            region: Region name for file naming
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering...")
        
        if data.empty:
            self.logger.warning("No data provided for feature engineering")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Add rolling statistics
        df = self._add_rolling_features(df)
        
        # Add lag features
        df = self._add_lag_features(df)
        
        # Add fire risk indicators
        df = self._add_fire_risk_features(df)
        
        # Add seasonal features
        df = self._add_seasonal_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Save engineered features
        self._save_features(df, region)
        
        self.logger.info(f"Feature engineering completed. Final dataset: {df.shape}")
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the dataset."""
        self.logger.info("Adding temporal features...")
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic temporal features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['quarter'] = df['date'].dt.quarter
        
        # Year and year progress
        df['year'] = df['date'].dt.year
        df['year_progress'] = df['day_of_year'] / 365.25
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        self.logger.info("Adding rolling features...")
        
        # Weather columns for rolling statistics
        weather_cols = [
            'temperature_2m_max', 'temperature_2m_min',
            'relative_humidity_2m_max', 'relative_humidity_2m_min',
            'precipitation_sum', 'wind_speed_10m_max',
            'surface_pressure', 'evapotranspiration'
        ]
        
        # Filter to columns that exist in the dataset
        available_cols = [col for col in weather_cols if col in df.columns]
        
        if not available_cols:
            self.logger.warning("No weather columns found for rolling features")
            return df
        
        # Sort by date for proper rolling calculations
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add rolling statistics for different windows
        for window in self.feature_config['rolling_windows']:
            for col in available_cols:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}d'] = df[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min/max
                df[f'{col}_rolling_min_{window}d'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max_{window}d'] = df[col].rolling(window=window, min_periods=1).max()
                
                # Rolling trend (slope of linear fit)
                df[f'{col}_rolling_trend_{window}d'] = self._calculate_rolling_trend(df[col], window)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        self.logger.info("Adding lag features...")
        
        # Weather columns for lag features
        weather_cols = [
            'temperature_2m_max', 'temperature_2m_min',
            'relative_humidity_2m_max', 'relative_humidity_2m_min',
            'precipitation_sum', 'wind_speed_10m_max'
        ]
        
        # Filter to columns that exist
        available_cols = [col for col in weather_cols if col in df.columns]
        
        if not available_cols:
            self.logger.warning("No weather columns found for lag features")
            return df
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add lag features for different periods
        for lag in self.feature_config['lag_features']:
            for col in available_cols:
                df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        return df
    
    def _add_fire_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fire risk indicator features."""
        self.logger.info("Adding fire risk features...")
        
        # Temperature anomaly (deviation from seasonal average)
        if 'temperature_2m_max' in df.columns:
            df['temperature_anomaly'] = self._calculate_temperature_anomaly(df)
        
        # Humidity deficit (how much below optimal humidity)
        if 'relative_humidity_2m_min' in df.columns:
            df['humidity_deficit'] = 40 - df['relative_humidity_2m_min']  # Optimal humidity ~40%
            df['humidity_deficit'] = df['humidity_deficit'].clip(lower=0)
        
        # Wind dryness index (wind speed * temperature / humidity)
        if all(col in df.columns for col in ['wind_speed_10m_max', 'temperature_2m_max', 'relative_humidity_2m_min']):
            df['wind_dryness_index'] = (
                df['wind_speed_10m_max'] * df['temperature_2m_max'] / 
                (df['relative_humidity_2m_min'] + 1)  # Add 1 to avoid division by zero
            )
        
        # Precipitation deficit (days since last significant rain)
        if 'precipitation_sum' in df.columns:
            df['precipitation_deficit'] = self._calculate_precipitation_deficit(df)
        
        # Fire weather index (simplified version)
        df['fire_weather_index'] = self._calculate_fire_weather_index(df)
        
        return df
    
    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal and cyclical features."""
        self.logger.info("Adding seasonal features...")
        
        # Season encoding
        df['season'] = pd.cut(
            df['month'],
            bins=[0, 3, 6, 9, 12],
            labels=['winter', 'spring', 'summer', 'fall']
        )
        
        # One-hot encode seasons
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Cyclical encoding for day of year
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _calculate_rolling_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling trend (slope) for a series."""
        def linear_trend(x):
            if len(x) < 2:
                return np.nan
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return series.rolling(window=window, min_periods=2).apply(linear_trend)
    
    def _calculate_temperature_anomaly(self, df: pd.DataFrame) -> pd.Series:
        """Calculate temperature anomaly from seasonal average."""
        # Calculate seasonal average temperature
        seasonal_avg = df.groupby('month')['temperature_2m_max'].transform('mean')
        return df['temperature_2m_max'] - seasonal_avg
    
    def _calculate_precipitation_deficit(self, df: pd.DataFrame) -> pd.Series:
        """Calculate days since last significant precipitation."""
        # Define significant precipitation as > 5mm
        significant_rain = df['precipitation_sum'] > 5
        
        # Calculate cumulative days since last significant rain
        deficit = pd.Series(index=df.index, dtype=int)
        last_rain_day = -1
        
        for i in range(len(df)):
            if significant_rain.iloc[i]:
                last_rain_day = i
            deficit.iloc[i] = i - last_rain_day if last_rain_day >= 0 else 0
        
        return deficit
    
    def _calculate_fire_weather_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate a simplified fire weather index."""
        # This is a simplified version - real FWI is more complex
        fwi = pd.Series(0.0, index=df.index)
        
        # Temperature component
        if 'temperature_2m_max' in df.columns:
            temp_component = (df['temperature_2m_max'] - 10) / 20  # Normalize to 0-1
            temp_component = temp_component.clip(0, 1)
            fwi += temp_component
        
        # Humidity component
        if 'relative_humidity_2m_min' in df.columns:
            humidity_component = (100 - df['relative_humidity_2m_min']) / 100
            humidity_component = humidity_component.clip(0, 1)
            fwi += humidity_component
        
        # Wind component
        if 'wind_speed_10m_max' in df.columns:
            wind_component = df['wind_speed_10m_max'] / 50  # Normalize to 0-1
            wind_component = wind_component.clip(0, 1)
            fwi += wind_component
        
        # Precipitation deficit component
        if 'precipitation_deficit' in df.columns:
            deficit_component = df['precipitation_deficit'] / 30  # Normalize to 0-1
            deficit_component = deficit_component.clip(0, 1)
            fwi += deficit_component
        
        return fwi / 4  # Average of components
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        self.logger.info("Handling missing values...")
        
        # Count missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.info(f"Missing values found: {missing_counts[missing_counts > 0]}")
        
        # Forward fill for temporal data
        df = df.fillna(method='ffill')
        
        # Backward fill for remaining missing values
        df = df.fillna(method='bfill')
        
        # Fill any remaining missing values with 0 for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _save_features(self, df: pd.DataFrame, region: str) -> None:
        """Save engineered features to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_file = Path(self.paths['features_data']) / f"features_{region}_{timestamp}.csv"
        
        df.to_csv(features_file, index=False)
        self.logger.info(f"Saved engineered features to {features_file}")
        
        # Also save as latest version
        latest_file = Path(self.paths['features_data']) / f"latest_features_{region}.csv"
        df.to_csv(latest_file, index=False)
        self.logger.info(f"Saved latest features to {latest_file}")
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding target and metadata)."""
        exclude_cols = [
            'date', 'latitude', 'longitude', 'region',
            'wildfire_occurred', 'fire_severity', 'confidence', 'frp'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols 