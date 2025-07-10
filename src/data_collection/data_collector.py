"""
Data collection module for FireSight.

This module handles collection of wildfire data from NASA FIRMS
and weather data from OpenMeteo API.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from ..utils.logger import LoggerMixin


class DataCollector(LoggerMixin):
    """Collect wildfire and weather data for FireSight."""
    
    def __init__(self, config: Dict):
        """
        Initialize the data collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_sources = config['data_sources']
        self.data_config = config['data_collection']
        self.paths = config['paths']
        
        # Create data directories
        self._create_directories()
        
        # Initialize API keys
        self.nasa_api_key = os.getenv('NASA_FIRMS_API_KEY')
        if not self.nasa_api_key:
            self.logger.warning("NASA FIRMS API key not found in environment variables")
    
    def _create_directories(self) -> None:
        """Create necessary data directories."""
        directories = [
            self.paths['raw_data'],
            self.paths['processed_data'],
            self.paths['features_data']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def collect_all_data(self, region: str = "Bay Area") -> None:
        """
        Collect all required data for the specified region.
        
        Args:
            region: Target region for data collection
        """
        self.logger.info(f"Starting data collection for region: {region}")
        
        # Find region configuration
        region_config = self._get_region_config(region)
        if not region_config:
            raise ValueError(f"Region '{region}' not found in configuration")
        
        # Collect wildfire data
        self.logger.info("Collecting wildfire data from NASA FIRMS...")
        wildfire_data = self.collect_wildfire_data(region_config)
        
        # Collect weather data
        self.logger.info("Collecting weather data from OpenMeteo...")
        weather_data = self.collect_weather_data(region_config)
        
        # Save raw data
        self._save_raw_data(wildfire_data, weather_data, region)
        
        # Process and merge data
        self.logger.info("Processing and merging data...")
        merged_data = self.process_and_merge_data(wildfire_data, weather_data)
        
        # Save processed data
        self._save_processed_data(merged_data, region)
        
        self.logger.info("Data collection completed successfully!")
    
    def collect_wildfire_data(self, region_config: Dict) -> pd.DataFrame:
        """
        Collect wildfire data from NASA FIRMS.
        
        Args:
            region_config: Configuration for the target region
            
        Returns:
            DataFrame containing wildfire data
        """
        if not self.nasa_api_key:
            self.logger.warning("Skipping NASA FIRMS data collection - no API key")
            return pd.DataFrame()
        
        bounds = region_config['bounds']
        start_date = self.data_config['start_date']
        end_date = self.data_config['end_date']
        
        # NASA FIRMS API parameters
        params = {
            'source': 'VIIRS_SNPP_NRT',
            'area': f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
            'date': f"{start_date}/{end_date}",
            'type': 'csv'
        }
        
        url = self.data_sources['nasa_firms']['base_url']
        
        try:
            self.logger.info(f"Fetching wildfire data from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse CSV data
            data = pd.read_csv(pd.StringIO(response.text))
            
            if data.empty:
                self.logger.warning("No wildfire data found for the specified region and time period")
                return pd.DataFrame()
            
            # Clean and standardize column names
            data.columns = data.columns.str.lower().str.replace(' ', '_')
            
            # Convert timestamp
            if 'acq_date' in data.columns and 'acq_time' in data.columns:
                data['timestamp'] = pd.to_datetime(
                    data['acq_date'] + ' ' + data['acq_time'], 
                    format='%Y-%m-%d %H%M'
                )
            
            # Add region information
            data['region'] = region_config['name']
            
            self.logger.info(f"Collected {len(data)} wildfire records")
            return data
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch wildfire data: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error processing wildfire data: {e}")
            return pd.DataFrame()
    
    def collect_weather_data(self, region_config: Dict) -> pd.DataFrame:
        """
        Collect weather data from OpenMeteo API.
        
        Args:
            region_config: Configuration for the target region
            
        Returns:
            DataFrame containing weather data
        """
        bounds = region_config['bounds']
        start_date = self.data_config['start_date']
        end_date = self.data_config['end_date']
        
        # Calculate centroid for weather data
        lat = (bounds[0] + bounds[2]) / 2
        lon = (bounds[1] + bounds[3]) / 2
        
        # OpenMeteo API parameters
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'timezone': self.data_sources['openmeteo']['default_timezone'],
            'daily': ','.join(self.data_config['weather_features'])
        }
        
        url = self.data_sources['openmeteo']['base_url']
        
        try:
            self.logger.info(f"Fetching weather data for coordinates ({lat}, {lon})")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'daily' not in data:
                self.logger.warning("No weather data found in API response")
                return pd.DataFrame()
            
            # Convert to DataFrame
            weather_df = pd.DataFrame(data['daily'])
            
            # Convert date column
            weather_df['date'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.drop('time', axis=1)
            
            # Add location information
            weather_df['latitude'] = lat
            weather_df['longitude'] = lon
            weather_df['region'] = region_config['name']
            
            self.logger.info(f"Collected {len(weather_df)} weather records")
            return weather_df
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch weather data: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error processing weather data: {e}")
            return pd.DataFrame()
    
    def process_and_merge_data(self, wildfire_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and merge wildfire and weather data.
        
        Args:
            wildfire_data: Wildfire occurrence data
            weather_data: Weather condition data
            
        Returns:
            Merged and processed DataFrame
        """
        if wildfire_data.empty or weather_data.empty:
            self.logger.warning("Cannot merge data - one or both datasets are empty")
            return pd.DataFrame()
        
        # Create wildfire occurrence indicator
        wildfire_data['wildfire_occurred'] = 1
        
        # Group wildfires by date and location
        wildfire_daily = wildfire_data.groupby([
            wildfire_data['timestamp'].dt.date,
            'latitude', 
            'longitude'
        ]).agg({
            'wildfire_occurred': 'max',
            'confidence': 'mean',
            'frp': 'sum'  # Fire Radiative Power
        }).reset_index()
        
        wildfire_daily['date'] = pd.to_datetime(wildfire_daily['timestamp'])
        wildfire_daily = wildfire_daily.drop('timestamp', axis=1)
        
        # Merge with weather data
        merged_data = weather_data.merge(
            wildfire_daily,
            on=['date', 'latitude', 'longitude'],
            how='left'
        )
        
        # Fill missing wildfire indicators
        merged_data['wildfire_occurred'] = merged_data['wildfire_occurred'].fillna(0)
        merged_data['confidence'] = merged_data['confidence'].fillna(0)
        merged_data['frp'] = merged_data['frp'].fillna(0)
        
        # Calculate fire severity (0 for no fire, FRP for fires)
        merged_data['fire_severity'] = merged_data['frp']
        
        # Sort by date
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
        
        self.logger.info(f"Merged dataset contains {len(merged_data)} records")
        self.logger.info(f"Wildfire occurrence rate: {merged_data['wildfire_occurred'].mean():.2%}")
        
        return merged_data
    
    def _get_region_config(self, region_name: str) -> Optional[Dict]:
        """
        Get configuration for a specific region.
        
        Args:
            region_name: Name of the region
            
        Returns:
            Region configuration dictionary or None if not found
        """
        for region in self.data_config['target_regions']:
            if region['name'] == region_name:
                return region
        return None
    
    def _save_raw_data(self, wildfire_data: pd.DataFrame, weather_data: pd.DataFrame, region: str) -> None:
        """Save raw data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not wildfire_data.empty:
            wildfire_file = Path(self.paths['raw_data']) / f"wildfire_data_{region}_{timestamp}.csv"
            wildfire_data.to_csv(wildfire_file, index=False)
            self.logger.info(f"Saved wildfire data to {wildfire_file}")
        
        if not weather_data.empty:
            weather_file = Path(self.paths['raw_data']) / f"weather_data_{region}_{timestamp}.csv"
            weather_data.to_csv(weather_file, index=False)
            self.logger.info(f"Saved weather data to {weather_file}")
    
    def _save_processed_data(self, data: pd.DataFrame, region: str) -> None:
        """Save processed data to file."""
        if data.empty:
            self.logger.warning("No processed data to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_file = Path(self.paths['processed_data']) / f"processed_data_{region}_{timestamp}.csv"
        
        data.to_csv(processed_file, index=False)
        self.logger.info(f"Saved processed data to {processed_file}")
        
        # Also save as the latest version
        latest_file = Path(self.paths['processed_data']) / f"latest_{region}.csv"
        data.to_csv(latest_file, index=False)
        self.logger.info(f"Saved latest data to {latest_file}") 