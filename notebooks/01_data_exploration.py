# FireSight: Data Exploration and Analysis
# This script explores the wildfire and weather data collected for the FireSight project.

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import FireSight modules
from src.utils.config_loader import ConfigLoader
from src.data_collection.data_collector import DataCollector

def main():
    print("FireSight: Data Exploration and Analysis")
    print("=" * 50)
    
    # Load configuration
    config_loader = ConfigLoader('../config.yaml')
    config = config_loader.load_config()
    
    print("Configuration loaded successfully!")
    print(f"Target region: {config['data_collection']['target_regions'][0]['name']}")
    print(f"Date range: {config['data_collection']['start_date']} to {config['data_collection']['end_date']}")
    
    # Load latest processed data
    data_path = Path('../data/processed/latest_Bay Area.csv')
    
    if data_path.exists():
        data = pd.read_csv(data_path)
        data['date'] = pd.to_datetime(data['date'])
        print(f"\nLoaded {len(data)} records from {data_path}")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        
        # Data overview
        print(f"\nData Shape: {data.shape}")
        print(f"Columns: {len(data.columns)}")
        
        # Wildfire analysis
        if 'wildfire_occurred' in data.columns:
            wildfire_stats = data['wildfire_occurred'].value_counts()
            print(f"\nWildfire Occurrence:")
            print(f"  No wildfires: {wildfire_stats.get(0, 0)} ({wildfire_stats.get(0, 0)/len(data)*100:.2f}%)")
            print(f"  Wildfires: {wildfire_stats.get(1, 0)} ({wildfire_stats.get(1, 0)/len(data)*100:.2f}%)")
        
        # Weather analysis
        weather_cols = [col for col in data.columns if any(x in col for x in ['temperature', 'humidity', 'precipitation', 'wind', 'pressure'])]
        print(f"\nWeather Variables: {len(weather_cols)}")
        
        # Feature engineering preview
        from src.feature_engineering.feature_engineer import FeatureEngineer
        
        feature_engineer = FeatureEngineer(config)
        features_data = feature_engineer.engineer_features(data, "Bay Area")
        
        print(f"\nFeature Engineering:")
        print(f"Original data shape: {data.shape}")
        print(f"Features data shape: {features_data.shape}")
        
        # Show new features
        original_cols = set(data.columns)
        new_cols = set(features_data.columns) - original_cols
        
        print(f"New engineered features: {len(new_cols)}")
        
        # Save engineered features
        features_data.to_csv('../data/features/latest_Bay Area.csv', index=False)
        print("Engineered features saved!")
        
    else:
        print("No processed data found. Please run data collection first.")
        print("\nTo collect data, run:")
        print("python main.py --collect-data --region 'Bay Area'")

if __name__ == "__main__":
    main() 