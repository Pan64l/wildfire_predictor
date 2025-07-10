# FireSight: Model Training and Evaluation
# This script trains and evaluates wildfire prediction models.

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import FireSight modules
from src.utils.config_loader import ConfigLoader
from src.models.model_trainer import ModelTrainer

def main():
    print("FireSight: Model Training and Evaluation")
    print("=" * 50)
    
    # Load configuration
    config_loader = ConfigLoader('../config.yaml')
    config = config_loader.load_config()
    
    print("Configuration loaded successfully!")
    
    # Initialize model trainer
    trainer = ModelTrainer(config)
    
    # Check if we have data
    if trainer.data.empty:
        print("No data available for training.")
        print("Please run data collection first:")
        print("python main.py --collect-data --region 'Bay Area'")
        return
    
    print(f"Data loaded: {trainer.data.shape}")
    
    # Train classification models
    print("\nTraining classification models...")
    trainer.train_classification_models()
    
    # Train regression models
    print("\nTraining regression models...")
    trainer.train_regression_models()
    
    # Evaluate models
    print("\nEvaluating models...")
    
    # Classification results
    classification_results = trainer.evaluate_classification_models()
    if not classification_results.empty:
        print("\nClassification Model Results:")
        print(classification_results.to_string(index=False))
        
        # Save results
        results_file = Path('../results/classification_results.csv')
        results_file.parent.mkdir(parents=True, exist_ok=True)
        classification_results.to_csv(results_file, index=False)
        print(f"Classification results saved to {results_file}")
    
    # Regression results
    regression_results = trainer.evaluate_regression_models()
    if not regression_results.empty:
        print("\nRegression Model Results:")
        print(regression_results.to_string(index=False))
        
        # Save results
        results_file = Path('../results/regression_results.csv')
        results_file.parent.mkdir(parents=True, exist_ok=True)
        regression_results.to_csv(results_file, index=False)
        print(f"Regression results saved to {results_file}")
    
    print("\nModel training completed!")
    print("\nNext steps:")
    print("1. Review model performance")
    print("2. Make predictions using: python main.py --predict --location '37.7749,-122.4194'")
    print("3. Deploy models for real-time predictions")

if __name__ == "__main__":
    main() 