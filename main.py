#!/usr/bin/env python3
"""
FireSight: Wildfire Prediction System
Main application entry point

This module provides a command-line interface for the FireSight wildfire prediction system.
It allows users to collect data, train models, and make predictions.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection.data_collector import DataCollector
from models.model_trainer import ModelTrainer
from models.predictor import WildfirePredictor
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FireSight: Wildfire Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data for the Bay Area
  python main.py --collect-data --region "Bay Area"

  # Train classification and regression models
  python main.py --train-models

  # Make a prediction for a specific location
  python main.py --predict --location "37.7749,-122.4194"

  # Evaluate model performance
  python main.py --evaluate --model-type classification
        """
    )
    
    # Data collection
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Collect wildfire and weather data"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="Bay Area",
        help="Target region for data collection"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for data collection (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for data collection (YYYY-MM-DD)"
    )
    
    # Model training
    parser.add_argument(
        "--train-models",
        action="store_true",
        help="Train classification and regression models"
    )
    parser.add_argument(
        "--model-type",
        choices=["classification", "regression", "both"],
        default="both",
        help="Type of model to train"
    )
    
    # Prediction
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Make wildfire predictions"
    )
    parser.add_argument(
        "--location",
        type=str,
        help="Location coordinates (lat,lon) for prediction"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date for prediction (YYYY-MM-DD), defaults to today"
    )
    
    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model performance"
    )
    
    # General options
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def collect_data(config: dict, args) -> None:
    """Collect wildfire and weather data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data collection...")
    
    try:
        collector = DataCollector(config)
        
        # Override config with command line arguments
        if args.start_date:
            config["data_collection"]["start_date"] = args.start_date
        if args.end_date:
            config["data_collection"]["end_date"] = args.end_date
        
        collector.collect_all_data(region=args.region)
        logger.info("Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise


def train_models(config: dict, args) -> None:
    """Train classification and regression models."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    try:
        trainer = ModelTrainer(config)
        
        if args.model_type in ["classification", "both"]:
            logger.info("Training classification models...")
            trainer.train_classification_models()
        
        if args.model_type in ["regression", "both"]:
            logger.info("Training regression models...")
            trainer.train_regression_models()
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def make_prediction(config: dict, args) -> None:
    """Make wildfire predictions for a given location."""
    logger = logging.getLogger(__name__)
    
    if not args.location:
        logger.error("Location coordinates are required for prediction")
        return
    
    try:
        predictor = WildfirePredictor(config)
        
        # Parse location coordinates
        lat, lon = map(float, args.location.split(","))
        
        # Make prediction
        prediction = predictor.predict(
            latitude=lat,
            longitude=lon,
            date=args.date
        )
        
        # Display results
        print("\n" + "="*50)
        print("FIRE SIGHT PREDICTION RESULTS")
        print("="*50)
        print(f"Location: {lat}, {lon}")
        print(f"Date: {prediction['date']}")
        print(f"Wildfire Risk: {prediction['wildfire_risk']:.2%}")
        print(f"Fire Severity: {prediction['fire_severity']:.2f}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        
        # Risk level interpretation
        risk_level = "LOW"
        if prediction['wildfire_risk'] > 0.7:
            risk_level = "HIGH"
        elif prediction['wildfire_risk'] > 0.4:
            risk_level = "MEDIUM"
        
        print(f"Risk Level: {risk_level}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def evaluate_models(config: dict, args) -> None:
    """Evaluate model performance."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    
    try:
        trainer = ModelTrainer(config)
        
        if args.model_type in ["classification", "both"]:
            logger.info("Evaluating classification models...")
            results = trainer.evaluate_classification_models()
            print("\nClassification Results:")
            print(results)
        
        if args.model_type in ["regression", "both"]:
            logger.info("Evaluating regression models...")
            results = trainer.evaluate_regression_models()
            print("\nRegression Results:")
            print(results)
        
        logger.info("Model evaluation completed!")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(config["logging"], log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("FireSight Wildfire Prediction System")
    logger.info("=" * 50)
    
    try:
        # Execute requested operations
        if args.collect_data:
            collect_data(config, args)
        
        if args.train_models:
            train_models(config, args)
        
        if args.predict:
            make_prediction(config, args)
        
        if args.evaluate:
            evaluate_models(config, args)
        
        # If no specific operation requested, show help
        if not any([args.collect_data, args.train_models, args.predict, args.evaluate]):
            print("No operation specified. Use --help for usage information.")
            sys.exit(1)
        
        logger.info("FireSight completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FireSight failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 