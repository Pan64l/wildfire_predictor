#!/usr/bin/env python3
"""
FireSight Setup Script

This script helps set up the FireSight wildfire prediction system.
It creates necessary directories, checks dependencies, and provides
initialization guidance.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/features",
        "models/classification",
        "models/regression",
        "results",
        "logs",
        "notebooks"
    ]
    
    print("\n📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost',
        'matplotlib', 'seaborn', 'requests', 'pyyaml',
        'python-dotenv', 'joblib'
    ]
    
    print("\n📦 Checking dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Set up environment file."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n🔧 Setting up environment file...")
        shutil.copy(env_example, env_file)
        print("  ✅ Created .env file from env.example")
        print("  📝 Please edit .env file with your API keys")
    elif env_file.exists():
        print("\n✅ Environment file already exists")
    else:
        print("\n⚠️  No env.example file found")

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API configuration...")
    
    # Check NASA FIRMS API key
    nasa_key = os.getenv('NASA_FIRMS_API_KEY')
    if nasa_key and nasa_key != 'your_nasa_firms_api_key_here':
        print("  ✅ NASA FIRMS API key configured")
    else:
        print("  ⚠️  NASA FIRMS API key not configured (optional)")
        print("     Get free API key from: https://firms.modaps.eosdis.nasa.gov/api/")
    
    # OpenMeteo doesn't require API key
    print("  ✅ OpenMeteo API (no key required)")

def run_tests():
    """Run basic system tests."""
    print("\n🧪 Running system tests...")
    
    try:
        # Test configuration loading
        from src.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader('config.yaml')
        config = config_loader.load_config()
        print("  ✅ Configuration loading")
        
        # Test data collector initialization
        from src.data_collection.data_collector import DataCollector
        collector = DataCollector(config)
        print("  ✅ Data collector initialization")
        
        # Test feature engineer initialization
        from src.feature_engineering.feature_engineer import FeatureEngineer
        feature_engineer = FeatureEngineer(config)
        print("  ✅ Feature engineer initialization")
        
        # Test model trainer initialization
        from src.models.model_trainer import ModelTrainer
        trainer = ModelTrainer(config)
        print("  ✅ Model trainer initialization")
        
        print("  ✅ All system tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ System test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 FireSight Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Configure API keys (optional):")
    print("   - Edit .env file with your NASA FIRMS API key")
    print("   - Get free API key from: https://firms.modaps.eosdis.nasa.gov/api/")
    
    print("\n2. Collect data:")
    print("   python main.py --collect-data --region 'Bay Area'")
    
    print("\n3. Explore data:")
    print("   python notebooks/01_data_exploration.py")
    
    print("\n4. Train models:")
    print("   python main.py --train-models")
    print("   # or")
    print("   python notebooks/02_model_training.py")
    
    print("\n5. Make predictions:")
    print("   python main.py --predict --location '37.7749,-122.4194'")
    
    print("\n6. Evaluate models:")
    print("   python main.py --evaluate")
    
    print("\n📚 Documentation:")
    print("- README.md: Project overview and usage")
    print("- config.yaml: Configuration options")
    print("- notebooks/: Analysis and training scripts")
    
    print("\n🔧 Troubleshooting:")
    print("- Check logs/ directory for error messages")
    print("- Ensure all dependencies are installed: pip install -r requirements.txt")
    print("- Verify API keys in .env file")

def main():
    """Main setup function."""
    print("🔥 FireSight Wildfire Prediction System Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies before continuing")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Check API keys
    check_api_keys()
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some system tests failed. Please check the errors above.")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 