
## Project Overview

FireSight is a comprehensive machine learning-based wildfire prediction system designed to address the critical need for enhanced real-time wildfire risk assessment. The system integrates data from multiple sources to predict both wildfire occurrence (classification) and fire severity (regression).

### Key Features
- **Dual Prediction Models**: Classification for wildfire occurrence and regression for severity estimation
- **Real-time Data Integration**: NASA FIRMS for wildfire data and OpenMeteo for weather data
- **Advanced Feature Engineering**: Temporal features, rolling statistics, and fire risk indicators
- **Multiple ML Algorithms**: Random Forest, XGBoost, AdaBoost, SVR, and TabNet
- **Comprehensive Evaluation**: Appropriate metrics for both classification and regression tasks
- **Production-Ready**: Command-line interface, logging, and configuration management


## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for data collection

### Quick Start
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd wildfire_predictor
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (optional):
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

### Manual Setup
If you prefer manual setup:

1. **Create directories**:
   ```bash
   mkdir -p data/{raw,processed,features}
   mkdir -p models/{classification,regression}
   mkdir -p {results,logs,notebooks}
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env file
   ```

## Data Collection

### Data Sources

#### NASA FIRMS (Fire Information for Resource Management System)
- **Purpose**: Historical wildfire occurrence data
- **Data**: Fire locations, timestamps, confidence scores, fire radiative power
- **API**: Free with registration
- **Coverage**: Global, near real-time

#### OpenMeteo API
- **Purpose**: Historical weather data
- **Data**: Temperature, humidity, precipitation, wind, pressure, evapotranspiration
- **API**: Free, no key required
- **Coverage**: Global, high-resolution

### Collection Process

1. **Configure target regions** in `config.yaml`:
   ```yaml
   data_collection:
     target_regions:
       - name: "Bay Area"
         bounds: [37.0, -123.0, 38.5, -121.5]
         counties: ["San Francisco", "Alameda", "Contra Costa", "San Mateo", "Santa Clara"]
   ```

2. **Run data collection**:
   ```bash
   python main.py --collect-data --region "Bay Area"
   ```

3. **Monitor progress** in logs:
   ```bash
   tail -f logs/firesight.log
   ```

### Data Processing
- **Merging**: Wildfire and weather data are merged by date and location
- **Cleaning**: Missing values are handled with forward/backward fill
- **Validation**: Data quality checks and outlier detection
- **Storage**: Processed data saved in CSV format with timestamps

## Feature Engineering

### Feature Categories

#### Temporal Features
- `day_of_year`: Day of the year (1-365)
- `month`: Month (1-12)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Binary indicator
- `quarter`: Quarter of the year
- `year_progress`: Progress through the year (0-1)

#### Rolling Statistics
- **Windows**: 3, 7, 14, 30 days
- **Statistics**: Mean, standard deviation, min, max, trend
- **Variables**: Temperature, humidity, precipitation, wind speed

#### Lag Features
- **Lags**: 1, 2, 3, 7 days
- **Variables**: Temperature, humidity, precipitation, wind speed
- **Purpose**: Capture delayed effects of weather conditions

#### Fire Risk Indicators
- `temperature_anomaly`: Deviation from seasonal average
- `humidity_deficit`: How much below optimal humidity (40%)
- `wind_dryness_index`: Wind speed × temperature / humidity
- `precipitation_deficit`: Days since last significant rain (>5mm)
- `fire_weather_index`: Composite risk indicator

#### Seasonal Features
- **Cyclical encoding**: Sin/cos transformations for day_of_year and month
- **Season encoding**: One-hot encoding for seasons
- **Purpose**: Capture seasonal patterns in wildfire occurrence

### Feature Engineering Process

1. **Load processed data**:
   ```python
   from src.feature_engineering.feature_engineer import FeatureEngineer
   
   feature_engineer = FeatureEngineer(config)
   features = feature_engineer.engineer_features(data, "Bay Area")
   ```

2. **Feature selection**:
   ```python
   feature_cols = feature_engineer.get_feature_columns(features)
   ```

3. **Save features**:
   ```python
   features.to_csv('data/features/latest_Bay Area.csv', index=False)
   ```

## Model Training

### Model Types

#### Classification Models
- **Target**: `wildfire_occurred` (0/1)
- **Algorithms**: Logistic Regression, Random Forest, XGBoost, AdaBoost, TabNet
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### Regression Models
- **Target**: `fire_severity` (continuous)
- **Algorithms**: Linear Regression, SVR, Random Forest, XGBoost
- **Metrics**: R², MAE, MSE, RMSE

### Training Process

1. **Data preparation**:
   - Split into training/test sets (80/20)
   - Stratified sampling for classification
   - Feature scaling with StandardScaler

2. **Model training**:
   ```bash
   python main.py --train-models
   ```

3. **Model evaluation**:
   ```bash
   python main.py --evaluate
   ```

4. **Model persistence**:
   - Models saved with timestamps
   - Latest models saved as `latest_*.pkl`
   - Scalers saved separately

### Model Configuration

Configure models in `config.yaml`:
```yaml
models:
  classification:
    algorithms:
      random_forest:
        n_estimators: 100
        max_depth: 10
        min_samples_split: 5
        min_samples_leaf: 2
      xgboost:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
        subsample: 0.8
```

## Prediction System

### Making Predictions

1. **Single location prediction**:
   ```bash
   python main.py --predict --location "37.7749,-122.4194"
   ```

2. **Batch predictions**:
   ```python
   from src.models.predictor import WildfirePredictor
   
   predictor = WildfirePredictor(config)
   locations = [(37.7749, -122.4194), (37.3382, -121.8863)]
   predictions = predictor.predict_batch(locations)
   ```

### Prediction Output

```json
{
  "date": "2024-01-15",
  "latitude": 37.7749,
  "longitude": -122.4194,
  "wildfire_risk": 0.23,
  "fire_severity": 45.6,
  "confidence": 0.85
}
```

### Real-time Prediction Process

1. **Weather data retrieval**: Fetch current weather for target location
2. **Feature engineering**: Apply same feature engineering pipeline
3. **Model prediction**: Use trained models to make predictions
4. **Confidence calculation**: Assess prediction reliability
5. **Risk interpretation**: Convert probabilities to risk levels

## Configuration

### Configuration File Structure

The `config.yaml` file contains all system configuration:

```yaml
# Data Sources
data_sources:
  nasa_firms:
    base_url: "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    api_key: "${NASA_FIRMS_API_KEY}"
  
  openmeteo:
    base_url: "https://archive-api.open-meteo.com/v1/archive"
    default_timezone: "America/Los_Angeles"

# Data Collection
data_collection:
  start_date: "2018-01-01"
  end_date: "2023-12-31"
  target_regions:
    - name: "Bay Area"
      bounds: [37.0, -123.0, 38.5, -121.5]
  
  weather_features:
    - temperature_2m_max
    - relative_humidity_2m_min
    - precipitation_sum
    - wind_speed_10m_max

# Feature Engineering
feature_engineering:
  temporal_features:
    - day_of_year
    - month
    - season
  rolling_windows: [3, 7, 14, 30]
  lag_features: [1, 2, 3, 7]

# Models
models:
  classification:
    target_column: "wildfire_occurred"
    algorithms:
      random_forest:
        n_estimators: 100
        max_depth: 10
  
  regression:
    target_column: "fire_severity"
    algorithms:
      xgboost_regressor:
        n_estimators: 100
        max_depth: 6

# Evaluation
evaluation:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

# File Paths
paths:
  data_dir: "data"
  models_dir: "models"
  results_dir: "results"

# Logging
logging:
  level: "INFO"
  file: "logs/firesight.log"
```

### Environment Variables

Set in `.env` file:
```bash
# NASA FIRMS API Key (optional)
NASA_FIRMS_API_KEY=your_api_key_here

# Logging level (optional)
LOG_LEVEL=INFO
```

## API Reference

### DataCollector

```python
class DataCollector:
    def __init__(self, config: Dict)
    def collect_all_data(self, region: str = "Bay Area") -> None
    def collect_wildfire_data(self, region_config: Dict) -> pd.DataFrame
    def collect_weather_data(self, region_config: Dict) -> pd.DataFrame
    def process_and_merge_data(self, wildfire_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame
```

### FeatureEngineer

```python
class FeatureEngineer:
    def __init__(self, config: Dict)
    def engineer_features(self, data: pd.DataFrame, region: str = "Bay Area") -> pd.DataFrame
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]
```

### ModelTrainer

```python
class ModelTrainer:
    def __init__(self, config: Dict)
    def train_classification_models(self) -> None
    def train_regression_models(self) -> None
    def evaluate_classification_models(self) -> pd.DataFrame
    def evaluate_regression_models(self) -> pd.DataFrame
```

### WildfirePredictor

```python
class WildfirePredictor:
    def __init__(self, config: Dict)
    def predict(self, latitude: float, longitude: float, date: Optional[str] = None) -> Dict
    def predict_batch(self, locations: List[Tuple[float, float]], date: Optional[str] = None) -> pd.DataFrame
```

## Troubleshooting

### Common Issues

#### 1. Missing Dependencies
**Error**: `ModuleNotFoundError: No module named 'xgboost'`
**Solution**: Install missing packages:
```bash
pip install -r requirements.txt
```

#### 2. API Key Issues
**Error**: `Failed to fetch wildfire data`
**Solution**: 
- Check NASA FIRMS API key in `.env` file
- Verify API key is valid at https://firms.modaps.eosdis.nasa.gov/api/
- Note: NASA FIRMS data is optional; system works with weather data only

#### 3. Data Collection Failures
**Error**: `No wildfire data found`
**Solution**:
- Check internet connection
- Verify date range in config
- Check region bounds are valid
- Review logs for specific error messages

#### 4. Model Training Issues
**Error**: `No data available for training`
**Solution**:
- Run data collection first: `python main.py --collect-data`
- Check data files exist in `data/processed/`
- Verify data contains required columns

#### 5. Prediction Failures
**Error**: `No trained models found`
**Solution**:
- Train models first: `python main.py --train-models`
- Check model files exist in `models/` directories
- Verify model files are not corrupted

### Debug Mode

Enable verbose logging:
```bash
python main.py --verbose --predict --location "37.7749,-122.4194"
```

### Log Analysis

Check system logs:
```bash
tail -f logs/firesight.log
```

### Data Validation

Validate data quality:
```python
from src.data_collection.data_collector import DataCollector
collector = DataCollector(config)
# Check data quality and completeness
```

### Performance Optimization

1. **Reduce data range**: Modify `start_date` and `end_date` in config
2. **Limit features**: Reduce rolling windows and lag features
3. **Use smaller models**: Reduce `n_estimators` and `max_depth`
4. **Enable caching**: Store intermediate results

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Make changes** and add tests
4. **Run tests**: `python -m pytest tests/`
5. **Submit pull request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- Include error handling
- Write unit tests for new features

### Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```
