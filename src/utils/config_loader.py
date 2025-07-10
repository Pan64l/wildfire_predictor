"""
Configuration loader for FireSight.

This module handles loading and parsing of configuration files,
including environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Load and parse configuration files with environment variable support."""
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable substitution.
        
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Read the configuration file
        with open(self.config_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Substitute environment variables
        content = self._substitute_env_vars(content)
        
        # Parse YAML
        try:
            config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")
        
        return config
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.
        
        Args:
            content: Configuration file content
            
        Returns:
            Content with environment variables substituted
        """
        # Pattern to match ${VAR_NAME} or $VAR_NAME
        pattern = r'\$\{([^}]+)\}|\$([a-zA-Z_][a-zA-Z0-9_]*)'
        
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, f'${{{var_name}}}')
        
        return re.sub(pattern, replace_var, content)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = [
            'data_sources',
            'data_collection', 
            'feature_engineering',
            'models',
            'evaluation',
            'paths',
            'logging'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data sources
        if 'nasa_firms' not in config['data_sources']:
            raise ValueError("NASA FIRMS configuration is required")
        
        if 'openmeteo' not in config['data_sources']:
            raise ValueError("OpenMeteo configuration is required")
        
        # Validate model configuration
        if 'classification' not in config['models']:
            raise ValueError("Classification model configuration is required")
        
        if 'regression' not in config['models']:
            raise ValueError("Regression model configuration is required")
        
        return True
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model type.
        
        Args:
            model_type: Type of model ('classification' or 'regression')
            
        Returns:
            Model configuration dictionary
        """
        config = self.load_config()
        return config['models'].get(model_type, {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data collection configuration.
        
        Returns:
            Data collection configuration dictionary
        """
        config = self.load_config()
        return config['data_collection']
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation configuration.
        
        Returns:
            Evaluation configuration dictionary
        """
        config = self.load_config()
        return config['evaluation'] 