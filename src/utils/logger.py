"""
Logging utilities for FireSight.

This module provides centralized logging configuration for the entire system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any


def setup_logger(config: Dict[str, Any], level: int = None) -> None:
    """
    Set up logging configuration for the FireSight system.
    
    Args:
        config: Logging configuration dictionary
        level: Override log level if provided
    """
    # Create logs directory if it doesn't exist
    log_file = Path(config.get('file', 'logs/firesight.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Set log level
    log_level = level or getattr(logging, config.get('level', 'INFO').upper())
    
    # Create formatter
    formatter = logging.Formatter(
        config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(self.__class__.__name__) 