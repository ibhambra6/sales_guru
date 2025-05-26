"""Centralized logging configuration for Sales Guru."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> None:
    """Setup centralized logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, only console logging is used
        max_file_size: Maximum size of log file before rotation (in bytes)
        backup_count: Number of backup files to keep
        format_string: Custom format string for log messages
    """
    # Default format string
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for third-party libraries to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Log the configuration
    logger = logging.getLogger("SalesGuru.Logging")
    logger.info(f"Logging configured - Level: {level}, File: {log_file or 'Console only'}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__ or module name)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_production_logging() -> None:
    """Setup logging configuration optimized for production use."""
    log_dir = Path("logs")
    log_file = log_dir / "sales_guru.log"
    
    setup_logging(
        level="INFO",
        log_file=str(log_file),
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10,
        format_string='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )


def setup_development_logging() -> None:
    """Setup logging configuration optimized for development use."""
    setup_logging(
        level="DEBUG",
        log_file=None,  # Console only for development
        format_string='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )


def setup_testing_logging() -> None:
    """Setup logging configuration optimized for testing."""
    setup_logging(
        level="WARNING",  # Reduce noise during testing
        log_file=None,
        format_string='%(levelname)s - %(name)s - %(message)s'
    ) 