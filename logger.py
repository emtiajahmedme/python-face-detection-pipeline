"""Logging utilities for the Face Detection Pipeline."""

import logging
import os
import sys
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored logging formatter."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: str = "face_detection",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """Set up logger with both console and file handlers."""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "face_detection") -> logging.Logger:
    """Get an existing logger or create a new one."""
    return logging.getLogger(name)


class ProgressLogger:
    """Progress logging utility with context management."""
    
    def __init__(self, logger: logging.Logger, operation: str, total: Optional[int] = None):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.current = 0
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration.total_seconds():.2f} seconds")
        else:
            self.logger.error(f"Failed {self.operation} after {duration.total_seconds():.2f} seconds: {exc_val}")
    
    def update(self, increment: int = 1, message: str = ""):
        """Update progress."""
        self.current += increment
        if self.total:
            percentage = (self.current / self.total) * 100
            progress_msg = f"{self.operation} progress: {self.current}/{self.total} ({percentage:.1f}%)"
        else:
            progress_msg = f"{self.operation} progress: {self.current} items processed"
        
        if message:
            progress_msg += f" - {message}"
        
        self.logger.info(progress_msg)
    
    def log_error(self, error: str):
        """Log an error during the operation."""
        self.logger.error(f"Error in {self.operation}: {error}")
    
    def log_warning(self, warning: str):
        """Log a warning during the operation."""
        self.logger.warning(f"Warning in {self.operation}: {warning}")