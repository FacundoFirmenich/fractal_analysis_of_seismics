"""
Structured logging for SFA framework.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class SFALogger:
    """
    Structured logger for SFA operations.
    Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    
    @staticmethod
    def setup(
        name: str = "sfa",
        level: int = logging.INFO,
        log_file: Path = None,
        console: bool = True
    ) -> logging.Logger:
        """
        Configure logger with file and console handlers.
        
        Args:
            name: Logger name
            level: Minimum logging level
            log_file: Optional file path
            console: Enable console output
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers = []  # Clear existing
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def get_logger(name: str = "sfa") -> logging.Logger:
        """Get existing or create default logger."""
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            # Setup default if not configured
            SFALogger.setup(name)
        
        return logger


# Convenience functions
def log_analysis_start(region: str, n_events: int, method: str = "GP"):
    """Log analysis initialization."""
    logger = SFALogger.get_logger()
    logger.info(f"Starting {method} analysis | Region: {region} | N={n_events}")


def log_result(region: str, D2: float, SEM: float = None):
    """Log computed result."""
    logger = SFALogger.get_logger()
    if SEM is not None:
        logger.info(f"Result | {region} | D₂={D2:.3f}±{SEM:.3f}")
    else:
        logger.info(f"Result | {region} | D₂={D2:.3f}")


def log_warning(message: str):
    """Log warning."""
    logger = SFALogger.get_logger()
    logger.warning(message)


def log_error(message: str, exception: Exception = None):
    """Log error with optional exception."""
    logger = SFALogger.get_logger()
    if exception:
        logger.error(f"{message} | {type(exception).__name__}: {exception}")
    else:
        logger.error(message)


def log_performance(operation: str, duration_ms: float):
    """Log performance metric."""
    logger = SFALogger.get_logger()
    logger.debug(f"Performance | {operation} | {duration_ms:.1f}ms")
