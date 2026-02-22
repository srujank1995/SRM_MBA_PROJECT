"""
Application Logging Module
================================================================================
This module provides comprehensive logging functionality for the retail
transaction prediction system. It includes:

- Structured logging with multiple handlers (file and console)
- Rotating file handler to prevent log file overflow
- Configurable log levels and formats
- Color-coded console output for better readability
- Centralized log configuration
- Performance monitoring and tracing

Features:
- Automatic log file creation with timestamps
- Daily log file rotation
- Detailed logging format with timestamps and context
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file handlers with different formats
- Thread-safe logging

Author: Srujan Vijay Kinjawadekar
Date: February 2026
Version: 1.0.0
================================================================================
"""

import logging as logging_module
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Logger configuration
LOG_DIRECTORY = os.path.join(os.getcwd(), "logs")
LOG_FILENAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, LOG_FILENAME)

# Log levels
LOG_LEVEL = logging_module.INFO
LOG_FORMAT_DETAIL = "[ %(asctime)s ] - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s"
LOG_FORMAT_SIMPLE = "[ %(asctime)s ] %(levelname)s - %(message)s"

# Console colors (ANSI escape codes)
class Colors:
    """ANSI color codes for console output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


# ============================================================================
# CUSTOM FORMATTERS
# ============================================================================

class ColoredFormatter(logging_module.Formatter):
    """
    Custom formatter that adds color to console output based on log level.
    
    Provides visual distinction between different log levels in console output
    for improved readability and faster issue identification.
    """
    
    # Color mapping for log levels
    LEVEL_COLORS = {
        logging_module.DEBUG: Colors.CYAN,
        logging_module.INFO: Colors.GREEN,
        logging_module.WARNING: Colors.YELLOW,
        logging_module.ERROR: Colors.RED,
        logging_module.CRITICAL: Colors.BG_RED + Colors.WHITE,
    }
    
    def format(self, record: logging_module.LogRecord) -> str:
        """
        Format log record with color based on log level.
        
        Args:
            record (logging_module.LogRecord): The log record to format
            
        Returns:
            str: Formatted log message with ANSI color codes
        """
        # Get color for this log level
        levelname_color = self.LEVEL_COLORS.get(record.levelno, '')
        
        # Add color to level name
        record.levelname = f"{levelname_color}{record.levelname}{Colors.RESET}"
        
        # Format the message
        return super().format(record)


class DetailedFormatter(logging_module.Formatter):
    """
    Custom formatter for detailed log output with complete context.
    
    Includes all available information about the log event:
    - Timestamp with milliseconds
    - File name and line number
    - Function name
    - Log level
    - Message
    """
    
    def format(self, record: logging_module.LogRecord) -> str:
        """Format with detailed information."""
        # Add milliseconds to timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Create formatted message
        formatted = (
            f"[{timestamp}] "
            f"[{record.levelname:^8}] "
            f"{record.name} - "
            f"{record.filename}:{record.lineno} - "
            f"{record.funcName}() - "
            f"{record.getMessage()}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


# ============================================================================
# LOGGER INITIALIZATION
# ============================================================================

def create_logger(
    name: str = 'retail_transaction_predictor',
    log_level: int = LOG_LEVEL,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10485760,
    backup_count: int = 5
) -> logging_module.Logger:
    """
    Create and configure a logger with file and console handlers.
    
    Creates a production-ready logger with:
    - File handler with rotation
    - Console handler with colors
    - Detailed formatting
    
    Args:
        name (str): Logger name (usually __name__)
        log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output (bool): Enable console output. Defaults to True.
        file_output (bool): Enable file output. Defaults to True.
        max_bytes (int): Max file size before rotation (bytes). Defaults to 10MB.
        backup_count (int): Number of backup files to keep. Defaults to 5.
        
    Returns:
        logging_module.Logger: Configured logger instance
        
    Example:
        >>> logger = create_logger(__name__)
        >>> logger.info("Application started")
        >>> logger.warning("Warning message")
        >>> logger.error("Error occurred")
    """
    
    # Create logger instance
    logger = logging_module.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # ========================================================================
    # CREATE LOG DIRECTORY
    # ========================================================================
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")
        file_output = False
    
    # ========================================================================
    # FILE HANDLER (with rotation)
    # ========================================================================
    if file_output:
        try:
            file_handler = RotatingFileHandler(
                filename=LOG_FILE_PATH,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            
            # File formatter (detailed)
            file_formatter = DetailedFormatter(LOG_FORMAT_DETAIL)
            file_handler.setFormatter(file_formatter)
            
            logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Error creating file handler: {e}")
    
    # ========================================================================
    # CONSOLE HANDLER (with colors)
    # ========================================================================
    if console_output:
        try:
            console_handler = logging_module.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            
            # Console formatter (colored)
            console_formatter = ColoredFormatter(LOG_FORMAT_SIMPLE)
            console_handler.setFormatter(console_formatter)
            
            logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"Error creating console handler: {e}")
    
    return logger


# ============================================================================
# GLOBAL LOGGER INSTANCE
# ============================================================================

# Create and configure the global logger
logging = create_logger(
    name='retail_transaction_predictor',
    log_level=LOG_LEVEL,
    console_output=True,
    file_output=True
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_logger(name: str = __name__) -> logging_module.Logger:
    """
    Get a logger instance with the specified name.
    
    Useful for creating module-specific loggers that inherit
    the global configuration while using a module-specific name.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging_module.Logger: Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging_module.getLogger(name)


def set_log_level(level: int) -> None:
    """
    Change the global log level.
    
    Args:
        level (int): New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        >>> set_log_level(logging_module.DEBUG)
        >>> set_log_level(logging_module.WARNING)
    """
    logging.setLevel(level)
    for handler in logging.handlers:
        handler.setLevel(level)
    logging.info(f"Log level changed to {logging_module.getLevelName(level)}")


def get_log_file_path() -> str:
    """
    Get the path to the current log file.
    
    Returns:
        str: Absolute path to the log file
        
    Example:
        >>> log_path = get_log_file_path()
        >>> print(f"Logs are saved to: {log_path}")
    """
    return LOG_FILE_PATH


def get_log_directory() -> str:
    """
    Get the path to the logs directory.
    
    Returns:
        str: Absolute path to the logs directory
        
    Example:
        >>> log_dir = get_log_directory()
        >>> print(f"Log directory: {log_dir}")
    """
    return LOG_DIRECTORY


def log_function_entry(func_name: str) -> None:
    """
    Log the entry point of a function.
    
    Args:
        func_name (str): Name of the function being entered
        
    Example:
        >>> def my_function():
        >>>     log_function_entry('my_function')
    """
    logging.debug(f"→ Entering function: {func_name}")


def log_function_exit(func_name: str) -> None:
    """
    Log the exit point of a function.
    
    Args:
        func_name (str): Name of the function being exited
        
    Example:
        >>> def my_function():
        >>>     log_function_exit('my_function')
    """
    logging.debug(f"← Exiting function: {func_name}")


def log_separator(char: str = '=', length: int = 80) -> None:
    """
    Log a separator line for visual organization.
    
    Args:
        char (str): Character to use for separator. Defaults to '='.
        length (int): Length of separator. Defaults to 80.
        
    Example:
        >>> log_separator()
        >>> logging.info("New Section")
        >>> log_separator()
    """
    logging.info(char * length)


# ============================================================================
# CONTEXT LOGGING
# ============================================================================

class LogContext:
    """
    Context manager for grouped logging with automatic indentation.
    
    Useful for logging related operations together with clear visual separation.
    
    Example:
        >>> with LogContext("Data Processing"):
        >>>     logging.info("Processing started")
        >>>     logging.info("Step 1 complete")
    """
    
    def __init__(self, context_name: str, level: int = logging_module.INFO):
        """
        Initialize LogContext.
        
        Args:
            context_name (str): Name of the context/section
            level (int): Logging level. Defaults to INFO.
        """
        self.context_name = context_name
        self.level = level
    
    def __enter__(self):
        """Enter context - log start."""
        logging.log(self.level, f"\n{'═' * 80}")
        logging.log(self.level, f"START: {self.context_name}")
        logging.log(self.level, f"{'═' * 80}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log end."""
        if exc_type is not None:
            logging.error(f"✗ {self.context_name} FAILED: {exc_val}")
        else:
            logging.log(self.level, f"✓ {self.context_name} COMPLETED")
        logging.log(self.level, f"{'═' * 80}\n")
        return False


# ============================================================================
# LOGGING STATISTICS
# ============================================================================

class LogStats:
    """
    Utility class to track logging statistics.
    
    Helps monitor the application's behavior through log analysis.
    """
    
    def __init__(self):
        """Initialize log statistics tracker."""
        self.counts = {
            logging_module.DEBUG: 0,
            logging_module.INFO: 0,
            logging_module.WARNING: 0,
            logging_module.ERROR: 0,
            logging_module.CRITICAL: 0,
        }
    
    def record(self, level: int) -> None:
        """Record a log at the specified level."""
        if level in self.counts:
            self.counts[level] += 1
    
    def get_summary(self) -> str:
        """Get formatted summary of log statistics."""
        return (
            f"Log Statistics:\n"
            f"  DEBUG:    {self.counts[logging_module.DEBUG]:>5}\n"
            f"  INFO:     {self.counts[logging_module.INFO]:>5}\n"
            f"  WARNING:  {self.counts[logging_module.WARNING]:>5}\n"
            f"  ERROR:    {self.counts[logging_module.ERROR]:>5}\n"
            f"  CRITICAL: {self.counts[logging_module.CRITICAL]:>5}\n"
            f"  TOTAL:    {sum(self.counts.values()):>5}"
        )


# ============================================================================
# EARLY LOGGING
# ============================================================================

# Log initialization
logging.info("=" * 80)
logging.info("LOGGER INITIALIZED SUCCESSFULLY")
logging.info("=" * 80)
logging.info(f"Log File: {LOG_FILE_PATH}")
logging.info(f"Log Level: {logging_module.getLevelName(LOG_LEVEL)}")
logging.info(f"Timestamp: {datetime.now().isoformat()}")
logging.info("=" * 80)


# ============================================================================
# END OF LOGGER MODULE
# ============================================================================
