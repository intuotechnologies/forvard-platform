import sys
import os
from loguru import logger
import json
from datetime import datetime

# Log configurations
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT", 
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}"
)
LOG_FILE = os.getenv("LOG_FILE", "./logs/api.log")

# Ensure log directory exists - handle both absolute and relative paths
log_dir = os.path.dirname(os.path.abspath(LOG_FILE) if os.path.isabs(LOG_FILE) else os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(LOG_FILE))))
try:
    os.makedirs(log_dir, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create log directory at {log_dir}: {e}")
    # Fallback to a user-writable location if we can't create the specified directory
    home_dir = os.path.expanduser("~")
    log_dir = os.path.join(home_dir, "forvard_logs")
    os.makedirs(log_dir, exist_ok=True)
    LOG_FILE = os.path.join(log_dir, "api.log")
    print(f"Using fallback log location: {LOG_FILE}")


class JsonSerializer:
    """
    Custom serializer for JSON logging
    """
    def __call__(self, record):
        log_data = {
            "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            "process_id": record["process"].id,
            "thread_id": record["thread"].id
        }
        
        # Add exception info if available
        if record["exception"]:
            log_data["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        # Add extra fields
        log_data.update(record["extra"])
        
        return json.dumps(log_data)


def setup_logging():
    """
    Configure application logging
    """
    # Clear default loggers
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        colorize=True
    )
    
    # Add file logger with rotation and retention - only if we successfully created the log directory
    if os.path.isdir(os.path.dirname(LOG_FILE)):
        try:
            logger.add(
                LOG_FILE,
                format=LOG_FORMAT,
                level=LOG_LEVEL,
                rotation="10 MB",     # Rotate when file reaches 10MB
                retention="1 week",   # Keep logs for 1 week
                compression="zip"     # Compress rotated logs
            )
            
            # Add JSON logger for structured logging (optional)
            logger.add(
                f"{os.path.dirname(LOG_FILE)}/api.json",
                serialize=JsonSerializer(),
                level=LOG_LEVEL,
                rotation="10 MB",
                retention="1 week",
                compression="zip"
            )
            
            logger.info(f"File logging initialized at {LOG_FILE}")
        except Exception as e:
            logger.error(f"Failed to initialize file logging: {e}")
    else:
        logger.warning(f"Skipping file logging as directory {os.path.dirname(LOG_FILE)} is not accessible")
    
    logger.info("Logging system initialized")
    
    
def get_request_logger(request_id=None):
    """
    Create a contextualized logger for a request
    """
    if not request_id:
        request_id = f"req-{datetime.now().strftime('%Y%m%d%H%M%S')}-{id(datetime.now())}"
    
    # Return logger with request context
    return logger.bind(request_id=request_id)


# Error logger for exception handling
def log_error(error, context=None):
    """
    Log an exception with context
    """
    error_context = context or {}
    logger.error(
        f"Error: {str(error)}",
        error_type=type(error).__name__,
        **error_context
    )
    
    if hasattr(error, "__traceback__"):
        logger.exception(error) 