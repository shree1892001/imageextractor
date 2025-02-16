import logging
import os

# ANSI escape sequences for different log levels
COLORS = {
    "WARNING": "\033[93m",  # Yellow
    "INFO": "\033[94m",  # Blue
    "DEBUG": "\033[92m",  # Green
    "CRITICAL": "\033[91m",  # Red
    "ERROR": "\033[91m",  # Red
    "RESET": "\033[0m",  # Reset to default
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Format string with levelname and message
        log_fmt = f"%(asctime)s {COLORS['DEBUG']}Flumina: {COLORS[record.levelname]}%(message)s{COLORS['RESET']}"
        # ISO8601 format
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%dT%H:%M:%S")
        return formatter.format(record)


logger = None


def get_logger():
    global logger
    if logger is None:
        # Retrieve LOGLEVEL from environment or set default to 'INFO'
        log_level = os.getenv("LOGLEVEL", "INFO").upper()

        # Create a custom logger
        logger = logging.getLogger("flumina_logger")
        logger.setLevel(log_level)  # Set the log level
        logger.propagate = False

        # Check if handlers are already configured
        if not logger.handlers:
            # Create a stream handler with a custom formatter
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(ColoredFormatter())

            # Add the stream handler to the custom logger
            logger.addHandler(stream_handler)

    return logger
