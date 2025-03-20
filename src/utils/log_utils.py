import logging
from colorama import Fore, Style, init

def init_logging() -> None:
    """Initialize logging with colored output."""
    init(autoreset=True)  # Initialize colorama for cross-platform compatibility

    class ColorFormatter(logging.Formatter):
        """Custom formatter to add colors to log levels."""
        LOG_COLORS = {
            logging.DEBUG: Fore.CYAN,
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.RED + Style.BRIGHT,
        }

        def format(self, record):
            log_color = self.LOG_COLORS.get(record.levelno, "")
            record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
            return super().format(record)

    handler = logging.StreamHandler()
    formatter = ColorFormatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[handler])

# # Example usage
# if __name__ == "__main__":
#     init_logging()
#     logging.debug("This is a debug message")
#     logging.info("This is an info message")
#     logging.warning("This is a warning message")
#     logging.error("This is an error message")
#     logging.critical("This is a critical message")


import logging
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""

    LOG_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.LOG_COLORS.get(record.levelno, "")
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

def setup_logger(name: str):
    """Set up a logger with colored output."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = ColorFormatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

# # Example usage
# logger = setup_logger("my_logger")
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")
