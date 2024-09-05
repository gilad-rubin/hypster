import logging


class CustomFormatter(logging.Formatter):
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: CYAN + "%(levelname)s" + reset + " - " + format,
        logging.INFO: GREEN + "%(levelname)s" + reset + " - " + format,
        logging.WARNING: YELLOW + "%(levelname)s" + reset + " - " + format,
        logging.ERROR: RED + "%(levelname)s" + reset + " - " + format,
    }


# Logging Configuration Function
def configure_logging(level: int = logging.INFO) -> logging.Logger:
    # Get the root logger
    logger = logging.getLogger()

    # Remove all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Initialize the handler with the custom formatter
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())

    # Set the handler for the logger
    logger.addHandler(handler)
    #logger.setLevel(level)
    logger.setLevel(logging.WARNING)

    return logger
