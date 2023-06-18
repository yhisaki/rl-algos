import logging


logger = logging.getLogger("RL_ALGOS")


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    # # ANSI escape codes
    # FORMAT = "\033[34m%(name)s\033[0m:\033[31m%(levelname)s\033[0m:%(message)s"

    def format(self, record):
        return (
            f"\033[34m{record.name}\033[0m:\033[31m{record.levelname}\033[0m:{record.getMessage()}"
        )


# Create a handler
handler = logging.StreamHandler()
# Set the format for the handler
handler.setFormatter(CustomFormatter())
# Add the handler to the logger
logger.addHandler(handler)

# Set the logging level
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger = logger.getChild("test")
    logger.info("test")
    logger.warning("test")
