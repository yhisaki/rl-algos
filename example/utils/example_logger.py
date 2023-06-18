from rl_algos.utils import logger


def example_logger():
    logger.info("Hello World!")
    logger.debug("Hello World!")
    logger.warning("Hello World!")
    logger.error("Hello World!")


if __name__ == "__main__":
    example_logger()
