from loguru import logger


def log_string(string: str) -> None:
    """Log a string using loguru."""
    logger.info(string)


if __name__ == "__main__":
    log_string()
