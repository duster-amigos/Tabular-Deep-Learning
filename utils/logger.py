from loguru import logger
import sys

try:
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", colorize=True)
    logger.info("Logger initialized successfully.")
except Exception as e:
    print(f"Failed to initialize logger: {e}")
    raise 