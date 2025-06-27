import logging
import os
from datetime import datetime

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

def setup_logger(name=__name__):
    """Configure and return a logger instance with color support in console."""

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # ─── File Formatter ────────────────────────────────────────────────
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(pathname)s:%(lineno)d]',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    log_file = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # ─── Console Formatter (Colorlog or Fallback) ──────────────────────
    if COLORLOG_AVAILABLE:
        color_formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            }
        )
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(color_formatter)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)

    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger
