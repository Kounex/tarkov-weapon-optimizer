"""
Application configuration and logging setup.
"""

import os
import sys

from loguru import logger


def setup_logging():
    """Configure loguru for Streamlit (reduce noise for UI)."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        filter=lambda record: record["level"].name != "DEBUG",
    )

    # Also log to file with rotation (if possible)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        logger.add(
            os.path.join(log_dir, "streamlit_app_{time}.log"),
            rotation="5 MB",
            retention="3 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )
    except (OSError, PermissionError):
        # File logging not available, continue with console only
        pass


def get_resource_path(filename: str) -> str:
    """Get the correct path for bundled resources.

    When running as a PyInstaller bundle, resources are extracted to a
    temporary directory (sys._MEIPASS). This function returns the correct
    path whether running from source or as a bundled executable.

    Args:
        filename: Name of the resource file (e.g., "tasks.json")

    Returns:
        Full path to the resource file
    """
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, filename)
    # When running from src/, go up one level to find root files
    root_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root_dir, filename)


# Page configuration constants
PAGE_CONFIG = {
    "page_title": "Tarkov Weapon Optimizer",
    "page_icon": "🔫",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Traders list (key, display_name)
TRADERS = [
    ("prapor", "Prapor"),
    ("skier", "Skier"),
    ("peacekeeper", "Peacekeeper"),
    ("mechanic", "Mechanic"),
    ("jaeger", "Jaeger"),
]
