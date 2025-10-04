"""
HaliosAI - Advanced AI Guardrails SDK

Configuration and utility functions.
"""

import os
import logging
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging for HaliosAI SDK
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logger for haliosai package
    logger = logging.getLogger("haliosai")
    
    # Don't add handlers if already configured
    if logger.handlers:
        return
        
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False


def get_api_key() -> Optional[str]:
    """Get API key from environment variable"""
    return os.getenv("HALIOS_API_KEY")


def get_base_url() -> str:
    """Get base URL from environment variable with default fallback"""
    return os.getenv("HALIOS_BASE_URL", "https://api.halios.ai")


# Auto-setup logging on import
setup_logging(os.getenv("HALIOS_LOG_LEVEL", "INFO"))
