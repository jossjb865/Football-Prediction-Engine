"""
Configuration module for Liga MX Football Prediction Engine.
Handles settings, logging, and environment validation.
"""

from .settings import settings
from .logging_config import setup_logging

__all__ = ["settings", "setup_logging"]
