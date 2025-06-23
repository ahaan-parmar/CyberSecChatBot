import re
import sys
from pathlib import Path
from typing import Dict

from .config import Config, MODEL_CONFIGS

class ConfigValidator:
    """Utility validation functions for initialization."""

    @staticmethod
    def validate_environment() -> Dict[str, bool]:
        """Validate basic environment settings."""
        checks = {
            "python_version": sys.version_info >= (3, 8),
            "required_packages": True,  # Assume packages installed if imports succeed
            "data_directory": Config.DATA_DIR.exists(),
            "vector_store_path": Path(Config.VECTOR_STORE_PATH).parent.exists(),
            "log_directory": Path(Config.LOG_FILE).parent.exists(),
        }
        return checks

    @staticmethod
    def validate_api_key(key: str, provider: str) -> bool:
        """Very simple API key format validation."""
        if not key:
            return False
        if provider == "openai":
            return bool(re.match(r"^sk-[A-Za-z0-9]+", key))
        if provider == "anthropic":
            return bool(re.match(r"^[a-zA-Z0-9-]{20,}$", key))
        return True

    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """Check that the model name is configured."""
        return model_name in MODEL_CONFIGS

class SecurityUtils:
    """Placeholder security related utilities."""

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Return a safe filename by stripping unsafe characters."""
        return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def format_file_size(size: int) -> str:
    """Return human readable file size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"
