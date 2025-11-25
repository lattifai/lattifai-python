import sys
import warnings
from importlib.metadata import version

# Re-export I/O classes
from .caption import Caption

# Re-export client classes
from .client import LattifAI

# Re-export config classes
from .config import (
    AUDIO_FORMATS,
    MEDIA_FORMATS,
    VIDEO_FORMATS,
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    MediaConfig,
)
from .errors import (
    AlignmentError,
    APIError,
    AudioFormatError,
    AudioLoadError,
    AudioProcessingError,
    CaptionParseError,
    CaptionProcessingError,
    ConfigurationError,
    DependencyError,
    LatticeDecodingError,
    LatticeEncodingError,
    LattifAIError,
    ModelLoadError,
)
from .logging import get_logger, set_log_level, setup_logger

try:
    __version__ = version("lattifai")
except Exception:
    __version__ = "0.1.0"  # fallback version


# Check and auto-install k2 if not present
def _check_and_install_k2():
    """Check if k2 is installed and attempt to install it if not."""
    try:
        import k2
    except ImportError:
        import subprocess

        print("k2 is not installed. Attempting to install k2...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "install-k2"])
            subprocess.check_call([sys.executable, "-m", "install_k2"])
            import k2  # Try importing again after installation

            print("k2 installed successfully.")
        except Exception as e:
            warnings.warn(f"Failed to install k2 automatically. Please install it manually. Error: {e}")
    return True


# Auto-install k2 on first import
_check_and_install_k2()


__all__ = [
    # Client classes
    "LattifAI",
    # Config classes
    "AlignmentConfig",
    "ClientConfig",
    "CaptionConfig",
    "DiarizationConfig",
    "MediaConfig",
    "AUDIO_FORMATS",
    "VIDEO_FORMATS",
    "MEDIA_FORMATS",
    # Error classes
    "LattifAIError",
    "AudioProcessingError",
    "AudioLoadError",
    "AudioFormatError",
    "CaptionProcessingError",
    "CaptionParseError",
    "AlignmentError",
    "LatticeEncodingError",
    "LatticeDecodingError",
    "ModelLoadError",
    "DependencyError",
    "APIError",
    "ConfigurationError",
    # Logging
    "setup_logger",
    "get_logger",
    "set_log_level",
    # I/O
    "Caption",
    # Version
    "__version__",
]
