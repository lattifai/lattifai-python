import sys
import warnings
from importlib.metadata import version

# Re-export client classes
from .client import AsyncLattifAI, LattifAI

# Re-export config classes
from .config import (
    AUDIO_FORMATS,
    MEDIA_FORMATS,
    VIDEO_FORMATS,
    AlignmentConfig,
    ClientConfig,
    MediaConfig,
    SubtitleConfig,
)
from .errors import (
    AlignmentError,
    APIError,
    AudioFormatError,
    AudioLoadError,
    AudioProcessingError,
    ConfigurationError,
    DependencyError,
    LatticeDecodingError,
    LatticeEncodingError,
    LattifAIError,
    ModelLoadError,
    SubtitleParseError,
    SubtitleProcessingError,
)
from .subtitle import SubtitleIO

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
    "AsyncLattifAI",
    # Config classes
    "AlignmentConfig",
    "ClientConfig",
    "SubtitleConfig",
    "MediaConfig",
    "AUDIO_FORMATS",
    "VIDEO_FORMATS",
    "MEDIA_FORMATS",
    # Error classes
    "LattifAIError",
    "AudioProcessingError",
    "AudioLoadError",
    "AudioFormatError",
    "SubtitleProcessingError",
    "SubtitleParseError",
    "AlignmentError",
    "LatticeEncodingError",
    "LatticeDecodingError",
    "ModelLoadError",
    "DependencyError",
    "APIError",
    "ConfigurationError",
    # I/O
    "SubtitleIO",
    # Version
    "__version__",
]
