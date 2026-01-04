import os
import warnings
from importlib.metadata import version

# Suppress SWIG deprecation warnings before any imports
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")

# Suppress PyTorch transformer nested tensor warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
