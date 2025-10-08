from .base_client import LattifAIError
from .io import SubtitleIO

try:
    from importlib.metadata import version
except ImportError:
    # Python < 3.8
    from importlib_metadata import version

try:
    __version__ = version('lattifai')
except Exception:
    __version__ = '0.1.0'  # fallback version


# Lazy import for LattifAI to avoid dependency issues during basic import
def __getattr__(name):
    if name == 'LattifAI':
        from .client import LattifAI

        return LattifAI
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'LattifAI',  # noqa: F822
    'LattifAIError',
    'SubtitleIO',
    '__version__',
]
