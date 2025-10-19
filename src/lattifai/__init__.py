import os
import sys
import warnings

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


# Check and auto-install k2 if not present
def _check_and_install_k2():
    """Check if k2 is installed and attempt to install it if not."""
    try:
        import k2
    except ImportError:
        import subprocess

        print('k2 is not installed. Attempting to install k2...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'install-k2'])
            subprocess.check_call([sys.executable, '-m', 'install_k2'])
            import k2  # Try importing again after installation

            print('k2 installed successfully.')
        except Exception as e:
            warnings.warn(f'Failed to install k2 automatically. Please install it manually. Error: {e}')
    return True


# Auto-install k2 on first import
_check_and_install_k2()


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
