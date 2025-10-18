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

        return True
    except ImportError:
        pass

    # k2 not found, try to install it
    if os.environ.get('SKIP_K2_INSTALL'):
        warnings.warn(
            '\n' + '=' * 70 + '\n'
            '  k2 is not installed and auto-installation is disabled.\n'
            '  \n'
            '  To use lattifai, please install k2 by running:\n'
            '  \n'
            '      install-k2\n'
            '  \n' + '=' * 70,
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    print('\n' + '=' * 70)
    print('  k2 is not installed. Attempting to install it now...')
    print('  This is a one-time setup and may take a few minutes.')
    print('=' * 70 + '\n')

    try:
        # Import and run the installation script
        from scripts.install_k2 import install_k2_main

        install_k2_main(dry_run=False)

        print('\n' + '=' * 70)
        print('  k2 has been installed successfully!')
        print('=' * 70 + '\n')
        return True
    except Exception as e:
        warnings.warn(
            '\n' + '=' * 70 + '\n'
            f'  Failed to auto-install k2: {e}\n'
            '  \n'
            '  Please install k2 manually by running:\n'
            '  \n'
            '      install-k2\n'
            '  \n' + '=' * 70,
            RuntimeWarning,
            stacklevel=2,
        )
        return False


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
