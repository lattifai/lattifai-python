"""Environment configuration for LattifAI.

Import this module early to suppress warnings before other imports.

Usage:
    import lattifai._init  # noqa: F401
    from lattifai.client import LattifAI
"""

import importlib.metadata
import os
import sys
import warnings

# Suppress SWIG deprecation warnings before any imports
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")

# Suppress PyTorch transformer nested tensor warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Expose __version__ on the namespace package so `import lattifai; lattifai.__version__` works.
_ns = sys.modules.get("lattifai")
if _ns is not None and not hasattr(_ns, "__version__"):
    try:
        _ns.__version__ = importlib.metadata.version("lattifai")
    except importlib.metadata.PackageNotFoundError:
        pass
