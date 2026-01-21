"""Environment configuration for LattifAI.

Import this module early to suppress warnings before other imports.

Usage:
    import lattifai._init  # noqa: F401
    from lattifai.client import LattifAI
"""

import os
import warnings

# Suppress SWIG deprecation warnings before any imports
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")

# Suppress PyTorch transformer nested tensor warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
