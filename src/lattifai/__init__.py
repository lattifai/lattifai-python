"""LattifAI Python SDK."""

# Preserve namespace package behaviour so that external packages
# (e.g. lattifai-captions → lattifai.caption) can still extend this namespace.
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

import lattifai._init  # noqa: F401 — suppress warnings, set __version__
