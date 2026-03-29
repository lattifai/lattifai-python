"""Implementation of 'lai doctor' for LattifAI CLI – environment diagnostics."""

import importlib.metadata
import os
import sys
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table

from lattifai.theme import _Theme as T

console = Console()


def _check_package_version() -> tuple[str, str, str]:
    """Check installed vs latest PyPI version."""
    try:
        current = importlib.metadata.version("lattifai")
    except importlib.metadata.PackageNotFoundError:
        return ("Package version", f"[{T.RICH_ERR}]Not installed[/{T.RICH_ERR}]", "FAIL")

    try:
        resp = requests.get("https://pypi.org/pypi/lattifai/json", timeout=5)
        if resp.status_code == 200:
            latest = resp.json()["info"]["version"]
            from packaging import version as v_parse

            if v_parse.parse(current) >= v_parse.parse(latest):
                return ("Package version", f"[{T.RICH_OK}]{current} (latest)[/{T.RICH_OK}]", "OK")
            else:
                return ("Package version", f"[{T.RICH_WARN}]{current} -> {latest} available[/{T.RICH_WARN}]", "WARN")
    except Exception:
        pass
    return ("Package version", f"[{T.RICH_OK}]{current}[/{T.RICH_OK}] (PyPI check failed)", "OK")


def _check_python_version() -> tuple[str, str, str]:
    """Check Python version meets requirements."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and 10 <= v.minor <= 14:
        return ("Python version", f"[{T.RICH_OK}]{version_str}[/{T.RICH_OK}]", "OK")
    return ("Python version", f"[{T.RICH_ERR}]{version_str} (requires 3.10-3.14)[/{T.RICH_ERR}]", "FAIL")


def _check_gpu() -> tuple[str, str, str]:
    """Check GPU / hardware acceleration."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return ("GPU acceleration", f"[{T.RICH_OK}]CUDA[/{T.RICH_OK}]", "OK")
        if "CoreMLExecutionProvider" in providers:
            return ("GPU acceleration", f"[{T.RICH_OK}]CoreML[/{T.RICH_OK}]", "OK")
        return ("GPU acceleration", f"[{T.RICH_WARN}]CPU only[/{T.RICH_WARN}]", "WARN")
    except ImportError:
        return ("GPU acceleration", f"[{T.RICH_ERR}]onnxruntime not installed[/{T.RICH_ERR}]", "FAIL")


def _check_model_cache() -> tuple[str, str, str]:
    """Check Lattice-1 model cache status."""
    from lattifai.utils import REQUIRED_MODEL_VERSIONS, _is_cache_valid

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        model_id = "LattifAI/Lattice-1"
        cache_dir = Path(HF_HUB_CACHE) / f'models--{model_id.replace("/", "--")}'

        if not cache_dir.exists():
            return ("Model cache", f"[{T.RICH_WARN}]Not downloaded yet[/{T.RICH_WARN}]", "WARN")

        if _is_cache_valid(cache_dir, model_name=model_id):
            # Find marker date
            markers = list(cache_dir.glob(".done*"))
            if markers:
                latest = max(markers, key=lambda p: p.stat().st_mtime)
                date_str = latest.name.replace(".done", "")
                return ("Model cache", f"[{T.RICH_OK}]Valid (cached {date_str})[/{T.RICH_OK}]", "OK")
            return ("Model cache", f"[{T.RICH_OK}]Valid[/{T.RICH_OK}]", "OK")
        else:
            req = REQUIRED_MODEL_VERSIONS.get(model_id, {})
            return (
                "Model cache",
                f"[{T.RICH_WARN}]Stale – update needed (min: {req.get('min_revision', '?')})[/{T.RICH_WARN}]",
                "WARN",
            )
    except ImportError:
        return ("Model cache", f"[{T.RICH_ERR}]huggingface_hub not installed[/{T.RICH_ERR}]", "FAIL")


def _check_api_key() -> tuple[str, str, str]:
    """Check LATTIFAI_API_KEY environment variable."""
    key = os.environ.get("LATTIFAI_API_KEY", "")
    if key:
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        return ("API key", f"[{T.RICH_OK}]Set ({masked})[/{T.RICH_OK}]", "OK")
    return ("API key", f"[{T.RICH_WARN}]LATTIFAI_API_KEY not set[/{T.RICH_WARN}]", "WARN")


def _check_dependencies() -> tuple[str, str, str]:
    """Check critical dependencies are importable."""
    deps = {"k2py": "k2", "lhotse": "lhotse", "onnxruntime": "onnxruntime", "lattifai-core": "lattifai_core"}
    missing = []
    for pkg_name, import_name in deps.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    if not missing:
        return ("Dependencies", f"[{T.RICH_OK}]All critical deps installed[/{T.RICH_OK}]", "OK")
    return ("Dependencies", f"[{T.RICH_ERR}]Missing: {', '.join(missing)}[/{T.RICH_ERR}]", "FAIL")


STATUS_ICONS = {
    "OK": f"[{T.RICH_OK}]✓[/{T.RICH_OK}]",
    "WARN": f"[{T.RICH_WARN}]![/{T.RICH_WARN}]",
    "FAIL": f"[{T.RICH_ERR}]✗[/{T.RICH_ERR}]",
}

CHECKS = [
    _check_package_version,
    _check_python_version,
    _check_gpu,
    _check_model_cache,
    _check_api_key,
    _check_dependencies,
]


def doctor():
    """
    Run environment diagnostics for LattifAI.

    Checks package version, Python version, GPU acceleration,
    model cache, API key, and critical dependencies.
    """
    console.print()
    console.print(f"[{T.RICH_HEADER}]LattifAI Doctor[/{T.RICH_HEADER}]")
    console.print()

    table = Table(show_header=True, header_style="bold", show_lines=False, pad_edge=False)
    table.add_column("", width=3)
    table.add_column("Check", min_width=20)
    table.add_column("Status", min_width=40)

    ok_count = warn_count = fail_count = 0

    for check_fn in CHECKS:
        name, detail, status = check_fn()
        icon = STATUS_ICONS.get(status, "?")
        table.add_row(icon, name, detail)
        if status == "OK":
            ok_count += 1
        elif status == "WARN":
            warn_count += 1
        else:
            fail_count += 1

    console.print(table)
    console.print()

    if fail_count:
        console.print(
            f"[{T.RICH_ERR}]{fail_count} issue(s) found.[/{T.RICH_ERR}]"
            f" Run [{T.RICH_LABEL}]lai update[/{T.RICH_LABEL}] or check above."
        )
    elif warn_count:
        console.print(f"[{T.RICH_WARN}]{warn_count} warning(s).[/{T.RICH_WARN}] Everything should still work.")
    else:
        console.print(f"[{T.RICH_OK}]All checks passed![/{T.RICH_OK}]")
    console.print()


def main():
    """Main entry point for lai-doctor script."""
    doctor()
