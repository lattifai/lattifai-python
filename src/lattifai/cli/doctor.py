"""Implementation of 'lai doctor' for LattifAI CLI – environment diagnostics."""

import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table

from lattifai.theme import _Theme as T

console = Console()


def _get_editable_source_dir() -> Path | None:
    """Return the source directory if lattifai is installed in editable mode, else None.

    Detection strategy (ordered by reliability):
    1. ``direct_url.json`` in dist metadata — standard PEP 610 marker.
    2. ``direct_url.json`` on disk in the .dist-info directory — some
       importlib.metadata backends fail to resolve it via ``read_text``.
    3. ``_path`` pointing to a ``.egg-info`` directory outside site-packages
       — legacy editable installs place egg-info in the project root.
    """
    try:
        import json
        import site

        dist = importlib.metadata.distribution("lattifai")

        # Strategy 1: PEP 610 direct_url.json via metadata API
        direct_url_text = dist.read_text("direct_url.json")

        # Strategy 2: fall back to reading from dist-info directory on disk
        if not direct_url_text:
            dist_path = getattr(dist, "_path", None)
            if dist_path:
                resolved = Path(dist_path).resolve()
                du_file = resolved / "direct_url.json"
                if du_file.exists():
                    direct_url_text = du_file.read_text()

        if direct_url_text:
            url_info = json.loads(direct_url_text)
            if url_info.get("dir_info", {}).get("editable", False):
                url = url_info.get("url", "")
                if url.startswith("file://"):
                    return Path(url[7:])

        # Strategy 3: egg-info in project root (not in site-packages)
        dist_path = getattr(dist, "_path", None)
        if dist_path:
            resolved = Path(dist_path).resolve()
            if resolved.name.endswith(".egg-info"):
                site_dirs = set(site.getsitepackages() + [site.getusersitepackages()])
                if not any(str(resolved).startswith(s) for s in site_dirs):
                    return resolved.parent
    except Exception:
        pass
    return None


def _get_source_version(source_dir: Path) -> str | None:
    """Read the version from pyproject.toml in the source directory."""
    pyproject_path = source_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return None
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("version")
    except Exception:
        return None


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
    except requests.RequestException as exc:
        return ("Package version", f"[{T.RICH_WARN}]{current} (network: {exc})[/{T.RICH_WARN}]", "WARN")
    except (KeyError, ValueError) as exc:
        return ("Package version", f"[{T.RICH_WARN}]{current} (parse error: {exc})[/{T.RICH_WARN}]", "WARN")
    except ImportError:
        return ("Package version", f"[{T.RICH_WARN}]{current} (packaging not installed)[/{T.RICH_WARN}]", "WARN")
    return ("Package version", f"[{T.RICH_WARN}]{current} (PyPI check failed)[/{T.RICH_WARN}]", "WARN")


def _find_stale_egg_info() -> list[Path]:
    """Find stale .egg-info directories that shadow the current editable install.

    When multiple metadata directories exist (e.g. an old egg-info at project
    root plus a newer dist-info in site-packages), importlib.metadata picks
    whichever appears first on sys.path — often the stale one.
    """
    versions: dict[str, list[Path]] = {}
    for dist in importlib.metadata.distributions():
        if dist.name == "lattifai":
            p = Path(getattr(dist, "_path", "")).resolve()
            versions.setdefault(dist.version, []).append(p)

    if len(versions) <= 1:
        return []

    # The latest version is correct; older versions are stale
    from packaging import version as v_parse

    sorted_versions = sorted(versions.keys(), key=v_parse.parse, reverse=True)
    stale_paths = []
    for v in sorted_versions[1:]:
        stale_paths.extend(versions[v])
    return stale_paths


def _check_editable_install() -> tuple[str, str, str]:
    """Check if editable install metadata is in sync with source pyproject.toml."""
    source_dir = _get_editable_source_dir()
    if source_dir is None:
        # Not an editable install — nothing to check
        return ("Editable install", f"[{T.RICH_DIM}]N/A (release install)[/{T.RICH_DIM}]", "OK")

    source_version = _get_source_version(source_dir)
    if source_version is None:
        return (
            "Editable install",
            f"[{T.RICH_WARN}]Cannot read source version from {source_dir}[/{T.RICH_WARN}]",
            "WARN",
        )

    # Check for stale metadata shadowing the current install
    stale = _find_stale_egg_info()
    if stale:
        stale_str = ", ".join(str(p) for p in stale)
        return (
            "Editable install",
            f"[{T.RICH_ERR}]Stale metadata found: {stale_str}. " f"Run: lai update[/{T.RICH_ERR}]",
            "FAIL",
        )

    # Compare installed version with source
    try:
        installed = importlib.metadata.version("lattifai")
    except importlib.metadata.PackageNotFoundError:
        return ("Editable install", f"[{T.RICH_ERR}]Package not found[/{T.RICH_ERR}]", "FAIL")

    if installed == source_version:
        return (
            "Editable install",
            f"[{T.RICH_OK}]{installed} (editable, in sync)[/{T.RICH_OK}]",
            "OK",
        )

    return (
        "Editable install",
        f"[{T.RICH_ERR}]Stale: installed={installed}, source={source_version}. " f"Run: lai update[/{T.RICH_ERR}]",
        "FAIL",
    )


def _check_os() -> tuple[str, str, str]:
    """Check operating system information."""
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        label = f"macOS {mac_ver} ({machine})"
    elif system == "Linux":
        label = f"Linux {platform.release()} ({machine})"
    elif system == "Windows":
        label = f"Windows {platform.version()} ({machine})"
    else:
        label = f"{system} {platform.release()} ({machine})"
    return ("OS", f"[{T.RICH_OK}]{label}[/{T.RICH_OK}]", "OK")


def _check_python_version() -> tuple[str, str, str]:
    """Check Python version meets requirements."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and 10 <= v.minor <= 14:
        return ("Python version", f"[{T.RICH_OK}]{version_str}[/{T.RICH_OK}]", "OK")
    return ("Python version", f"[{T.RICH_ERR}]{version_str} (requires 3.10-3.14)[/{T.RICH_ERR}]", "FAIL")


def _check_gpu() -> tuple[str, str, str]:
    """Check GPU / hardware acceleration."""
    accel = []

    # ONNX Runtime providers
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            accel.append("CUDA")
        if "CoreMLExecutionProvider" in providers:
            accel.append("CoreML")
    except ImportError:
        pass

    # PyTorch MPS (Apple Silicon GPU)
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accel.append("MPS")
    except ImportError:
        pass

    # MLX (Apple ML framework)
    if importlib.util.find_spec("mlx") is not None:
        accel.append("MLX")

    if accel:
        label = ", ".join(accel)
        return ("GPU acceleration", f"[{T.RICH_OK}]{label}[/{T.RICH_OK}]", "OK")
    if importlib.util.find_spec("onnxruntime") is None:
        return ("GPU acceleration", f"[{T.RICH_ERR}]onnxruntime not installed[/{T.RICH_ERR}]", "FAIL")
    return ("GPU acceleration", f"[{T.RICH_WARN}]CPU only[/{T.RICH_WARN}]", "WARN")


def _check_model_cache() -> tuple[str, str, str]:
    """Check Lattice-1 model cache status."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        from lattifai.utils import REQUIRED_MODEL_VERSIONS, _is_cache_valid

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
    except OSError as exc:
        return ("Model cache", f"[{T.RICH_ERR}]Cache unreadable: {exc}[/{T.RICH_ERR}]", "FAIL")


def _check_api_key() -> tuple[str, str, str]:
    """Check LATTIFAI_API_KEY: env var > config.toml [auth] > .env."""
    key = os.environ.get("LATTIFAI_API_KEY", "")
    source = "env"

    if not key:
        try:
            from lattifai.cli.config import get_auth_value

            key = get_auth_value("LATTIFAI_API_KEY") or ""
            source = "config.toml"
        except (ImportError, OSError):
            pass

    if not key:
        try:
            from dotenv import dotenv_values

            key = dotenv_values().get("LATTIFAI_API_KEY", "")
            source = ".env"
        except ImportError:
            pass

    if key:
        masked = f"...{key[-4:]}" if len(key) > 4 else "***"
        return ("API key", f"[{T.RICH_OK}]Set ({masked}) [{source}][/{T.RICH_OK}]", "OK")
    return ("API key", f"[{T.RICH_WARN}]LATTIFAI_API_KEY not set[/{T.RICH_WARN}]", "WARN")


def _check_dependencies() -> tuple[str, str, str]:
    """Check critical dependencies are importable."""
    deps = {"k2py": "k2py", "lhotse": "lhotse", "onnxruntime": "onnxruntime", "lattifai-core": "lattifai_core"}
    missing = []
    for pkg_name, module_name in deps.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(pkg_name)
    if not missing:
        return ("Dependencies", f"[{T.RICH_OK}]All critical deps installed[/{T.RICH_OK}]", "OK")
    return ("Dependencies", f"[{T.RICH_ERR}]Missing: {', '.join(missing)}[/{T.RICH_ERR}]", "FAIL")


def _check_selftest() -> tuple[str, str, str]:
    """Verify bundled test data and core caption pipeline."""
    import importlib.resources
    import tempfile

    data_pkg = "lattifai.data.selftest"
    passed = []
    errors = []
    files_ref = None

    # 1) Check bundled files exist
    try:
        files_ref = importlib.resources.files(data_pkg)
        wav_path = files_ref / "test.wav"
        vtt_path = files_ref / "test.vtt"
        srt_path = files_ref / "test.srt"
        for name, ref in [("test.wav", wav_path), ("test.vtt", vtt_path), ("test.srt", srt_path)]:
            # Materialize to a real path to verify it's bundled
            with importlib.resources.as_file(ref) as p:
                if not p.exists() or p.stat().st_size == 0:
                    errors.append(f"{name} missing")
        if not errors:
            passed.append("data")
    except Exception as exc:
        errors.append(f"data: {exc}")

    # Short-circuit if resource package lookup failed — later phases need files_ref
    if files_ref is None:
        if errors:
            return ("Self-test", f"[{T.RICH_ERR}]FAIL: {'; '.join(errors)}[/{T.RICH_ERR}]", "FAIL")
        return ("Self-test", f"[{T.RICH_ERR}]FAIL: bundled data unavailable[/{T.RICH_ERR}]", "FAIL")

    # 2) Caption read/write roundtrip (VTT → SRT → VTT)
    try:
        from lattifai.data.caption import Caption

        with importlib.resources.as_file(files_ref / "test.vtt") as vtt_file:
            cap = Caption.read(str(vtt_file))
        if not cap.supervisions or len(cap.supervisions) == 0:
            errors.append("VTT parse empty")
        else:
            with tempfile.NamedTemporaryFile(suffix=".srt", delete=True) as tmp:
                cap.write(tmp.name)
                cap2 = Caption.read(tmp.name)
                if len(cap2.supervisions) != len(cap.supervisions):
                    errors.append("roundtrip mismatch")
                else:
                    passed.append("caption")
    except Exception as exc:
        errors.append(f"caption: {exc}")

    # 3) Audio loading (uses private _load_audio — no public API exists yet)
    try:
        from lattifai.audio2 import AudioLoader

        loader = AudioLoader()
        with importlib.resources.as_file(files_ref / "test.wav") as wav_file:
            audio = loader._load_audio(str(wav_file), sampling_rate=16000, channel_selector=None)
        if audio is None or len(audio) == 0:
            errors.append("audio empty")
        else:
            passed.append("audio")
    except Exception as exc:
        errors.append(f"audio: {exc}")

    if errors:
        return ("Self-test", f"[{T.RICH_ERR}]FAIL: {'; '.join(errors)}[/{T.RICH_ERR}]", "FAIL")
    label = ", ".join(passed)
    return ("Self-test", f"[{T.RICH_OK}]OK ({label})[/{T.RICH_OK}]", "OK")


STATUS_ICONS = {
    "OK": f"[{T.RICH_OK}]✓[/{T.RICH_OK}]",
    "WARN": f"[{T.RICH_WARN}]![/{T.RICH_WARN}]",
    "FAIL": f"[{T.RICH_ERR}]✗[/{T.RICH_ERR}]",
}

CHECKS = [
    _check_os,
    _check_package_version,
    _check_editable_install,
    _check_python_version,
    _check_gpu,
    _check_model_cache,
    _check_api_key,
    _check_dependencies,
    _check_selftest,
]

_CHECK_HINTS = {
    "_check_package_version": "Run 'lai update' to upgrade.",
    "_check_editable_install": "Run 'lai update' or 'pip install --no-deps -e .' to re-sync.",
    "_check_python_version": "Install Python 3.10–3.14.",
    "_check_gpu": "Install onnxruntime with GPU support.",
    "_check_model_cache": "Run an alignment to auto-download the model.",
    "_check_api_key": "Set LATTIFAI_API_KEY in your environment.",
    "_check_dependencies": "Run 'pip install lattifai[all]' to install missing deps.",
    "_check_selftest": "Try reinstalling: pip install --force-reinstall lattifai",
}


def doctor() -> int:
    """
    Run environment diagnostics for LattifAI.

    Checks package version, Python version, GPU acceleration,
    model cache, API key, and critical dependencies.

    Returns:
        Exit code: 0 if all OK/WARN, 1 if any FAIL.
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
        try:
            name, detail, status = check_fn()
        except Exception as exc:
            name = check_fn.__doc__.split(".")[0].strip() if check_fn.__doc__ else check_fn.__name__
            hint = _CHECK_HINTS.get(check_fn.__name__, "")
            hint_suffix = f" — {hint}" if hint else ""
            detail = f"[{T.RICH_ERR}]Unexpected error: {exc}{hint_suffix}[/{T.RICH_ERR}]"
            status = "FAIL"
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

    return 1 if fail_count else 0


def main():
    """Main entry point for lai-doctor script."""
    sys.exit(doctor())
