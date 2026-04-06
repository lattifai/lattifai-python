"""Implementation of 'lai update' for LattifAI CLI with nemo_run."""

import importlib.metadata
import importlib.util
import subprocess
import sys

import requests
from packaging import version as v_parse
from rich.console import Console

from lattifai.theme import _Theme as T

console = Console()

_PIP_TIMEOUT_SECS = 300  # 5 minutes for pip install


class AutoUpdater:
    """Automated updater for LattifAI package.

    The private mirror (index_url) is the authoritative source for releases.
    PyPI JSON API is used only for version *discovery*, not installation.
    """

    def __init__(self, package_name="lattifai"):
        self.package_name = package_name
        # Private mirror is authoritative; PyPI is the extra fallback.
        self.index_url = "https://lattifai.github.io/pypi/simple/"
        self.extra_index_url = "https://pypi.org/simple"

    def get_latest_version(self) -> str | None:
        """Fetch the latest version from PyPI JSON API (discovery only).

        Returns None when the network is unreachable or the response is
        malformed.  Callers must tolerate a None return.
        """
        try:
            response = requests.get(f"https://pypi.org/pypi/{self.package_name}/json", timeout=5)
            if response.status_code == 200:
                return response.json()["info"]["version"]
        except (requests.RequestException, KeyError, ValueError):
            return None
        return None

    def run(self, force: bool = False) -> int:
        """Execute the full automation update flow. Returns exit code."""
        console.print(f"[{T.RICH_STEP}]Checking for LattifAI updates...[/{T.RICH_STEP}]")

        try:
            current_v = importlib.metadata.version(self.package_name)
        except importlib.metadata.PackageNotFoundError:
            console.print(f"[{T.RICH_ERR}]Package '{self.package_name}' not found.[/{T.RICH_ERR}]")
            return 1

        latest_v = self.get_latest_version()

        if not latest_v:
            if force:
                # --force can bypass a failed version lookup and install directly
                console.print(
                    f"[{T.RICH_WARN}]Version lookup failed — proceeding with --force reinstall.[/{T.RICH_WARN}]"
                )
            else:
                console.print(
                    f"[{T.RICH_WARN}]Update server unreachable. "
                    f"Retry with --force to reinstall anyway.[/{T.RICH_WARN}]"
                )
                return 1
        else:
            is_newer = v_parse.parse(latest_v) > v_parse.parse(current_v)

            if not is_newer and not force:
                console.print(f"[{T.RICH_OK}]You are already using the latest version ({current_v}).[/{T.RICH_OK}]")
                return 0

            if is_newer:
                console.print(
                    f"[{T.RICH_STEP}]New version available:[/{T.RICH_STEP}]"
                    f" [{T.RICH_LABEL}]{current_v}[/{T.RICH_LABEL}]"
                    f" -> [{T.RICH_OK}]{latest_v}[/{T.RICH_OK}]"
                )
            else:
                console.print(f"[{T.RICH_STEP}]Reinstalling version {current_v} as requested...[/{T.RICH_STEP}]")

        return self._pip_install(force=force)

    def _pip_install(self, force: bool = False) -> int:
        """Run pip install and stream progress output."""
        pip_args = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--index-url",
            self.index_url,
            "--extra-index-url",
            self.extra_index_url,
            self.package_name,
        ]
        if force:
            pip_args.insert(-1, "--force-reinstall")

        try:
            result = subprocess.run(pip_args, capture_output=True, text=True, timeout=_PIP_TIMEOUT_SECS)
            if result.returncode != 0:
                console.print(f"[{T.RICH_ERR}]Upgrade failed (exit code {result.returncode}).[/{T.RICH_ERR}]")
                if result.stderr:
                    # Show last 10 lines of stderr for context
                    for line in result.stderr.strip().splitlines()[-10:]:
                        console.print(f"  [{T.RICH_DIM}]{line}[/{T.RICH_DIM}]")
                return 1
            console.print(f"[{T.RICH_OK}]Upgrade successful![/{T.RICH_OK}]")
            self.post_check()
            return 0
        except subprocess.TimeoutExpired:
            console.print(
                f"[{T.RICH_ERR}]pip timed out after {_PIP_TIMEOUT_SECS}s. "
                f"Check your network or run manually.[/{T.RICH_ERR}]"
            )
            return 1
        except OSError as exc:
            console.print(f"[{T.RICH_ERR}]Failed to run pip: {exc}[/{T.RICH_ERR}]")
            return 1

    def post_check(self):
        """Self-healing health check after update."""
        console.print(f"[{T.RICH_STEP}]Running environment health check...[/{T.RICH_STEP}]")
        accel = []

        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                accel.append("CUDA")
            if "CoreMLExecutionProvider" in providers:
                accel.append("CoreML")
        except ImportError:
            console.print(f"  [{T.RICH_DIM}]ONNXRuntime not installed, skipping engine check.[/{T.RICH_DIM}]")
            return
        except Exception as e:
            console.print(f"  [{T.RICH_ERR}]Health check error: {e}[/{T.RICH_ERR}]")
            return

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
            console.print(f"  GPU acceleration: [{T.RICH_OK}]Active ({', '.join(accel)})[/{T.RICH_OK}]")
        else:
            console.print(f"  GPU acceleration: [{T.RICH_WARN}]Not found (Using CPU)[/{T.RICH_WARN}]")


def update(force: bool = False) -> int:
    """
    Update LattifAI CLI and core components to the latest version.

    Checks for new versions via PyPI JSON API, installs from the private
    mirror (authoritative), and runs a health check to ensure hardware
    acceleration is working.

    Args:
        force: Force reinstall even if already at latest version, or if
               version discovery fails.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    updater = AutoUpdater()
    return updater.run(force=force)


def main():
    """Main entry point for lai-update script."""
    import argparse

    parser = argparse.ArgumentParser(description="Update LattifAI CLI")
    parser.add_argument("--force", action="store_true", help="Force update")
    args = parser.parse_args()
    sys.exit(update(force=args.force))
