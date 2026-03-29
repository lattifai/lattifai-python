"""Implementation of 'lai update' for LattifAI CLI with nemo_run."""

import subprocess
import sys

import nemo_run as run
import pkg_resources
import requests
from packaging import version as v_parse
from rich.console import Console

console = Console()


class AutoUpdater:
    """Automated updater for LattifAI package."""

    def __init__(self, package_name="lattifai"):
        self.package_name = package_name
        # Priority: Local Mirror (if configured) > PyPI
        self.index_urls = ["https://lattifai.github.io/pypi/simple/", "https://pypi.org/simple"]

    def get_latest_version(self):
        """Fetch the latest version from PyPI."""
        try:
            response = requests.get(f"https://pypi.org/pypi/{self.package_name}/json", timeout=5)
            if response.status_code == 200:
                return response.json()["info"]["version"]
        except Exception:
            return None
        return None

    def run(self, force: bool = False):
        """Execute the full automation update flow."""
        console.print("🔍 [bold blue]Checking for OmniCaptions (lattifai) updates...[/bold blue]")

        try:
            current_v = pkg_resources.get_distribution(self.package_name).version
        except pkg_resources.DistributionNotFound:
            console.print(f"[bold red]Package '{self.package_name}' not found in the current environment.[/bold red]")
            return

        latest_v = self.get_latest_version()
        if not latest_v:
            console.print("[dark_orange]Update server unreachable. Skipping version check.[/dark_orange]")
            return

        is_newer = v_parse.parse(latest_v) > v_parse.parse(current_v)

        if not is_newer and not force:
            console.print(f"✅ [bold green]You are already using the latest version ({current_v}).[/bold green]")
            return

        if is_newer:
            console.print(f"🚀 New version available: [bold]{current_v}[/bold] -> [bold green]{latest_v}[/bold green]")
        else:
            console.print(f"🔄 Reinstalling version [bold]{current_v}[/bold] as requested...")

        # Construct pip upgrade command
        pip_args = [sys.executable, "-m", "pip", "install", "--upgrade", self.package_name]
        for url in self.index_urls:
            pip_args.extend(["--extra-index-url", url])

        try:
            subprocess.check_call(pip_args)
            console.print("✨ [bold green]Upgrade successful![/bold green]")
            self.post_check()
        except subprocess.CalledProcessError:
            console.print("[bold red]Upgrade failed.[/bold red] Please check your network or permissions.")

    def post_check(self):
        """Self-healing health check after update."""
        console.print("🧪 Running environment health check...")
        try:
            # Check ONNXRuntime GPU provider (core released without k2 dependency)
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                console.print("🟢 GPU acceleration: [bold green]Active (CUDA)[/bold green]")
            elif "CoreMLExecutionProvider" in providers:
                console.print("🟢 GPU acceleration: [bold green]Active (CoreML)[/bold green]")
            else:
                console.print("🟡 GPU acceleration: [dark_orange]Not found (Using CPU)[/dark_orange]")
        except ImportError:
            console.print("⚪ [dim]ONNXRuntime not installed, skipping engine check.[/dim]")
        except Exception as e:
            console.print(f"🔴 Health check encountered an error: {str(e)}")


@run.cli.entrypoint(name="run", namespace="update")
def update(force: bool = False):
    """
    Update LattifAI CLI and core components to the latest version.

    This command checks for new versions of the 'lattifai' package on PyPI
    and the private mirror, performs an automated upgrade, and runs
    a health check to ensure hardware acceleration is working.

    Args:
        force: Force reinstall even if already at latest version.
    """
    updater = AutoUpdater()
    updater.run(force=force)


def main():
    """Main entry point for lai-update script."""
    import argparse

    parser = argparse.ArgumentParser(description="Update LattifAI CLI")
    parser.add_argument("--force", action="store_true", help="Force update")
    args = parser.parse_args()
    update(force=args.force)
