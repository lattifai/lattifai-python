"""CLI tool to install lai-app (frontend web application)."""

import platform
import subprocess
import sys
from pathlib import Path

from lattifai.utils import safe_print


def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        subprocess.run([cmd, "--version"], check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_nodejs():
    """Install Node.js based on the operating system."""
    system = platform.system().lower()

    safe_print("📦 Node.js not found. Installing Node.js...\n")

    try:
        if system == "darwin":  # macOS
            # Check if Homebrew is installed
            if check_command_exists("brew"):
                safe_print("🍺 Using Homebrew to install Node.js...")
                subprocess.run(["brew", "install", "node"], check=True)
                safe_print("✓ Node.js installed via Homebrew\n")
            else:
                safe_print("❌ Homebrew not found.")
                print("   Please install Homebrew first:")
                print(
                    '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                )
                print("\n   Or install Node.js manually from: https://nodejs.org/")
                sys.exit(1)

        elif system == "linux":
            # Try common package managers
            if check_command_exists("apt"):
                safe_print("🐧 Using apt to install Node.js...")
                subprocess.run(["sudo", "apt", "update"], check=True)
                subprocess.run(["sudo", "apt", "install", "-y", "nodejs", "npm"], check=True)
                safe_print("✓ Node.js installed via apt\n")
            elif check_command_exists("yum"):
                safe_print("🐧 Using yum to install Node.js...")
                subprocess.run(["sudo", "yum", "install", "-y", "nodejs", "npm"], check=True)
                safe_print("✓ Node.js installed via yum\n")
            elif check_command_exists("dnf"):
                safe_print("🐧 Using dnf to install Node.js...")
                subprocess.run(["sudo", "dnf", "install", "-y", "nodejs", "npm"], check=True)
                safe_print("✓ Node.js installed via dnf\n")
            elif check_command_exists("pacman"):
                safe_print("🐧 Using pacman to install Node.js...")
                subprocess.run(["sudo", "pacman", "-S", "--noconfirm", "nodejs", "npm"], check=True)
                safe_print("✓ Node.js installed via pacman\n")
            else:
                safe_print("❌ No supported package manager found (apt/yum/dnf/pacman).")
                print("   Please install Node.js manually from: https://nodejs.org/")
                sys.exit(1)

        elif system == "windows":
            safe_print("❌ Automatic installation on Windows is not supported.")
            print("   Please download and install Node.js from: https://nodejs.org/")
            print("   Then run this command again.")
            sys.exit(1)

        else:
            safe_print(f"❌ Unsupported operating system: {system}")
            print("   Please install Node.js manually from: https://nodejs.org/")
            sys.exit(1)

        # Verify installation
        if not check_command_exists("npm"):
            safe_print("❌ Node.js installation verification failed.")
            print("   Please restart your terminal and try again.")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        safe_print(f"\n❌ Error during Node.js installation: {e}")
        print("   Please install Node.js manually from: https://nodejs.org/")
        sys.exit(1)


def main():
    """Install lai-app Node.js application."""
    # Get the app directory relative to this package
    app_dir = Path(__file__).parent.parent.parent.parent / "app"

    if not app_dir.exists():
        safe_print(f"❌ Error: app directory not found at {app_dir}")
        print("   Make sure you're in the lattifai-python repository.")
        sys.exit(1)

    safe_print("🚀 Installing lai-app (LattifAI Web Application)...\n")

    # Check if npm is installed, if not, install Node.js
    if not check_command_exists("npm"):
        install_nodejs()
    else:
        npm_version = subprocess.run(["npm", "--version"], capture_output=True, text=True, check=True).stdout.strip()
        safe_print(f"✓ npm is already installed (v{npm_version})\n")

    # Change to app directory and run installation
    try:
        safe_print(f"📁 Working directory: {app_dir}\n")

        # Install dependencies
        safe_print("📦 Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=app_dir, check=True)
        safe_print("✓ Dependencies installed\n")

        # Build the application
        safe_print("🔨 Building application...")
        subprocess.run(["npm", "run", "build"], cwd=app_dir, check=True)
        safe_print("✓ Application built\n")

        # Link globally
        safe_print("🔗 Linking lai-app command globally...")
        subprocess.run(["npm", "link"], cwd=app_dir, check=True)
        safe_print("✓ lai-app command linked globally\n")

        safe_print("=" * 60)
        safe_print("✅ lai-app installed successfully!")
        safe_print("=" * 60)
        safe_print("\n🎉 You can now run:")
        print("   lai-app              # Start the web application")
        print("   lai-app --help       # Show help")
        print("   lai-app --port 8080  # Use custom port")
        safe_print("\n📖 For more information, see app/CLI_USAGE.md\n")

    except subprocess.CalledProcessError as e:
        safe_print(f"\n❌ Error during installation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
