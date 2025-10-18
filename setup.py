#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom setup.py to handle k2 installation after package installation.
"""

import os
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.bdist_wheel import bdist_wheel


def try_install_k2():
    """Try to install k2 dependency."""
    # Skip if explicitly disabled
    if os.environ.get('SKIP_K2_INSTALL'):
        print("\n[INFO] Skipping k2 installation (explicitly disabled)")
        print("[INFO] Run 'install-k2' after installation completes\n")
        return

    # Skip during wheel building
    if os.environ.get('BDIST_WHEEL_BUILDING'):
        print("\n[INFO] Skipping k2 installation during wheel build")
        print("[INFO] k2 will be installed after 'pip install' completes")
        print("[INFO] Or run 'install-k2' manually after installation\n")
        return

    # Skip during wheel building - check if pip module is available
    try:
        import pip
    except ImportError:
        print("\n[INFO] Skipping k2 installation (pip module not available)")
        print("[INFO] k2 will be installed after 'pip install' completes")
        print("[INFO] Or run 'install-k2' manually after installation\n")
        return

    try:
        print("\n" + "=" * 60)
        print("Installing k2 dependency...")
        print("=" * 60 + "\n")

        # Import and run the installation script
        sys.path.insert(0, os.path.dirname(__file__))
        from scripts.install_k2 import install_k2_main

        install_k2_main(dry_run=False)

        print("\n" + "=" * 60)
        print("k2 installation completed successfully!")
        print("=" * 60 + "\n")
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[WARNING] Failed to auto-install k2: {e}")
        print("[INFO] You can manually install k2 later by running:")
        print("    install-k2")
        print("=" * 60 + "\n")


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        try_install_k2()


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        try_install_k2()


class CustomBdistWheel(bdist_wheel):
    """Custom bdist_wheel command that skips k2 installation during wheel building."""

    def run(self):
        # Set environment variable to skip k2 installation
        os.environ['BDIST_WHEEL_BUILDING'] = '1'

        # Explicitly skip k2 during wheel build
        print("\n" + "=" * 70)
        print("  Building wheel for lattifai...")
        print("  ")
        print("  NOTE: k2 will NOT be installed during wheel build.")
        print("  After installing the package, please run:")
        print("  ")
        print("      install-k2")
        print("  ")
        print("  This command will install the appropriate k2 wheel for your system.")
        print("=" * 70 + "\n")

        try:
            bdist_wheel.run(self)
        finally:
            # Clean up environment variable
            os.environ.pop('BDIST_WHEEL_BUILDING', None)


if __name__ == "__main__":
    setup(
        cmdclass={
            "develop": PostDevelopCommand,
            "install": PostInstallCommand,
            "bdist_wheel": CustomBdistWheel,
        },
    )
