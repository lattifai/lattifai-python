#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom setup.py to handle k2 installation after package installation.
"""

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        try:
            print("\n" + "=" * 60)
            print("Installing k2 dependency...")
            print("=" * 60 + "\n")
            from scripts.install_k2 import install_k2_main

            install_k2_main(dry_run=False)
        except Exception as e:
            print(f"\n[WARNING] Failed to auto-install k2: {e}")
            print("[INFO] You can manually install k2 later by running: install-k2")


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        try:
            print("\n" + "=" * 60)
            print("Installing k2 dependency...")
            print("=" * 60 + "\n")
            from scripts.install_k2 import install_k2_main

            install_k2_main(dry_run=False)
        except Exception as e:
            print(f"\n[WARNING] Failed to auto-install k2: {e}")
            print("[INFO] You can manually install k2 later by running: install-k2")


if __name__ == "__main__":
    setup(
        cmdclass={
            "develop": PostDevelopCommand,
            "install": PostInstallCommand,
        },
    )
