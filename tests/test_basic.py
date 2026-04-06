#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic tests for lattifai package (base installation includes alignment)
"""

import sys


def test_import():
    """Test that the package can be imported successfully"""
    import lattifai

    assert lattifai is not None


def test_package_structure():
    """Test basic package structure"""
    from importlib.metadata import version

    pkg_version = version("lattifai")
    assert pkg_version is not None, "Package should have a version"


def test_LattifAI_import():
    """Test that base modules can be imported"""
    from lattifai.errors import LattifAIError

    assert LattifAIError is not None


def test_io_imports():
    """Test that I/O modules can be imported"""
    from lattifai.caption import Caption, Supervision

    assert Supervision is not None
    assert Caption is not None


def test_client_class_exists():
    """Test that LattifAI client class can be imported"""
    from lattifai.client import LattifAI

    assert LattifAI is not None


def test_alignment_method_exists():
    """Test that alignment method exists on LattifAI class"""
    from lattifai.client import LattifAI

    assert hasattr(LattifAI, "alignment"), "LattifAI should have alignment method"


if __name__ == "__main__":
    # Run without pytest dependency for CI wheel smoke tests
    tests = [
        test_import,
        test_package_structure,
        test_LattifAI_import,
        test_io_imports,
        test_client_class_exists,
        test_alignment_method_exists,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    sys.exit(failed)
