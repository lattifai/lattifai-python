#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic tests for lattifai package
"""

import sys


def test_import():
    """Test that the package can be imported successfully"""
    try:
        import lattifai

        assert lattifai is not None
        print('✓ Package import test passed')
        return True
    except ImportError as e:
        print(f'✗ Failed to import lattifai: {e}')
        return False


def test_package_structure():
    """Test basic package structure"""
    try:
        import lattifai

        # Check if package has expected attributes
        assert hasattr(lattifai, '__version__'), 'Package should have a __version__ attribute'
        print(f'✓ Package structure test passed (version: {lattifai.__version__})')
        return True
    except (ImportError, AssertionError) as e:
        print(f'✗ Package structure test failed: {e}')
        return False


def test_LattifAI_import():
    """Test that base modules can be imported without complex dependencies"""
    try:
        from lattifai import LattifAIError

        assert LattifAIError is not None
        print('✓ Successfully imported LattifAIError')
        return True
    except ImportError as e:
        print(f'✗ Failed to import lattifai.LattifAIError: {e}')
        return False


def test_io_imports():
    """Test that I/O modules can be imported"""
    try:
        from lattifai.io import SubtitleIO, Supervision

        assert Supervision is not None
        assert SubtitleIO is not None
        print('✓ Successfully imported I/O modules')
        return True
    except ImportError as e:
        print(f'✗ Failed to import I/O modules: {e}')
        return False


def test_client_class_exists():
    """Test that LattifAI client class can be imported"""
    try:
        from lattifai import LattifAI

        assert LattifAI is not None
        print('✓ Successfully imported LattifAI client class')
        return True
    except ImportError as e:
        print(f'✗ Failed to import LattifAI client class: {e}')
        return False


def test_alignment_method_exists():
    """Test that alignment method exists on LattifAI class"""
    try:
        from lattifai import LattifAI

        assert hasattr(LattifAI, 'alignment'), 'LattifAI should have alignment method'
        print('✓ LattifAI.alignment method exists')
        return True
    except (ImportError, AssertionError) as e:
        print(f'✗ alignment method check failed: {e}')
        return False


if __name__ == '__main__':
    print('Running basic lattifai tests...')

    results = []
    results.append(test_import())
    results.append(test_package_structure())
    results.append(test_LattifAI_import())
    results.append(test_io_imports())
    results.append(test_client_class_exists())
    results.append(test_alignment_method_exists())

    passed = sum(results)
    total = len(results)

    print(f'\nTest Results: {passed}/{total} tests passed')

    if passed == total:
        print('🎉 All basic tests passed!')
        sys.exit(0)
    else:
        print('❌ Some tests failed!')
        sys.exit(1)
