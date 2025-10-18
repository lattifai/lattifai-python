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


if __name__ == '__main__':
    print('Running basic lattifai tests...')

    results = []
    results.append(test_import())
    results.append(test_package_structure())
    results.append(test_LattifAI_import())

    passed = sum(results)
    total = len(results)

    print(f'\nTest Results: {passed}/{total} tests passed')

    if passed == total:
        print('🎉 All basic tests passed!')
        sys.exit(0)
    else:
        print('❌ Some tests failed!')
        sys.exit(1)
