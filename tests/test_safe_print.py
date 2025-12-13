"""Test safe_print function for Windows Unicode handling."""

import sys

from lattifai.utils import safe_print


def test_safe_print_with_emojis():
    """Test that safe_print handles emojis correctly."""
    # Test with emoji
    test_text = "🎤 Starting transcription..."

    try:
        # This should not raise UnicodeEncodeError
        safe_print(test_text)
        print(f"✓ safe_print handled emoji successfully")
    except UnicodeEncodeError as e:
        print(f"✗ safe_print failed with UnicodeEncodeError: {e}")
        raise


def test_safe_print_with_regular_text():
    """Test that safe_print handles regular ASCII text correctly."""
    test_text = "Starting transcription..."

    try:
        safe_print(test_text)
        print(f"✓ safe_print handled regular text successfully")
    except Exception as e:
        print(f"✗ safe_print failed with exception: {e}")
        raise


if __name__ == "__main__":
    print("Testing safe_print function...")
    print(f"Platform: {sys.platform}")
    print(f"Stdout encoding: {sys.stdout.encoding}\n")

    test_safe_print_with_regular_text()
    test_safe_print_with_emojis()

    print("\n✓ All tests passed!")
