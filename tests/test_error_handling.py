#!/usr/bin/env python3
"""Test script for LattifAI error handling system."""

import os
import sys

from lattifai.errors import (
    AlignmentError,
    APIError,
    AudioFormatError,
    AudioLoadError,
    ConfigurationError,
    DependencyError,
    LatticeDecodingError,
    LatticeEncodingError,
    LattifAIError,
    ModelLoadError,
    SubtitleProcessingError,
)


def test_basic_error_functionality():
    """Test basic error functionality and inheritance."""
    print("🧪 Testing basic error functionality...")

    # Test base LattifAI error
    try:
        raise LattifAIError("Test base error", error_code="TEST_001")
    except LattifAIError as e:
        print(f"✓ LattifAIError caught: {e.error_code}")
        # Support info is in get_support_info(), not in __str__
        support_info = e.get_support_info()
        assert "github.com" in support_info
        assert "discord.gg" in support_info

    # Test specific error types
    try:
        raise AudioLoadError("/path/to/nonexistent.wav", Exception("File not found"))
    except AudioLoadError as e:
        print(f"✓ AudioLoadError caught with context: {e.context}")
        assert e.context["media_path"] == "/path/to/nonexistent.wav"

    # Test configuration error
    try:
        raise ConfigurationError("Missing API key")
    except ConfigurationError as e:
        print(f"✓ ConfigurationError caught: {e.message}")

    print("✅ Basic error functionality tests passed!\n")


def test_error_inheritance():
    """Test that all errors inherit from LattifAIError."""
    print("🧪 Testing error inheritance...")

    error_classes = [
        AudioLoadError,
        AudioFormatError,
        SubtitleProcessingError,
        AlignmentError,
        LatticeEncodingError,
        LatticeDecodingError,
        ModelLoadError,
        DependencyError,
        APIError,
        ConfigurationError,
    ]

    for error_class in error_classes:
        assert issubclass(error_class, LattifAIError), f"{error_class.__name__} doesn't inherit from LattifAIError"
        print(f"✓ {error_class.__name__} inherits from LattifAIError")

    print("✅ Error inheritance tests passed!\n")


def test_support_info():
    """Test that error messages include support information."""
    print("🧪 Testing support information...")

    error = LattifAIError("Test error with support info")
    support_info = error.get_support_info()

    # Check for GitHub issue link
    assert "github.com" in support_info, "GitHub link not found in support info"
    assert "issues" in support_info, "Issues link not found in support info"

    # Check for Discord link
    assert "discord.gg/vzmTzzZgNu" in support_info, "Discord link not found in support info"

    # Check for guidance text
    assert "audio file format" in support_info, "Audio format guidance not found"
    assert "text/subtitle content" in support_info, "Text content guidance not found"

    print("✓ Support information included in get_support_info()")
    print("✅ Support information tests passed!\n")


def test_context_propagation():
    """Test that context information is properly propagated."""
    print("🧪 Testing context propagation...")

    # Test AudioLoadError with context
    audio_error = AudioLoadError(
        "/test/audio.mp3", Exception("Permission denied"), context={"file_size": 1024, "format": "mp3"}
    )

    assert audio_error.context["media_path"] == "/test/audio.mp3"
    assert audio_error.context["file_size"] == 1024
    assert audio_error.context["format"] == "mp3"
    assert "Permission denied" in audio_error.context["original_error"]

    print("✓ Context information properly stored and accessible")

    # Test LatticeEncodingError with text preview
    text_content = "This is a very long text that should be truncated in the context preview. " * 10
    lattice_error = LatticeEncodingError(text_content, Exception("Tokenization failed"))

    assert lattice_error.context["text_content_length"] == len(text_content)
    assert len(lattice_error.context["text_preview"]) <= 103  # 100 chars + "..."
    assert lattice_error.context["text_preview"].endswith("...")

    print("✓ Text content properly truncated in context")
    print("✅ Context propagation tests passed!\n")


def test_configuration_error_simulation():
    """Test configuration error similar to real usage."""
    print("🧪 Testing configuration error simulation...")

    # Simulate missing API key
    original_env = os.environ.get("LATTIFAI_API_KEY")
    if "LATTIFAI_API_KEY" in os.environ:
        del os.environ["LATTIFAI_API_KEY"]

    try:
        # This would be similar to what happens in the client
        api_key = os.environ.get("LATTIFAI_API_KEY")
        if api_key is None:
            raise ConfigurationError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the LATTIFAI_API_KEY environment variable"
            )
    except ConfigurationError as e:
        print("✓ Configuration error properly raised for missing API key")
        assert "LATTIFAI_API_KEY" in str(e)
    finally:
        # Restore original environment
        if original_env is not None:
            os.environ["LATTIFAI_API_KEY"] = original_env

    print("✅ Configuration error simulation passed!\n")


def main():
    """Run all error handling tests."""
    print("🚀 Starting LattifAI Error Handling Tests\n")

    try:
        test_basic_error_functionality()
        test_error_inheritance()
        test_support_info()
        test_context_propagation()
        test_configuration_error_simulation()

        print("🎉 All error handling tests passed successfully!")
        print("\n📝 Error handling system ready for use!")
        print("   - Users will get helpful error messages with support links")
        print("   - GitHub issue creation guidance is included")
        print("   - Discord community link is provided")
        print("   - Context information helps with debugging")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
