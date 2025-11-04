#!/usr/bin/env python3
"""Demonstration script showing LattifAI error handling in action."""

import os


def demonstrate_audio_error():
    """Demonstrate audio-related error handling."""
    print("🎵 Demonstrating audio error handling...")

    from lattifai.errors import AudioLoadError

    try:
        # Simulate an audio loading error
        raise AudioLoadError(
            "/path/to/missing_file.mp3",
            FileNotFoundError("No such file or directory"),
            context={"file_size_expected": "unknown", "format": "mp3"},
        )
    except AudioLoadError as e:
        print("Caught AudioLoadError:")
        print(f"Error code: {e.error_code}")
        print(f'Audio path: {e.context.get("audio_path")}')
        print(f'Original error: {e.context.get("original_error")}')
        print("Support info included in error message:")
        print(str(e))
        print()


def demonstrate_configuration_error():
    """Demonstrate configuration error with realistic scenario."""
    print("⚙️ Demonstrating configuration error handling...")

    from lattifai.errors import ConfigurationError

    # Simulate missing API key scenario
    original_key = os.environ.get("LATTIFAI_API_KEY")
    if "LATTIFAI_API_KEY" in os.environ:
        del os.environ["LATTIFAI_API_KEY"]

    try:
        # This mimics what happens in the real client
        api_key = os.environ.get("LATTIFAI_API_KEY")
        if api_key is None:
            raise ConfigurationError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the LATTIFAI_API_KEY environment variable"
            )
    except ConfigurationError as e:
        print("Caught ConfigurationError:")
        print(f"Error code: {e.error_code}")
        print("This is what users would see:")
        print("-" * 60)
        print(str(e))
        print("-" * 60)
        print()
    finally:
        # Restore environment
        if original_key:
            os.environ["LATTIFAI_API_KEY"] = original_key


def demonstrate_alignment_error():
    """Demonstrate alignment error with full context."""
    print("🔗 Demonstrating alignment error handling...")

    from lattifai.errors import LatticeDecodingError

    try:
        # Simulate a lattice decoding error
        raise LatticeDecodingError(
            "lattice_12345",
            RuntimeError("Failed to decode path due to invalid states"),
            context={"num_paths": 42, "beam_size": 100, "audio_duration": 30.5},
        )
    except LatticeDecodingError as e:
        print("Caught LatticeDecodingError:")
        print(f'Lattice ID: {e.context.get("lattice_id")}')
        print(f"Context: {e.context}")
        print("User-facing error message:")
        print("-" * 60)
        print(str(e))
        print("-" * 60)
        print()


def demonstrate_dependency_error():
    """Demonstrate dependency error with installation guidance."""
    print("📦 Demonstrating dependency error handling...")

    from lattifai.errors import DependencyError

    try:
        # Simulate missing k2 dependency
        raise DependencyError(
            "k2",
            install_command="pip install install-k2 && python -m install_k2",
            context={"import_attempted": "k2", "system": "macOS"},
        )
    except DependencyError as e:
        print("Caught DependencyError:")
        print(f'Missing dependency: {e.context.get("dependency_name")}')
        print(f'Install command: {e.context.get("install_command")}')
        print("User-facing error message:")
        print("-" * 60)
        print(str(e))
        print("-" * 60)
        print()


def demonstrate_text_processing_error():
    """Demonstrate text processing error with content preview."""
    print("📝 Demonstrating text processing error handling...")

    from lattifai.errors import LatticeEncodingError

    # Create a long text to show truncation
    long_text = "This is a very long subtitle text that contains many words and sentences. " * 20

    try:
        raise LatticeEncodingError(
            long_text, ValueError("Invalid character sequence in text"), context={"language": "en", "num_sentences": 15}
        )
    except LatticeEncodingError as e:
        print("Caught LatticeEncodingError:")
        print(f'Text length: {e.context.get("text_content_length")}')
        print(f'Text preview: {e.context.get("text_preview")}')
        print(f'Language: {e.context.get("language")}')
        print("User-facing error message:")
        print("-" * 60)
        print(str(e))
        print("-" * 60)
        print()


def main():
    """Run all error handling demonstrations."""
    print("🎯 LattifAI Error Handling Demonstration\n")
    print("This shows how users will experience errors and get helpful guidance.\n")

    demonstrate_audio_error()
    demonstrate_configuration_error()
    demonstrate_alignment_error()
    demonstrate_dependency_error()
    demonstrate_text_processing_error()

    print("🎉 Error handling demonstration complete!")
    print("\n✨ Key benefits of this error system:")
    print("   • Clear, actionable error messages")
    print("   • Context information for debugging")
    print("   • Direct links to GitHub and Discord support")
    print("   • Specific guidance for common issues")
    print("   • Proper error hierarchy for catching specific issues")


if __name__ == "__main__":
    main()
