"""Error handling and exception classes for LattifAI SDK."""

import traceback
from typing import Any, Dict, Optional

import colorful

# Error help messages
LATTICE_DECODING_FAILURE_HELP = (
    "Failed to decode lattice alignment. Possible reasons:\n\n"
    "1) Audio and text content mismatch:\n"
    "   - The transcript/subtitle does not accurately match the audio content\n"
    "   - Text may be from a different version or section of the audio\n"
    "   ⚠️  Note: Gemini transcription may occasionally skip large segments of audio, causing alignment failures.\n"
    "       We will detect and fix this issue in the next version.\n\n"
    "2) Unsupported audio type:\n"
    "   - Singing is not yet supported, this will be optimized in future versions\n\n"
    "💡 Troubleshooting tips:\n"
    "   • Verify the transcript matches the audio by listening to a few segments\n"
    "   • For YouTube videos, manually check if auto-generated transcript are accurate\n"
    "       • Consider using a different transcription source if Gemini results are incomplete"
)


class LattifAIError(Exception):
    """Base exception for LattifAI errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Initialize LattifAI error.

        Args:
            message: Error message
            error_code: Optional error code for categorization
            context: Optional context information about the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}

    def get_support_info(self) -> str:
        """Get support information for users."""
        return (
            f'\n{colorful.green("🔧 Need help? Here are two ways to get support:")}\n'
            f'   1. 📝 Create a GitHub issue: {colorful.green("https://github.com/lattifai/lattifai-python/issues")}\n'
            "      Please include:\n"
            "      - Your audio file format and duration\n"
            "      - The text/subtitle content you're trying to align\n"
            "      - This error message and stack trace\n"
            f'   2. 💬 Join our Discord community: {colorful.green("https://discord.gg/vzmTzzZgNu")}\n'
            "      Our team and community can help you troubleshoot\n"
        )

    def get_message(self) -> str:
        """Return formatted error message without support information."""
        base_message = f'{colorful.red(f"[{self.error_code}] {self.message}")}'
        if self.context:
            context_str = f'\n{colorful.yellow("Context:")} ' + ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_message += context_str
        return base_message

    def __str__(self) -> str:
        """Return formatted error message without support information.

        Note: Support info should be displayed explicitly at the CLI level,
        not automatically appended to avoid duplication when errors are re-raised.
        """
        return self.get_message()


class AudioProcessingError(LattifAIError):
    """Error during audio processing operations."""

    def __init__(self, message: str, audio_path: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if audio_path:
            context["audio_path"] = audio_path
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class AudioLoadError(AudioProcessingError):
    """Error loading or reading audio file."""

    def __init__(self, audio_path: str, original_error: Optional[Exception] = None, **kwargs):
        message = f"Failed to load audio file: {colorful.red(audio_path)}"
        if original_error:
            message += f" - {colorful.red(str(original_error))}"

        context = kwargs.get("context", {})
        context.update({"audio_path": audio_path, "original_error": str(original_error) if original_error else None})
        kwargs["context"] = context

        super().__init__(message, audio_path=audio_path, **kwargs)


class AudioFormatError(AudioProcessingError):
    """Error with audio format or codec."""

    def __init__(self, audio_path: str, format_issue: str, **kwargs):
        message = f"Audio format error for {colorful.red(audio_path)}: {colorful.red(format_issue)}"
        context = kwargs.get("context", {})
        context.update({"audio_path": audio_path, "format_issue": format_issue})
        kwargs["context"] = context
        super().__init__(message, audio_path=audio_path, **kwargs)


class SubtitleProcessingError(LattifAIError):
    """Error during subtitle/text processing operations."""

    def __init__(self, message: str, subtitle_path: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if subtitle_path:
            context["subtitle_path"] = subtitle_path
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class SubtitleParseError(SubtitleProcessingError):
    """Error parsing subtitle or text file."""

    def __init__(self, subtitle_path: str, parse_issue: str, **kwargs):
        message = f"Failed to parse subtitle file {subtitle_path}: {parse_issue}"
        context = kwargs.get("context", {})
        context.update({"subtitle_path": subtitle_path, "parse_issue": parse_issue})
        kwargs["context"] = context
        super().__init__(message, subtitle_path=subtitle_path, **kwargs)


class AlignmentError(LattifAIError):
    """Error during audio-text alignment process."""

    def __init__(self, message: str, audio_path: Optional[str] = None, subtitle_path: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if audio_path:
            context["audio_path"] = audio_path
        if subtitle_path:
            context["subtitle_path"] = subtitle_path
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class LatticeEncodingError(AlignmentError):
    """Error generating lattice graph from text."""

    def __init__(self, text_content: str, original_error: Optional[Exception] = None, **kwargs):
        message = "Failed to generate lattice graph from text"
        if original_error:
            message += f": {colorful.red(str(original_error))}"

        context = kwargs.get("context", {})
        context.update(
            {
                "text_content_length": len(text_content),
                "text_preview": text_content[:100] + "..." if len(text_content) > 100 else text_content,
                "original_error": str(original_error) if original_error else None,
            }
        )
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class LatticeDecodingError(AlignmentError):
    """Error decoding lattice alignment results."""

    def __init__(self, lattice_id: str, original_error: Optional[Exception] = None, **kwargs):
        message = f"Failed to decode lattice alignment results for lattice ID: {colorful.red(lattice_id)}"

        # Don't duplicate the help message if it's already in original_error
        if original_error and str(original_error) != LATTICE_DECODING_FAILURE_HELP:
            message += f" - {colorful.red(str(original_error))}"

        context = kwargs.get("context", {})
        # Don't store the entire help message in context to avoid duplication
        if original_error and str(original_error) != LATTICE_DECODING_FAILURE_HELP:
            context["original_error"] = str(original_error)
        context["lattice_id"] = lattice_id
        kwargs["context"] = context
        super().__init__(message, **kwargs)

    def get_message(self) -> str:
        """Return formatted error message with help text."""
        base_message = f'{colorful.red(f"[{self.error_code}]")} {self.message}'
        if self.context and self.context.get("lattice_id"):
            # Only show essential context (lattice_id), not the duplicated help message
            base_message += f'\n{colorful.yellow("Lattice ID:")} {self.context["lattice_id"]}'
        # Append help message once at the end
        base_message += f"\n\n{colorful.yellow(LATTICE_DECODING_FAILURE_HELP)}"
        return base_message


class ModelLoadError(LattifAIError):
    """Error loading AI model."""

    def __init__(self, model_name: str, original_error: Optional[Exception] = None, **kwargs):
        message = f"Failed to load model: {colorful.red(model_name)}"
        if original_error:
            message += f" - {colorful.red(str(original_error))}"

        context = kwargs.get("context", {})
        context.update({"model_name": model_name, "original_error": str(original_error) if original_error else None})
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class DependencyError(LattifAIError):
    """Error with required dependencies."""

    def __init__(self, dependency_name: str, install_command: Optional[str] = None, **kwargs):
        message = f"Missing required dependency: {colorful.red(dependency_name)}"
        if install_command:
            message += f"\nPlease install it using: {colorful.yellow(install_command)}"

        context = kwargs.get("context", {})
        context.update({"dependency_name": dependency_name, "install_command": install_command})
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class APIError(LattifAIError):
    """Error communicating with LattifAI API."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        context.update({"status_code": status_code, "response_text": response_text})
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class ConfigurationError(LattifAIError):
    """Error with client configuration."""

    def __init__(self, config_issue: str, **kwargs):
        message = f"Configuration error: {config_issue}"
        super().__init__(message, **kwargs)


def handle_exception(func):
    """Decorator to handle exceptions and convert them to LattifAI errors."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LattifAIError:
            # Re-raise LattifAI errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to LattifAI errors
            error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
            context = {
                "function": func.__name__,
                "original_exception": e.__class__.__name__,
                "traceback": traceback.format_exc(),
            }
            raise LattifAIError(error_msg, context=context) from e

    return wrapper
