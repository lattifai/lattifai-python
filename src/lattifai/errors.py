"""Error handling and exception classes for LattifAI SDK."""

import functools
import traceback
from typing import Any, Dict, Optional

import colorful


def format_exception(e: "LattifAIError") -> str:
    """Format LattifAIError with filtered traceback (only lattifai frames)."""
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    filtered = []
    skip_next_code_line = False

    for i, line in enumerate(tb_lines):
        if skip_next_code_line:
            skip_next_code_line = False
            continue

        if line.startswith("Traceback") or not line.startswith("  File"):
            filtered.append(line)
        elif "lattifai" in line:
            filtered.append(line)
            if i + 1 < len(tb_lines) and tb_lines[i + 1].startswith("    "):
                filtered.append(tb_lines[i + 1])
                skip_next_code_line = True
        elif i + 1 < len(tb_lines) and tb_lines[i + 1].startswith("    "):
            skip_next_code_line = True

    return "".join(filtered)


def _merge_context(kwargs: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Merge updates into kwargs['context'], creating it if needed."""
    context = kwargs.setdefault("context", {})
    context.update(updates)


# Error help messages
LATTICE_DECODING_FAILURE_HELP = (
    "Failed to decode lattice alignment. Possible reasons:\n\n"
    "1) Media(Audio/Video) and text content mismatch:\n"
    "   - The transcript/caption does not accurately match the media content\n"
    "   - Text may be from a different version or section of the media\n"
    "2) Text formatting issues:\n"
    "   - Special characters, HTML entities, or unusual punctuation may cause alignment failures\n"
    "   - Text normalization is enabled by default (caption.normalize_text=True)\n"
    "     If you disabled it, try re-enabling: caption.normalize_text=True\n"
    "3) Unsupported media type:\n"
    "   - Singing is not yet supported, this will be optimized in future versions\n\n"
    "💡 Troubleshooting tips:\n"
    "   • Text normalization is enabled by default to handle special characters\n"
    "     (no action needed unless you explicitly set caption.normalize_text=False)\n"
    "   • Verify the transcript matches the media by listening to a few segments\n"
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
            "      - The text/caption content you're trying to align\n"
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

    def __init__(self, message: str, media_path: Optional[str] = None, **kwargs):
        if media_path:
            _merge_context(kwargs, {"media_path": media_path})
        super().__init__(message, **kwargs)


class AudioLoadError(AudioProcessingError):
    """Error loading or reading audio file."""

    def __init__(self, media_path: str, original_error: Optional[Exception] = None, **kwargs):
        message = f"Failed to load audio file: {colorful.red(media_path)}"
        if original_error:
            message += f" - {colorful.red(str(original_error))}"
        _merge_context(
            kwargs, {"media_path": media_path, "original_error": str(original_error) if original_error else None}
        )
        super().__init__(message, media_path=media_path, **kwargs)


class AudioFormatError(AudioProcessingError):
    """Error with audio format or codec."""

    def __init__(self, media_path: str, format_issue: str, **kwargs):
        message = f"Audio format error for {colorful.red(media_path)}: {colorful.red(format_issue)}"
        _merge_context(kwargs, {"media_path": media_path, "format_issue": format_issue})
        super().__init__(message, media_path=media_path, **kwargs)


class CaptionProcessingError(LattifAIError):
    """Error during caption/text processing operations."""

    def __init__(self, message: str, caption_path: Optional[str] = None, **kwargs):
        if caption_path:
            _merge_context(kwargs, {"caption_path": caption_path})
        super().__init__(message, **kwargs)


class CaptionParseError(CaptionProcessingError):
    """Error parsing caption or text file."""

    def __init__(self, caption_path: str, parse_issue: str, **kwargs):
        message = f"Failed to parse caption file {caption_path}: {parse_issue}"
        _merge_context(kwargs, {"caption_path": caption_path, "parse_issue": parse_issue})
        super().__init__(message, caption_path=caption_path, **kwargs)


class AlignmentError(LattifAIError):
    """Error during audio-text alignment process."""

    def __init__(self, message: str, media_path: Optional[str] = None, caption_path: Optional[str] = None, **kwargs):
        updates = {}
        if media_path:
            updates["media_path"] = media_path
        if caption_path:
            updates["caption_path"] = caption_path
        if updates:
            _merge_context(kwargs, updates)
        super().__init__(message, **kwargs)


class LatticeEncodingError(AlignmentError):
    """Error generating lattice graph from text."""

    def __init__(self, text_content: str, original_error: Optional[Exception] = None, **kwargs):
        message = "Failed to generate lattice graph from text"
        if original_error:
            message += f": {colorful.red(str(original_error))}"
        text_preview = text_content[:100] + "..." if len(text_content) > 100 else text_content
        _merge_context(
            kwargs,
            {
                "text_content_length": len(text_content),
                "text_preview": text_preview,
                "original_error": str(original_error) if original_error else None,
            },
        )
        super().__init__(message, **kwargs)


class LatticeDecodingError(AlignmentError):
    """Error decoding lattice alignment results."""

    def __init__(
        self,
        lattice_id: str,
        message: Optional[str] = None,
        original_error: Optional[Exception] = None,
        skip_help: bool = False,
        **kwargs,
    ):
        message = message or f"Failed to decode lattice alignment results for lattice ID: {colorful.red(lattice_id)}"

        error_str = str(original_error) if original_error else None
        is_help_message = error_str == LATTICE_DECODING_FAILURE_HELP

        if original_error and not is_help_message:
            message += f" - {colorful.red(error_str)}"

        context_updates = {"lattice_id": lattice_id}
        if original_error and not is_help_message:
            context_updates["original_error"] = error_str
        _merge_context(kwargs, context_updates)

        super().__init__(message, **kwargs)
        self.skip_help = skip_help

    def get_message(self) -> str:
        """Return formatted error message with help text."""
        base_message = f'{colorful.red(f"[{self.error_code}]")} {self.message}'
        if self.context and self.context.get("lattice_id"):
            # Only show essential context (lattice_id), not the duplicated help message
            base_message += f'\n{colorful.yellow("Lattice ID:")} {self.context["lattice_id"]}'
        # Append help message only if not skipped (e.g., when anomaly info is provided)
        if not self.skip_help:
            base_message += f"\n\n{colorful.yellow(LATTICE_DECODING_FAILURE_HELP)}"
        return base_message


class ModelLoadError(LattifAIError):
    """Error loading AI model."""

    def __init__(self, model_name: str, original_error: Optional[Exception] = None, **kwargs):
        message = f"Failed to load model: {colorful.red(model_name)}"
        if original_error:
            message += f" - {colorful.red(str(original_error))}"
        _merge_context(
            kwargs, {"model_name": model_name, "original_error": str(original_error) if original_error else None}
        )
        super().__init__(message, **kwargs)


class DependencyError(LattifAIError):
    """Error with required dependencies."""

    def __init__(self, dependency_name: str, install_command: Optional[str] = None, **kwargs):
        message = f"Missing required dependency: {colorful.red(dependency_name)}"
        if install_command:
            message += f"\nPlease install it using: {colorful.yellow(install_command)}"
        _merge_context(kwargs, {"dependency_name": dependency_name, "install_command": install_command})
        super().__init__(message, **kwargs)


class APIError(LattifAIError):
    """Error communicating with LattifAI API."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None, **kwargs):
        _merge_context(kwargs, {"status_code": status_code, "response_text": response_text})
        super().__init__(message, **kwargs)


class ConfigurationError(LattifAIError):
    """Error with client configuration."""

    def __init__(self, config_issue: str, **kwargs):
        message = f"Configuration error: {config_issue}"
        super().__init__(message, **kwargs)


class QuotaExceededError(APIError):
    """Error when user quota or API key limit is exceeded."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=402, **kwargs)


def handle_exception(func):
    """Decorator to handle exceptions and convert them to LattifAI errors."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LattifAIError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
            context = {
                "function": func.__name__,
                "original_exception": e.__class__.__name__,
                "traceback": traceback.format_exc(),
            }
            raise LattifAIError(error_msg, context=context) from e

    return wrapper
