"""Translation service configuration for LattifAI."""

import os
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TranslationConfig:
    """
    Translation service configuration.

    Settings for caption translation using various LLM providers.
    """

    model_name: str = "gemini-3-flash-preview"
    """Model name for translation."""

    provider: Literal["gemini", "openai"] = "gemini"
    """LLM provider: 'gemini' (Google GenAI) or 'openai' (OpenAI-compatible)."""

    api_key: Optional[str] = None
    """API key. If None, reads from GEMINI_API_KEY or OPENAI_API_KEY environment variable."""

    api_base_url: Optional[str] = None
    """Base URL for OpenAI-compatible API (e.g. http://localhost:8000/v1)."""

    target_lang: str = "zh"
    """Target language code (BCP 47 / ISO 639-1).
    See lattifai.languages.SUPPORTED_LANGUAGES for the full list (55+ languages).
    Common codes: zh, zh-TW, en, ja, ko, es, fr, de, pt, ru, ar, hi, th, vi, id, tr."""

    source_lang: Optional[str] = None
    """Source language code. Auto-detected if None."""

    mode: Literal["quick", "normal", "refined"] = "normal"
    """Translation mode: 'quick' (direct), 'normal' (analyze+translate), 'refined' (analyze+translate+review)."""

    bilingual: bool = True
    """Output bilingual captions (original + translation)."""

    style: str = "technical"
    """Translation style hint (e.g. 'storytelling', 'formal', 'casual', 'technical')."""

    batch_size: int = 30
    """Number of caption segments per API call."""

    context_lines: int = 5
    """Number of surrounding lines for context in each batch."""

    max_concurrent: int = 5
    """Maximum number of concurrent batch requests."""

    glossary_file: Optional[str] = None
    """Path to custom glossary file (YAML or Markdown)."""

    save_artifacts: bool = False
    """Save intermediate artifacts (analysis, prompts, drafts, review notes, revisions)."""

    artifacts_dir: Optional[str] = None
    """Directory for artifacts. Defaults to output file's parent directory."""

    ask_refine_after_normal: bool = True
    """Prompt interactively to continue with refined review after normal mode translation."""

    auto_refine_after_normal: bool = False
    """Automatically run refined review after normal mode without prompting."""

    verbose: bool = False
    """Enable verbose logging."""

    def __post_init__(self):
        """Validate and auto-populate configuration."""
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))

        if self.api_key is None:
            if self.provider == "gemini":
                self.api_key = os.environ.get("GEMINI_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.context_lines < 0:
            raise ValueError("context_lines must be >= 0")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
