"""Translation service configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from lattifai.config.llm import LLMConfig


@dataclass
class TranslationLLMConfig(LLMConfig):
    """LLM config bound to [translation.llm] section. Survives nemo_run reconstruction."""

    section: str = "translation.llm"
    fallback_model: Optional[str] = "gemini-3-flash-preview"


@dataclass
class TranslationConfig:
    """
    Translation service configuration.

    Settings for caption translation using various LLM providers.
    """

    _toml_section = "translation"

    llm: TranslationLLMConfig = field(default_factory=TranslationLLMConfig)
    """LLM provider configuration. Reads defaults from config.toml [translation]."""

    target_lang: str = "zh"
    """Target language code (BCP 47 / ISO 639-1).
    See lattifai.languages.SUPPORTED_LANGUAGES for the full list (55+ languages).
    Common codes: zh, zh-TW, en, ja, ko, es, fr, de, pt, ru, ar, hi, th, vi, id, tr."""

    source_lang: Optional[str] = None
    """Source language code. Auto-detected if None."""

    mode: Literal["quick", "normal", "refined"] = "normal"
    """Translation mode: 'quick' (direct), 'normal' (analyze+translate), 'refined' (analyze+translate+review)."""

    approach: Literal["rewrite", "translate"] = "rewrite"
    """Translation approach:
    - 'rewrite': Express the speaker's intent naturally in the target language.
      Prioritizes fluency, idiom adaptation, and emotional fidelity. Best for
      storytelling, casual content, and publication-quality caption/subtitles.
    - 'translate': Stay close to the original wording and structure.
      Prioritizes accuracy, completeness, and source fidelity. Best for
      technical content, language learning, and compliance caption/subtitles."""

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
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.context_lines < 0:
            raise ValueError("context_lines must be >= 0")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
