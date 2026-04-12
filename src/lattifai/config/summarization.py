"""Summarization service configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from lattifai.config.llm import LLMConfig


@dataclass
class SummarizationConfig:
    """Summarization service configuration.

    Controls LLM provider, output language, summary length, chunking
    behaviour for long inputs, and rendering format.
    """

    _toml_section = "summarization"

    llm: LLMConfig = field(default_factory=lambda: LLMConfig(model_name="gemini-2.5-flash"))
    """LLM provider configuration (provider, model, api_key, api_base_url)."""

    lang: str = "en"
    """Output language code (BCP 47 / ISO 639-1).
    Common codes: en, zh, ja, ko, es, fr, de."""

    length: Literal["auto", "short", "medium", "long"] = "auto"
    """Summary length preset:
    - auto: select based on input length (default)
    - short: 2-4 sentence summary, 3-5 key points
    - medium: 1-3 paragraph summary, 5-8 key points
    - long: 3-6 paragraph summary, 8-12 key points"""

    output_format: Literal["markdown", "json"] = "markdown"
    """Output rendering format."""

    source_lang: Optional[str] = None
    """Source content language hint. Auto-detected if None."""

    max_input_chars: int = 24000
    """Character threshold above which map-reduce chunking is triggered."""

    chunk_chars: int = 12000
    """Per-chunk character budget for map-reduce."""

    max_chunks: int = 12
    """Safety cap on the number of chunks processed."""

    overlap_chars: int = 400
    """Overlap between adjacent chunks to reduce boundary information loss."""

    include_chapters: bool = True
    """Inject chapter information into the prompt when available."""

    include_metadata: bool = True
    """Inject source metadata (title, channel, duration) into the prompt."""

    temperature: float = 0.2
    """LLM sampling temperature. Lower = more deterministic."""

    verbose: bool = False
    """Enable verbose diagnostics logging."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_input_chars < 2000:
            raise ValueError("max_input_chars must be >= 2000")
        if self.chunk_chars < 1000:
            raise ValueError("chunk_chars must be >= 1000")
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if not 0 <= self.overlap_chars < self.chunk_chars:
            raise ValueError("overlap_chars must be >= 0 and < chunk_chars")
