# CHANGELOG


## [1.5.8] - 2026-04-26

### Features
- CLI pre-flight warns before the stored trial key expires (or has expired), so users see the issue before the backend rejects with 401. Wired into backend-bound subcommands (alignment / youtube / transcribe / summarize / translate / diarize / serve); local-only commands stay silent.
- `SummarizationConfig.llm` now reads the `[summarization]` section of `config.toml` and falls back to the `SUMMARIZATION_MODEL_NAME` env var, so operators can route summarization to any OpenAI-compatible endpoint (SiliconFlow, vLLM, …) without code edits. Default stays Gemini.

### Tests
- Drop VTT dedup expectations for explicit speakers in multi-speaker E2E roundtrips (matches `lattifai-captions>=0.4.11` behavior; explicit speaker tags are emitted unconditionally).

### CI
- Release Tests workflow now plumbs optional `OPENAI_API_KEY` / `OPENAI_API_BASE_URL` / `SUMMARIZATION_MODEL_NAME` secrets to the test environment so the summarization integration suite can avoid Gemini 503 / malformed-JSON flakiness.


## [1.5.7] - 2026-04-26

### Features
- Auto-suppress duplicate output for lyrics input in alignment
- Add `G` option to keep genuine duplicate blocks during dedup
- Currency / percent / thousands aware tokenization in alignment
- `AudioData.stats()` plus CLI for amplitude distribution

### Fixes
- Use dark orange for `warn` so it stays readable on white-background terminals

### Refactor
- Rename `logging.py` to `log.py` to avoid shadowing the stdlib `logging` module

### Dependencies
- Bump `lattifai-captions` to >=0.4.11 (preserves explicit speaker prefix on consecutive VTT cues)


## [1.5.6] - 2026-04-18

### Features
- Add `honor_meta_chapters` hard constraint to summarize (preserves meta chapters verbatim) with tests

### Fixes
- Preserve input `Supervision.id` when `split_sentence=False` in alignment
- Update stale `caption.behavior` refs to `caption.render` in CLI

### Dependencies
- Bump `lattifai-core` to >=0.7.5, `lattifai-captions` to >=0.4.6


## [1.5.5] - 2026-04-13

### Features
- Chapter-based summaries with metadata and notable quotes
- Surface `kinetic_style` in caption convert CLI with auto-enable for word-level output
- Add brand-name and no-meta-commentary rules to translation prompts

### Fixes
- Propagate auth/quota errors from alignment API instead of wrapping as LatticeDecodingError
- Respect explicit `render.word_level=false` when karaoke mode is set
- Decouple `kinetic_style` from `word_level` auto-enable
- Use json-repair for robust LLM JSON parsing (llm module)
- Retry on LLM JSON failure in summarize, log raw output on parse error
- Use correct theme methods in summarize CLI (warn/err not warning/error)

### Refactor
- Drop obsolete karaoke→word_level shim, gate kinetic shim behind explicit flag

### Dependencies
- Bump lattifai-core to >=0.7.4, lattifai-captions to >=0.4.4
- Declare mlx/all/transcription/diarization extras as conflicting (protobuf version split)


## [1.5.2] - 2026-04-08

### Fixes
- Auto-inject `X-Device-Auth` header in `ClientConfig` for device-bound API keys
- Preserve TOML comments in `config.toml` during auth persist (tomlkit round-trip)
- `_persist_auth` / `_persist_trial_auth` no longer clear existing auth section

### Refactor
- Trim 23k lines of dead vendored `qwen_asr` code (CLI demos, vLLM backend, Korean dict)
- Remove `transformers>=5.5.0` and `torch` from `[transcription]` extra (disabled features)

### Tests
- Add auth tests: X-Device-Auth injection, comment preservation, trial persist
- Add auth/config tests to CI fast-fail stage


## [1.5.1] - 2026-04-08

### Fixes
- Fix 15 bugs from v1.5.0 third-party install test report:
  - Fix README SDK examples: `CaptionConfig(split_sentence=...)` → `CaptionConfig(input=CaptionInputConfig(...))`
  - Fix README CLI commands: `lai alignment youtube` → `lai youtube align`, `lai translate run` → `lai translate caption`
  - Fix `lai-summarize --help` crash (remove `from __future__ import annotations` for nemo_run compat)
  - Remove false `lhotse` dependency check from `lai doctor`
  - Add `mock_api_key` fixture for tests constructing `LattifAI()` without real API key
  - Add `@requires_nemo` skip marker for transcription tests needing `[transcription]` extra
  - Add `lai --version` via Typer callback
  - Expose `lattifai.__version__` on namespace package via `_init.py` injection
  - Fix LICENSE classifier mismatch (Apache → MIT)
  - Remove non-existent `[youtube]` from `[all]` extra
  - Fix `laicap-convert` help text typo
  - Fix auth callback `ValueError` on double `server_close()`
  - Pin `pytest-asyncio<2.0.0`
- Fix `qwen-asr` / `transformers` version conflict in `[transcription]` extra (remove `qwen-asr` pip dep, use vendored copy)
- Skip Gemini 503 errors in transcription save test instead of failing
- Correct Breaking Changes table in CHANGELOG and RELEASE_NOTES (v1.4.x flat `CaptionConfig`, not intermediate `OutputBehavior`)
- Fix `lai translate run` → `lai translate caption` in RELEASE_NOTES

### Refactor
- Format-specific CLI configs for caption convert (ASS, TTML, FCPXML, Premiere, LRC)
- Shared audio duration limits and VAD-aware chunking for transcription

### Dependencies
- `lattifai-captions` ≥ 0.4.2 (was ≥ 0.4.0)
- Remove `qwen-asr` from `[transcription]` extra (vendored)


## [1.5.0] - 2026-04-06

> This release contains **breaking changes** — see migration guide below. Config API overhaul, new auth system, 10 new CLI commands, and major feature additions across transcription, translation, diarization, and YouTube pipelines.

### Breaking Changes

**CaptionConfig restructured** — flat fields split into nested sub-configs:

| Before (v1.4.x) | After (v1.5.0) |
|------------------|----------------|
| `CaptionConfig(split_sentence=True)` | `CaptionConfig(input=CaptionInputConfig(split_sentence=True))` |
| `CaptionConfig(word_level=True)` | `CaptionConfig(render=RenderConfig(word_level=True))` |
| `CaptionConfig(karaoke=KaraokeConfig(enabled=True))` | `CaptionConfig(ass=ASSConfig(karaoke_effect="sweep"))` |
| `caption.write(path, word_level=..., karaoke_config=...)` | `caption.write(path, render=..., format_config=...)` |
| `caption.word_level=true` (CLI) | `render.word_level=true` (CLI) |
| `caption.karaoke.enabled=true` (CLI) | `ass.karaoke_effect=sweep` (CLI) |
| `AlignmentConfig(flush=...)` | `AlignmentConfig(flush_interval=...)` |

**Migration steps:**
1. `CaptionConfig(split_sentence=..., normalize_text=...)` → wrap with `input=CaptionInputConfig(...)`
2. `CaptionConfig(word_level=..., include_speaker_in_text=..., translation_first=...)` → wrap with `render=RenderConfig(...)`
3. `CaptionConfig(karaoke=KaraokeConfig(...))` → replace with `ass=ASSConfig(karaoke_effect="sweep")`
4. `caption.write(path, word_level=..., karaoke_config=...)` → `caption.write(path, render=..., format_config=...)`
5. `AlignmentConfig(flush=N)` → `AlignmentConfig(flush_interval=N)`

### Features

#### Authentication
- New CLI commands: `lai auth login`, `lai auth logout`, `lai auth whoami`
- `lai auth trial` — no-signup quick start with auto-provisioned trial key

#### Transcription
- **vLLM/SGLang backend**: Voxtral Realtime (WebSocket), Gemma-3n, Fun-ASR-Nano
- **Qwen3-ASR** local model support
- Batch concurrent transcription for VAD chunks
- Auto-inject ASR system prompt and temperature for general-purpose LLMs
- Dual API mode (transcriptions / chat), event detection, verbose_json support

#### Translation
- Full caption translation pipeline with multi-provider support (Gemini / OpenAI)
- Three modes: quick (direct), normal (analyze → translate), refined (analyze → translate → review → polish)
- Glossary management and bilingual output with `translation_first` display order
- Retry with exponential backoff and checkpoint/resume for long documents

#### Speaker Diarization
- LLM-based speaker name inference using YouTube metadata and dialogue context
- Candidate extraction with voting, talk format detection, post-LLM audience correction
- `lai diarize naming` — dedicated command for speaker identification

#### YouTube
- 4+ transcript extraction strategies (Dwarkesh, Substack, podscripts.co, Rescript API)
- Chrome CDP and Safari fallbacks for SPA transcript pages
- SSL hijack detection, auto-detect caption language from video metadata
- Parent channel metadata resolution and reordered speaker palette

#### CLI
- `lai doctor` — diagnostics with bundled selftest data
- `lai update` — automated updater with stale editable install detection
- `lai config set KEY=VALUE` — config.toml management
- `lai serve` — local web playground (4-tab: align, transcribe, convert, translate)
- `lai summarize caption` — caption summarization
- `lai youtube run` — top-level YouTube processing command
- `lai translate youtube` — YouTube-to-translation pipeline
- Broadcast standardization parameter for `lai caption convert`

#### Config System
- `config.toml` auto-resolution: CLI defaults sourced from `~/.config/lattifai/config.toml` via `_toml_section`
- Structured `CaptionConfig` with sub-configs: `CaptionInputConfig`, `CaptionOutputConfig`, `RenderConfig`
- `CaptionConfig.write_kwargs()` helper — centralizes `Caption.write()` argument construction
- `LRCConfig` support for LRC lyric format export
- Nested TOML sections for per-module LLM configs

#### Alignment
- Streaming flush-every-N-chunks for O(chunk) memory usage
- Duplicate text block detection — identifies repeated subtitle segments across flush boundaries with diff-aware detokenization
- RMS volume normalization before ONNX inference
- `start_margin` / `end_margin` unified defaults (0.10s)

#### ASS / Karaoke
- 12 karaoke color schemes (azure-gold, sakura-purple, mint-ocean, etc.)
- 10-color speaker palette with `ass.speaker_color=auto`
- `ASSConfig` self-contained: font, colors, outline, shadow, positioning, karaoke effect

#### LLM
- Unified `lattifai.llm` module: `BaseLLM`, `GeminiLLM`, `OpenAICompatLLM`
- Shared `LLMConfig` with reasoning toggle and section-based auto-resolution
- Gemini routed through OpenAI-compatible endpoint via shared client

### Fixes
- YouTube: Strategy 3b tx_start logic, strip inline UI noise, SSL hijack detection, unreachable host fallback
- Alignment: low word-level ratio warning, punctuation-inflated duplicate filter, multilingual tokenizer, label-aware boundary merge
- Transcription: numpy array wrapping for Qwen3-ASR, propagate detected language, verbose_json fallback, correct Supervision timing
- CLI: CJK support in `align_timestamps_from_ref`, wire config.toml into runtime, harden doctor/update, check API key in config.toml and .env
- Translation: handle dict response from LLM, replace tqdm.write with stderr, phase progress display
- Diarization: make llm field non-optional for CLI, section-specific config priority
- Media: honor `output_format` in `normalize_format`, fix video download
- Config: uppercase key names for config.toml lookup, load .env in LLMConfig

### Dependencies
- `lattifai-captions` ≥ 0.4.0 (breaking: RenderConfig API)
- `lattifai-core` ≥ 0.7.3
- `lattifai-run` ≥ 1.0.4
- `k2py` 0.4.0 (upgraded from 0.2.4)
- Replaced `cgi.FieldStorage` with stdlib-only multipart parser (Python 3.13 ready)


## [1.4.2] - 2026-02-26

### Features
- **Inline Transcription**: Core transcription logic inlined into lattifai-python, removing dependency on `lattifai_core.transcription`
- **New Gemini Models**: Added support for `gemini-3.1-pro-preview`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`

### Fixes
- Updated `lattifai-captions` dependency to v0.2.2

### Refactor
- Extracted event detector initialization to `_ensure_event_detector()` in mixin (shared lazy init)
- Removed dead code from transcription module (~45 lines): unused `GEM_URL`, `get_gem_info()`, `_build_result()`, dead `except ImportError` blocks
- Consolidated `_to_supervisions` from 4 code paths to `_make_sup` helper + 2 branches
- Replaced `hasattr` pattern with `getattr` in `_extract_response_metadata`
- Moved inline imports (`tempfile`, `soundfile`, `asyncio`) to module level


## [1.3.4] - 2026-02-04

### Added
- **Event Detection**: Automatic detection and timestamp alignment of non-speech audio events (`[MUSIC]`, `[APPLAUSE]`, etc.)
- **Streaming Confidence Scores**: Real-time confidence scoring with anomaly detection during alignment
- **Extended Caption Class**: New pipeline fields for transcription, events, diarization in Caption dataclass
- **Transcription Enhancements**:
  - Thinking mode with `include_thoughts` config for Gemini
  - Custom prompt support for transcription models
  - Temperature, top_k, top_p generation parameters
  - Citation metadata extraction from Gemini responses
- **Transition Penalty**: New `transition_penalty` parameter in AlignmentConfig for tuning alignment behavior

### Changed
- **Caption Module Migration**: Moved caption formats to `lattifai-captions` package (reduced core dependencies)
- **Config Centralization**: Migrated CaptionConfig from lattifai-captions to lattifai-python
- **Dependency Restructure**: Base install now includes alignment; optional extras for `transcription`, `youtube`, `diarization`, `event`

### Fixed
- GeminiTranscriber now properly supports Caption objects in `write()` method
- Use `importlib.metadata` for SDK version detection (more reliable)
- Suppressed CoreMLExecutionProvider warnings about partial graph support
- Sanity check and partial results support in Lattice1Aligner

## [1.2.2] - 2026-01-18

### Added
- Benchmark page enhancements with social sharing metadata and dedicated card

### Changed
- Refactored project structure to remove legacy backend server components

### Fixed
- Improved alignment error text extraction logic

## [1.2.1] - 2026-01-15

### Added
- Smart model caching mechanism with 7-day validity period
- Cache validation for Hugging Face and ModelScope model downloads
- Date-based cache markers (.doneYYYYMMDD) for tracking model freshness
- Centralized `_resolve_model_path()` function for unified model resolution

### Changed
- Improved ModelScope cache directory detection to match actual structure

### Fixed
- ModelScope cache path detection now correctly uses `~/.cache/modelscope/hub/models/` structure
- Automatic cleanup of old cache markers when downloading new models

## [1.2.0] - 2025-12-31

### Added
- k2py integration: Complete migration from k2 to k2py for improved performance
- Hugging Face model revision support: Specify model revisions and fetch latest commit SHA

### Changed
- **Reduced PyTorch dependency**: Switched to NumPy for emission processing
- Updated modelscope dependency to version 1.33.0
- Refactored acoustic model to process raw audio with ONNX runtime and return numpy emissions
- Relocated caption file processing logic to run after yt-dlp execution
- Removed k2 installation steps from test workflows

### Fixed
- Better error handling in caption file processing
- Enhanced no-caption return paths in YouTube downloads

## [1.1.0] - 2025-12-28

### Added
- Speaker diarization support
- Support for Gemini 3 Flash Preview model (gemini-3-flash-preview)
- ModelScope download support for alignment and transcription configurations
- `start_margin` and `end_margin` parameters to alignment configuration and tokenizer
- New CLI command: `lai-diarize` for speaker diarization workflows

### Changed
- Updated `nemo_toolkit_asr` dependency to version 2.7.0rc4
- Upgraded `lattifai-core` dependency to 0.6.0
- Added `scipy!=1.16.3` version constraint
- Changed default value of `include_speaker_in_text` to False in convert function

### Fixed
- Improved error handling in tokenizer loading functions

## [1.0.5] - 2025-12-14

### Changed
- Improved streaming transcription stability
- Enhanced error handling and logging

### Added
- Streaming support for long audio processing with configurable chunk size and overlap
- Progress tracking for audio chunk emission generation
- SBV format support for reading and writing captions
- Comprehensive test suite for all caption formats with coverage reports
- Safe print utility for better Unicode handling across the codebase

### Fixed
- Chromium installation method updated to use snap for Ubuntu 24.04 compatibility in CI
- Temporary file handling on Windows - ensure files are closed before writing and cleaned up afterwards
- Unicode handling in console output with new safe_print utility
- Audio data handling to remove tensor field and work directly with ndarray
- Correct ndarray shape access in emission computation
- Documentation URLs to include 'blob/main' for correct linking
- Standardized spelling of 'LattifAI' across the codebase

### Changed
- Enhanced AudioData class with streaming mode detection and chunk iteration support
- Updated Lattice1Worker with beam search parameters and streaming alignment capabilities
- Improved audio processing for very short audio files
- Modified AlignmentConfig with additional beam search configuration options
- Updated batch size in sentence splitting for LatticeTokenizer
- Refactored code structure for improved readability and maintainability
- Enhanced CI workflow for cross-platform compatibility
- Updated dependencies and enhanced transcribe function with client configuration
- Renamed workflow from 'test' to 'test_uv' for clarity
- Updated OmniSenseVoice dependency version to 0.4.2

## [1.0.4] - 2025-12-09

### Fixed
- Fixed YouTube download issues in GitHub Actions CI by installing Chromium browser
- Added browser environment setup for yt-dlp cookie extraction
- Improved error handling for YouTube downloads in CI environments
- Made YouTube tests non-blocking to handle platform restrictions gracefully

### Changed
- Upgraded yt-dlp installation to include all optional dependencies
- Added Chromium browser and chromedriver to CI test environment
- Improved test reliability with conditional execution based on download success

## [1.0.2] - 2025-12-09

### Fixed
- Fixed package-data entry for lattifai in pyproject.toml to correctly include app files
- Updated installation instructions to reference version 1.0.2
- Set fixed output format for AlignmentForm component in web application

### Changed
- Package version incremented to 1.0.2

## [1.0.0] - 2025-12-07

> ⚠️ **BREAKING CHANGE**: This major release introduces a completely refactored CLI architecture and updated API. Previous commands and scripts will need to be updated.

### Major Changes

#### CLI & Configuration Refactor
- **Unified Command Structure**: New subcommand-based CLI (e.g., `lai alignment align`, `lai caption convert`).
- **Positional Arguments**: Support for clean, intuitive commands without excessive flags.
- **Advanced Configuration**: Powered by `nemo_run`, enabling composable configs and type safety.

#### Core System Updates
- **New Caption Class**: A unified `Caption` class replaces `CaptionIO` for robust subtitle handling (SRT, VTT, ASS, TextGrid, JSON).
- **YouTube Support**: Native parsing of YouTube auto-generated captions with word-level timestamps.

#### Transcription & Diarization
- **Enhanced Integration**: Seamless transcription toggle and model selection.
- **Speaker Diarization**: Configurable diarization with results stored immediately in `Caption` objects.

#### Caption/Subtitle Tools
- **Conversion & Normalization**: Dedicated commands to convert formats and normalize subtitle text.
- **Multilingual Tokenization**: Improved text processing for Chinese, English, and German.

### Breaking Changes
- **Renamed Classes**: `Lattice1AlphaWorker` -> `Lattice1Worker`.
- **Removed**: `AsyncLattifAI` and `AsyncLatticeTokenizer` (use synchronous counterparts).
- **CLI Structure**: Old command styles are replaced by the new `lai <subcommand> <action>` syntax.

## [0.4.6] - 2025-11-05

### Added
- Python 3.14 support in package requirements and classifiers

### Improved
- Subtitle parsing now preserves original HTML entities (`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&#39;`, etc.)
- Better subtitle content integrity during the parsing phase
- Reduced unnecessary HTML entity decoding for cleaner subtitle output

### Changed
- Package version incremented to 0.4.6

## [0.4.5] - 2025-11-03

### Improved
- **Audio Loading Performance**: Refactored `load_audio` method to remove dependency on `lhotse.audio.read_audio`
  - Replaced `resampy` with `lhotse` built-in resampler for better performance
  - Enhanced audio resampling with chunked processing for long audio files (3600s chunks)
  - Improved memory efficiency with proper tensor cleanup
  - Better multi-channel audio handling with configurable channel selection
  - Removed `resampy` dependency from requirements

### Changed
- Package version incremented to 0.4.5

## [0.4.4] - 2025-11-03

### Improved
- **Alignment Success Rate**: Significantly increased beam search parameters for improved alignment accuracy
  - Increased `search_beam` from 50 to 200 (4x improvement)
  - Increased `output_beam` from 20 to 80 (4x improvement)
  - Enhanced decoding performance and reduced alignment failures
- **Speaker Name Parsing**: Fixed speaker text name parsing and restoration with correct regex patterns
  - Fixed `SPEAKER_PATTERN2` regex to properly match uppercase speaker names
  - Improved speaker name extraction for formats like "JOHN DOE: text"
  - Enhanced speaker text restoration in subtitle output

### Changed
- Package version incremented to 0.4.4

## [0.4.3] - 2025-11-02

### Fixed
- **CRITICAL**: Fixed `No module named 'lattifai.workflows.prompts'` error that prevented agentic workflows from running
- Added `lattifai.workflows.prompts` package to `pyproject.toml` configuration
- Included workflow prompt template files in package distribution
- Removed hardcoded cwd in CLI command tests for better portability
- Enhanced text integrity check in sentence splitting tests
- Improved error handling to preserve original exception context

### Added
- Short option flags for API keys: `-K, -L, --api-key` and `-G, --gemini-api-key`
- Support for Python 3.14 in version requirements

### Improved
- Environment variable handling with enhanced dotenv loading in CLI
- API key validation and logging in GeminiTranscriber with better error messages
- Package configuration by properly including workflow packages and data files
- Cache management for alignment models with better error preservation

### Changed
- Updated Python version requirements to `>=3.10,<3.14`
- Package version incremented to 0.4.3ifAI library adheres to [Semantic Versioning](http://semver.org/).
LattifAI-Py has a `major.minor.patch` version scheme. The major and minor version
numbers of LattifAI-Py are the major and minor version of the bound core library,
respectively. The patch version is incremented independently of the core
library.

We use [this changelog structure](http://keepachangelog.com/).

## Unreleased

None.

## [0.4.3] - 2025-11-02

### Added
- Short option flags for API keys: `-k, --api-key` and `-g, --gemini-api-key`
- Support for Python 3.14 in version requirements

### Improved
- Environment variable handling with enhanced dotenv loading in CLI
- API key validation and logging in GeminiTranscriber with better error messages
- Package configuration by updating package list in `pyproject.toml`
- Cache management for alignment models with better error preservation

### Changed
- Updated Python version requirements to `>=3.10,<3.14`
- Package version incremented to 0.4.3

### Fixed
- Removed hardcoded cwd in CLI command tests for better portability
- Enhanced text integrity check in sentence splitting tests
- Improved error handling to preserve original exception context

## [0.4.1] - 2025-11-02

### Fixed
- Text parsing issue where newline characters in subtitle text were not properly cleaned up
- Subtitle text now automatically replaces newline characters with spaces for improved readability
- Enhanced text parser robustness for handling edge cases and special characters

### Improved
- Python 3.9 compatibility with type hint adjustments
- README documentation for clarity on YouTube processing commands
- Project branding with logo asset

### Changed
- Bumped `lattifai-core` dependency to `>=0.2.1` for better stability
- Removed outdated demo file (`demos/demo_agentic_workflow.py`)

## [0.4.0] - 2025-11-01

### Added
- **Shorthand CLI Command**: Introduced `lai` as a convenient alias for `lattifai` command
- **Agentic Workflows**: Intelligent autonomous pipelines for end-to-end YouTube subtitle generation
  - `lai agent --youtube` command for automated YouTube processing
  - Integration with Google Gemini 2.5 Pro for AI-powered transcription
  - YouTubeSubtitleAgent for programmatic workflow automation
  - Support for both audio (MP3, WAV, M4A, AAC, OPUS) and video (MP4, WebM, MKV) formats
  - Smart file management with user confirmation prompts
  - Retry mechanism for robust processing
- Python API for agentic workflows with `YouTubeSubtitleAgent`, `YouTubeDownloader`, and `GeminiTranscriber`
- Updated README documentation to promote `lai` as the recommended command for daily use

### Improved
- CLI user experience with faster command typing
- Documentation with `lai` command examples throughout
- End-to-end workflow automation from YouTube URL to aligned subtitles

### Dependencies
- Added `yt-dlp` for YouTube video/audio downloading
- Added `google-genai` for Google Gemini API integration
- Added `questionary` for interactive CLI prompts
- Added `pycryptodome` for secure download handling

## [0.2.5] - 2025-10-26

### Added
- Comprehensive error handling system with clear error messages and support links
- Error hierarchy with specific exception classes for different error types
- Context information in error messages for better debugging

## [0.2.4] - 2025-10-19

### Added
- Format support for: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, AIFF (audio) and MP4, MKV, MOV, WEBM, AVI (video)

### Improved
- Audio format compatibility with expanded codec support


## [0.2.2] - 2025-10-19

### Added
- **Smart Sentence Splitting Feature**: New `--split_sentence` option for intelligent sentence re-splitting based on punctuation and semantic boundaries
- `_resplit_special_sentence_types()` method for detecting and handling special sentence patterns
- Support for event markers (`[APPLAUSE]`, `[MUSIC]`, etc.), HTML-encoded separators (`&gt;&gt;`), and natural punctuation boundaries
- Comprehensive documentation for the new `--split_sentence` feature in README
- Test suite for special sentence type re-splitting (`test_special_sentence_resplit.py`)

### Improved
- Alignment accuracy for complex subtitle formats with mixed content types
- Subtitle processing pipeline with semantic boundary awareness
- Documentation clarity and English language consistency throughout codebase

### Changed
- All code comments and documentation converted to English for better international accessibility

## [0.2.0] - 2025-10-08

### Added
- First public release.
