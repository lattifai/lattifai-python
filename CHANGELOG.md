# CHANGELOG


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
