# CHANGELOG

## Versioning

The LattifAI library adheres to [Semantic Versioning](http://semver.org/).
LattifAI-Py has a `major.minor.patch` version scheme. The major and minor version
numbers of LattifAI-Py are the major and minor version of the bound core library,
respectively. The patch version is incremented independently of the core
library.

We use [this changelog structure](http://keepachangelog.com/).

## Unreleased

None.

## [0.2.8] - 2025-11-01

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

