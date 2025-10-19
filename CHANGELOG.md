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

