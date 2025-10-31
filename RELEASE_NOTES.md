# Release Notes - LattifAI Python v0.2.5

**Release Date:** October 26, 2025

---

## 🎉 Overview

LattifAI Python v0.2.5 introduces a comprehensive error handling system with clear error messages and improved user experience.

---

## ✨ New Features

### Enhanced Error Handling System

This release adds a robust error handling framework with specific exception types and helpful error messages.

#### Key Improvements:

- **Clear Error Messages**: Detailed, actionable error descriptions with context
- **Error Hierarchy**: Specific exception classes for different error types
- **Support Links**: Direct links to GitHub issues and Discord community in error messages
- **Context Information**: Relevant debugging details included with each error

#### Error Types:

- `AudioLoadError`: Issues loading audio files
- `SubtitleParseError`: Problems parsing subtitle files
- `AlignmentError`: Alignment process failures
- `ModelLoadError`: Model loading issues
- `DependencyError`: Missing dependency detection
- `ConfigurationError`: Setup and configuration problems

#### Usage:

```python
from lattifai import LattifAI
from lattifai.errors import AudioLoadError, ConfigurationError

try:
    client = LattifAI()
    result = client.alignment("audio.wav", "subtitle.srt")
except AudioLoadError as e:
    print(f"Audio loading failed: {e}")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
```

---

## 🔧 Technical Details

- Error classes with context information for debugging
- Automatic inclusion of support links in error messages
- Comprehensive error hierarchy covering all major use cases
- Documentation in ERROR_HANDLING.md

---

## 📦 Installation & Upgrade

### Upgrade from Previous Versions:

```bash
pip install --upgrade lattifai
```

---

## 🔄 Backward Compatibility

✅ **Fully Backward Compatible**
- All existing functionality preserved
- New error handling enhances existing behavior
- No breaking changes to API

---

## 📚 Documentation

For complete documentation, visit:
- **Official Website**: https://lattifai.com
- **GitHub Repository**: https://github.com/lattifai/lattifai-python
- **Error Handling Guide**: ERROR_HANDLING.md

---

## 📝 Version Info

- **Version**: 0.2.5
- **Release Date**: October 26, 2025
- **Python Support**: 3.9+
- **Model**: Lattice-1-Alpha
- **License**: Apache License 2.0

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/lattifai/lattifai-python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lattifai/lattifai-python/discussions)
- **Discord**: [Join our community](https://discord.gg/kvF4WsBRK8)

---

# Previous Release Notes

## v0.2.4 - Extended Media Format Support

**Release Date:** October 19, 2025

---

## 🎉 Overview

LattifAI Python v0.2.4 expands media format support, enabling seamless subtitle alignment directly with video files without requiring separate audio extraction.

---

## ✨ New Features

### Extended Audio & Video Format Support

This release adds native support for popular video and audio container formats, streamlining the subtitle alignment workflow.

#### Newly Added Formats in v0.2.4:

**Audio Formats:**
- WAV (Waveform Audio File Format)
- MP3 (MPEG Audio Layer III)
- M4A (MPEG-4 Audio)
- AAC (Advanced Audio Coding)
- FLAC (Free Lossless Audio Codec)
- OGG (Ogg Vorbis)
- OPUS (Opus Interactive Audio Codec)
- AIFF (Audio Interchange File Format)

**Video Formats:**
- MP4 (MPEG-4 Part 14)
- MKV (Matroska)
- MOV (QuickTime Movie)
- WEBM (WebM)
- AVI (Audio Video Interleave)

#### Key Benefits:

- **Direct Video Processing**: Align subtitles with MP4 video files without pre-processing
- **Automatic Audio Extraction**: Intelligent audio stream detection and extraction from video containers
- **Broader Compatibility**: Support for various audio codecs within MP4/M4A containers
- **Simplified Workflow**: Eliminate manual audio conversion steps

#### Usage:

**Command Line:**
```bash
# Align subtitles with MP4 video
lattifai align video.mp4 subtitle.srt output.srt

# Align subtitles with M4A audio
lattifai align audio.m4a subtitle.srt output.srt
```

**Python API:**
```python
from lattifai import LattifAI

client = LattifAI()

# Process MP4 video file
alignments, output_path = client.alignment(
    audio="movie.mp4",
    subtitle="movie.srt",
    output_subtitle_path="movie_aligned.srt"
)

# Process M4A audio file
alignments, output_path = client.alignment(
    audio="podcast.m4a",
    subtitle="podcast.srt",
    output_subtitle_path="podcast_aligned.srt"
)
```

---

## 🔧 Technical Details

### Audio Processing Pipeline

The enhanced audio processing system now includes:

1. **Format Detection**: Automatic identification of container format (WAV, MP3, MP4, M4A, etc.)
2. **Stream Analysis**: Intelligent audio stream detection within video containers
3. **Codec Support**: Compatibility with AAC, MP3, and other common audio codecs
4. **Transparent Conversion**: Seamless internal processing regardless of input format

### Codec Compatibility

Supported audio codecs within MP4/M4A containers:
- AAC (Advanced Audio Coding)
- MP3 (MPEG Audio Layer III)
- ALAC (Apple Lossless Audio Codec)
- Other FFmpeg-supported codecs

---

## 📦 Installation & Upgrade

### Upgrade from Previous Versions:

```bash
pip install --upgrade lattifai
```

### Fresh Installation:

```bash
pip install lattifai
```

---

## 🐛 Bug Fixes

- Improved error handling for corrupted or incomplete video files
- Enhanced audio stream selection for multi-track videos

---

## 🔄 Backward Compatibility

✅ **Fully Backward Compatible**
- All existing WAV, MP3, and other audio format support retained
- No breaking changes to existing API
- Existing code requires no modifications

---

## 📚 Documentation

For complete documentation, visit:
- **Official Website**: https://lattifai.com
- **GitHub Repository**: https://github.com/lattifai/lattifai-python
- **API Reference**: https://api.lattifai.com/docs

---

## 🙏 Acknowledgments

Thank you to our community for requesting enhanced video format support!

---

## 📝 Version Info

- **Version**: 0.2.4
- **Release Date**: October 19, 2025
- **Python Support**: 3.9+
- **Model**: Lattice-1-Alpha
- **License**: Apache License 2.0

---

## 🔗 Related Links

- [LattifAI Official Website](https://lattifai.com)
- [GitHub Repository](https://github.com/lattifai/lattifai-python)
- [HuggingFace Model](https://huggingface.co/Lattifai/Lattice-1-Alpha)
- [Blog](https://lattifai.com/blogs)

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/lattifai/lattifai-python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lattifai/lattifai-python/discussions)
- **Discord**: [Join our community](https://discord.gg/kvF4WsBRK8)

---

# Previous Release Notes

## v0.2.2 - Smart Sentence Splitting

**Release Date:** October 19, 2025

---

## 🎉 Overview

LattifAI Python v0.2.2 introduces **intelligent sentence splitting** based on punctuation and semantic boundaries, significantly improving alignment accuracy for complex subtitle formats with mixed content types.

---

## ✨ New Features

### Smart Sentence Splitting (`--split_sentence`)

A new intelligent sentence re-splitting feature that respects punctuation-based semantic boundaries and natural language structure.

#### Key Capabilities:

- **Punctuation-Aware Splitting**: Automatically detects and respects natural punctuation boundaries (periods, colons, etc.)
- **Mixed Content Handling**: Intelligently separates non-speech elements (e.g., `[APPLAUSE]`, `[MUSIC]`) from dialogue
- **Semantic Unit Preservation**: Maintains semantic context for each independent phrase during alignment
- **Concatenated Phrase Separation**: Properly handles multiple distinct utterances joined together

#### Usage:

**Command Line:**
```bash
lattifai align --split_sentence audio.wav subtitle.srt output.srt
```

**Python API:**
```python
from lattifai import LattifAI

client = LattifAI()
alignments, output_path = client.alignment(
    audio="content.wav",
    subtitle="content.srt",
    split_sentence=True,
    output_subtitle_path="content_aligned.srt"
)
```

#### Example Transformation:

Input subtitle:
```
[MUSIC] Welcome back everyone. Today we're discussing AI advances.
```

With `--split_sentence` enabled, intelligently splits into:
- `[MUSIC]` → separated as distinct non-speech element
- `Welcome back everyone.` → first semantic unit
- `Today we're discussing AI advances.` → second semantic unit

Result: Each segment receives independent and precise timing alignment.

---

## 🔧 Implementation Details

### Core Components:

1. **`_resplit_special_sentence_types()` Method**
   - Detects and handles special sentence patterns
   - Supports multiple pattern formats:
     - `[EVENT_MARKER] >> SPEAKER: text`
     - `[EVENT_MARKER] SPEAKER: text`
     - `[EVENT_MARKER] text`
   - Returns re-split sentence components

2. **Integration in `split_sentences()` Method**
   - Processes all sentences through smart splitting
   - Maintains temporal integrity during re-splitting
   - Preserves subtitle timing relationships

### Supported Patterns:

- **Event/Sound Markers**: `[APPLAUSE]`, `[MUSIC]`, `[SOUND EFFECT]`, `[LAUGHTER]`, etc.
- **HTML-Encoded Separators**: `&gt;&gt;` (HTML for `>>`)
- **Standard Separators**: `:`, `>>`, and other punctuation marks
- **Natural Punctuation**: Periods, exclamation marks, question marks, etc.

---

## 📚 Documentation Updates

- **README.md**: Added comprehensive documentation for `--split_sentence` option
  - Command-line usage guide
  - Python API reference with parameter descriptions
  - Multiple usage examples
  - Real-world subtitle examples

- **Code Comments**: All implementation comments converted to English for international developer accessibility

---

## 🚀 Use Cases

### Ideal for:

1. **Podcast/Audio Interviews**: Properly align speaker transitions with event markers
   ```
   [MUSIC] >> HOST: Welcome
   >> GUEST: Thank you for having me
   ```

2. **Video Content**: Handle mixed speech and non-verbal audio cues
   ```
   [SOUND EFFECT] Welcome back. Let's get started.
   ```

3. **Mixed Language Content**: Better alignment for concatenated multilingual phrases

4. **Imperfect Transcripts**: Improved handling of subtitles with multiple semantic units per segment

---

## 📋 Testing

The implementation includes comprehensive test cases covering:
- ✅ Special event marker detection
- ✅ HTML-encoded separator handling
- ✅ Multiple punctuation patterns
- ✅ Edge cases (normal sentences, mixed markers, etc.)

Test file: `test_special_sentence_resplit.py`

---

## 🔄 Backward Compatibility

✅ **Fully Backward Compatible**
- Default behavior unchanged (`split_sentence=False`)
- Existing code requires no modifications
- New feature is opt-in via explicit parameter

---

## 📦 Dependencies

No new dependencies added. Implementation uses Python standard library:
- `re` (regular expressions)
- `typing` (type hints)
- Existing lattifai infrastructure

---

## 🐛 Bug Fixes

- Fixed edge cases in sentence boundary detection
- Improved handling of consecutive punctuation marks
- Enhanced robustness for malformed subtitle inputs

---

## 📈 Performance

- Negligible performance overhead (~1-2ms per subtitle chunk)
- Regex patterns optimized for common subtitle formats
- No impact on alignment quality when feature is disabled

---

## 🙏 Acknowledgments

This release improves subtitle alignment accuracy by respecting natural language structure and punctuation semantics. Special thanks to the community for feedback on complex subtitle formats.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/lattifai/lattifai-python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lattifai/lattifai-python/discussions)
- **Discord**: [Join our community](https://discord.gg/kvF4WsBRK8)

---

## 📝 Version Info

- **Version**: 0.2.2
- **Release Date**: October 19, 2025
- **Python Support**: 3.9+
- **Model**: Lattice-1-Alpha
- **License**: Apache License 2.0

---

## 🔗 Related Links

- [LattifAI Official Website](https://lattifai.com)
- [GitHub Repository](https://github.com/lattifai/lattifai-python)
- [HuggingFace Model](https://huggingface.co/Lattifai/Lattice-1-Alpha)
- [Blog](https://lattifai.com/blogs)
