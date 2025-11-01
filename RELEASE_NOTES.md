# Release Notes - LattifAI Python v0.4.0

**Release Date:** November 1, 2025

---

## 🎉 Overview

LattifAI Python v0.4.0 introduces two major enhancements:

1. **`lai` Command**: A convenient shorthand alias for `lattifai`, making the CLI faster and easier to use in your daily workflow.

2. **Agentic Workflows**: Intelligent, autonomous pipelines for end-to-end YouTube subtitle generation. Process videos from URL to aligned subtitles with a single command, powered by Google Gemini 2.5 Pro transcription and LattifAI's Lattice-1-Alpha alignment.

---

## ✨ New Features

### 1. Shorthand CLI Command: `lai`

This release adds `lai` as a convenient shorthand command that provides identical functionality to `lattifai`.

#### Key Benefits:

- **Faster Typing**: Save keystrokes with a shorter command name
- **Improved Productivity**: Streamlined workflow for frequent CLI users
- **Identical Functionality**: All features and options work exactly the same
- **Full Compatibility**: Use `lai` and `lattifai` interchangeably

### 2. Agentic Workflows for YouTube Processing

LattifAI now includes intelligent agentic workflows that automate end-to-end subtitle generation from YouTube videos.

#### What are Agentic Workflows?

Agentic workflows are intelligent, autonomous pipelines that handle complex multi-step tasks through automated agents. The YouTube workflow combines:
1. **Automatic video/audio download** from YouTube URLs
2. **AI-powered transcription** using Google's Gemini 2.5 Pro
3. **Precise alignment** with LattifAI's Lattice-1-Alpha model

#### Key Features:

- **One-Command Processing**: From YouTube URL to aligned subtitles
- **Gemini Integration**: State-of-the-art transcription with Gemini 2.5 Pro (Thinking mode)
- **Flexible Media Formats**: Download as audio (MP3, WAV, M4A, AAC, OPUS) or video (MP4, WebM, MKV)
- **Smart File Management**: Automatic handling of existing files with user confirmation
- **Retry Mechanism**: Configurable retry logic for robust processing
- **Multiple Output Formats**: SRT, VTT, ASS, TXT, TextGrid, JSON, and more

#### Usage - `lai` Command:

**All existing commands work with `lai`:**

```bash
# Align audio with subtitle
lai align audio.wav subtitle.srt output.srt

# Convert subtitle format
lai subtitle convert input.srt output.vtt

# Use advanced options
lai align --split_sentence audio.wav subtitle.srt output.json

# GPU acceleration
lai align --device cuda audio.wav subtitle.srt output.srt
```

#### Usage - Agentic Workflow:

**Basic YouTube Processing:**

```bash
# Process YouTube video with default settings
lai agent --youtube "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify output format and directory
lai agent --youtube "https://youtu.be/VIDEO_ID" \
  --output-format srt \
  --output-dir ./subtitles

# Download as audio format with semantic splitting
lai agent --youtube "URL" \
  --media-format mp3 \
  --split-sentence \
  --output-format json
```

**Advanced Options:**

```bash
# Word-level alignment with custom Gemini API key
lai agent --youtube "URL" \
  --word-level \
  --gemini-api-key YOUR_API_KEY \
  --media-format mp4 \
  --output-format TextGrid

# Video format with retry mechanism
lai agent --youtube "URL" \
  --media-format webm \
  --max-retries 3 \
  --force \
  --verbose
```

**Required Setup for Agentic Workflows:**

```bash
# Set Gemini API key (get it from https://ai.google.dev)
export GEMINI_API_KEY="your-gemini-api-key"

# Or provide it via command line
lai agent --youtube "URL" --gemini-api-key "your-key"
```

**Equivalent to:**

```bash
lattifai align audio.wav subtitle.srt output.srt
lattifai subtitle convert input.srt output.vtt
lattifai align --split_sentence --word_level audio.wav subtitle.srt output.json
lattifai align --device cuda audio.wav subtitle.srt output.srt
```

#### Help and Documentation:

```bash
# View help with either command
lai --help
lai align --help

# Both commands provide identical output
lattifai --help
lattifai align --help
```

---

## 🔧 Technical Details

### CLI Command Implementation

- Added `lai` entry point in `pyproject.toml` pointing to the same CLI function
- Zero performance difference between `lai` and `lattifai` commands
- Both commands share the same codebase and configuration

### Command Structure

```toml
[project.scripts]
lattifai = 'lattifai.bin:cli'
lai = 'lattifai.bin:cli'
```

### Agentic Workflow Architecture

The agentic workflow system is built on a modular architecture:

#### Core Components:

1. **WorkflowAgent**: Base class for all workflow agents
2. **YouTubeSubtitleAgent**: Specialized agent for YouTube processing
3. **YouTubeDownloader**: Handles video/audio download using yt-dlp
4. **GeminiTranscriber**: Integrates Google Gemini 2.5 Pro for transcription
5. **FileExistenceManager**: Smart file management with user prompts
6. **AsyncLattifAI**: Async client for subtitle alignment

#### Workflow Pipeline:

```
YouTube URL → Download Media → Gemini Transcription → LattifAI Alignment → Export Subtitles
```

Each step includes:
- **Error handling** with automatic retry logic
- **Progress reporting** with colorful CLI output
- **State management** for workflow continuity
- **File validation** and existence checks

#### Dependencies:

Agentic workflows require additional packages:
- `yt-dlp`: YouTube video/audio downloading
- `google-genai`: Google Gemini API client
- `questionary`: Interactive CLI prompts
- `pycryptodome`: Secure download handling

These are automatically installed with the base `lattifai` package.

---

## 📚 Documentation Updates

- **README.md**: Updated all CLI examples to showcase `lai` as the recommended command
- Added helpful tip encouraging users to use `lai` for daily workflow
- Maintained `lattifai` documentation for reference and backward compatibility
- Added comprehensive agentic workflow documentation with usage examples
- Included Gemini API setup instructions

## 🎯 Use Cases

### Traditional Workflow (Manual Steps):
```bash
# 1. Download YouTube video manually
# 2. Extract audio if needed
# 3. Get transcript (manual or ASR)
# 4. Align with LattifAI
lai align audio.wav transcript.txt output.srt
```

### Agentic Workflow (Automated):
```bash
# Single command does everything!
lai agent --youtube "https://youtube.com/watch?v=VIDEO_ID"
```

### When to Use Agentic Workflows:

✅ **Perfect for:**
- YouTube videos without existing subtitles
- Quick subtitle generation from video URLs
- Batch processing multiple YouTube videos
- Content creation and video editing workflows
- Podcast and educational content processing

✅ **Benefits:**
- No manual download or transcription needed
- State-of-the-art Gemini 2.5 Pro transcription
- Precise alignment with Lattice-1-Alpha
- Automatic file management and organization

### When to Use Traditional Alignment:

✅ **Perfect for:**
- Local audio/video files
- Existing transcripts that need alignment
- Custom transcription pipelines
- Fine-tuned control over each step

---

## � Python API for Agentic Workflows

The agentic workflow system is also available as a Python API:

```python
import asyncio
from lattifai.workflows import YouTubeSubtitleAgent

async def process_youtube_video():
    # Initialize agent
    agent = YouTubeSubtitleAgent(
        gemini_api_key="your-gemini-api-key",
        video_format="mp4",
        output_format="srt",
        split_sentence=True,
        word_level=True,
        max_retries=3
    )

    # Process YouTube URL
    result = await agent.process_youtube_url(
        url="https://www.youtube.com/watch?v=VIDEO_ID",
        output_dir="./output"
    )

    # Access results
    print(f"Title: {result['metadata']['title']}")
    print(f"Subtitle count: {result['subtitle_count']}")
    print(f"Files: {result['exported_files']}")

# Run the agent
asyncio.run(process_youtube_video())
```

**Advanced Usage with Individual Components:**

```python
from lattifai.workflows import YouTubeSubtitleAgent
from lattifai.workflows.youtube import YouTubeDownloader
from lattifai.workflows.gemini import GeminiTranscriber

# Use individual components for custom workflows
downloader = YouTubeDownloader(media_format='mp3')
transcriber = GeminiTranscriber(api_key='your-key')

# Custom workflow
async def custom_workflow(url: str):
    # Download audio
    audio_path = await downloader.download_audio(url, output_dir='./downloads')

    # Transcribe with Gemini
    transcript = await transcriber.transcribe_audio(audio_path)

    # Align with LattifAI
    from lattifai import AsyncLattifAI
    async with AsyncLattifAI() as client:
        alignments, output_path = await client.alignment(
            audio=audio_path,
            subtitle=transcript,
            split_sentence=True,
            output_subtitle_path="output.srt"
        )

    return alignments
```

---

## �📦 Installation & Upgrade

### Upgrade from Previous Versions:

```bash
pip install --upgrade lattifai
```

After upgrading, the `lai` command will be immediately available:

```bash
lai --version
```

---

## 🔄 Backward Compatibility

✅ **100% Backward Compatible**
- All existing `lattifai` commands continue to work unchanged
- No breaking changes to API or CLI
- Existing scripts and documentation require no modifications
- Users can adopt `lai` at their own pace

---

## 💡 Migration Guide

**No migration required!** Both commands coexist:

```bash
# These are identical
lai align audio.wav subtitle.srt output.srt
lattifai align audio.wav subtitle.srt output.srt

# Mix and match as you prefer
lai align audio1.wav sub1.srt out1.srt
lattifai align audio2.wav sub2.srt out2.srt
```

**Recommendation**: Start using `lai` for new workflows while existing scripts continue to work with `lattifai`.

---

## 📚 Documentation

For complete documentation, visit:
- **Official Website**: https://lattifai.com
- **GitHub Repository**: https://github.com/lattifai/lattifai-python
- **README**: Updated with `lai` examples

---

## 📝 Version Info

- **Version**: 0.4.0
- **Release Date**: November 1, 2025
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

## v0.2.5 - Enhanced Error Handling

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
