# LattifAI Python

[![PyPI version](https://badge.fury.io/py/lattifai.svg)](https://badge.fury.io/py/lattifai)

<p align="center">
   🌐 <a href="https://lattifai.com"><b>Official Website</b></a> &nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://github.com/lattifai/lattifai-python">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/Lattifai/Lattice-1-Alpha">Model</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://lattifai.com/blogs">Blog</a> &nbsp&nbsp | &nbsp&nbsp <a href="https://discord.gg/kvF4WsBRK8"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" style="vertical-align: middle;"></a>
</p>

Advanced forced alignment and subtitle generation powered by [Lattice-1-Alpha](https://huggingface.co/Lattifai/Lattice-1-Alpha) model.

## Installation

```bash
pip install "install-k2>=0.0.6"
# The installation will automatically detect and use your already installed PyTorch version(up to 2.8).
install-k2  # Install k2

pip install lattifai
```

> **⚠️ Important**: You must run `install-k2` before using the lattifai library.

## Quick Start

### Command Line

The library provides two equivalent commands: `lai` (recommended for convenience) and `lattifai`.

```bash
# Align audio with subtitle (using lai command)
lai align audio.wav subtitle.srt output.srt

# Or use the full command
lattifai align audio.wav subtitle.srt output.srt

# Process YouTube videos with intelligent workflow
lai agent --youtube https://www.youtube.com/watch?v=VIDEO_ID

# Download and align YouTube content directly
lai youtube https://www.youtube.com/watch?v=VIDEO_ID

# Convert subtitle format
lai subtitle convert input.srt output.vtt
```

> **💡 Tip**: Use `lai` for faster typing in your daily workflow!

#### Command Quick Reference

| Command | Use Case | Best For |
|---------|----------|----------|
| `lai align` | Align existing audio + subtitle files | Local files, custom workflows |
| `lai youtube` | Download & align YouTube content | Quick one-off YouTube processing |
| `lai agent` | Intelligent YouTube workflow with retries | Production, batch jobs, automation |
| `lai subtitle` | Convert subtitle formats | Format conversion only |

#### lai align options
```
> lai align --help
Usage: lattifai align [OPTIONS] INPUT_AUDIO_PATH INPUT_SUBTITLE_PATH OUTPUT_SUBTITLE_PATH

  Command used to align audio with subtitles

Options:
  -F, --subtitle_format [srt|vtt|ass|ssa|sub|sbv|txt|auto|gemini]  Input subtitle format.
  -S, --split_sentence                                              Re-segment subtitles by semantics.
  -W, --word_level                                                  Include word-level alignment timestamps.
  -D, --device [cpu|cuda|mps]                                       Device to use for inference.
  -M, --model_name_or_path TEXT                                     Model name or path for alignment.
  --api_key TEXT                                                    API key for LattifAI.
  --help                                                            Show this message and exit.
```

#### lai youtube command

Download and align YouTube videos in one step. Automatically downloads media, fetches subtitles (or uses Gemini transcription if unavailable), and performs forced alignment.

```bash
# Basic usage
lai youtube https://www.youtube.com/watch?v=VIDEO_ID

# Common options: audio format, sentence splitting, word-level, GPU
lai youtube --media-format mp3 --split-sentence --word-level --device mps \
  --output-dir ./output --output-format srt https://www.youtube.com/watch?v=VIDEO_ID

# Use Gemini for transcription fallback
lai youtube --gemini-api-key YOUR_KEY https://www.youtube.com/watch?v=VIDEO_ID
```

**Options**:
```
> lai youtube --help
Usage: lattifai youtube [OPTIONS] YT_URL

  Download media and subtitles from YouTube for further alignment.

Options:
  -M, --media-format [mp3|wav|m4a|aac|flac|ogg|opus|aiff|mp4|webm|mkv|avi|mov]  Media format for YouTube download.
  -S, --split-sentence                                                           Re-segment subtitles by semantics.
  -W, --word-level                                                               Include word-level alignment timestamps.
  -O, --output-dir PATH                                                          Output directory (default: current directory).
  -D, --device [cpu|cuda|mps]                                                    Device to use for inference.
  -M, --model-name-or-path TEXT                                                  Model name or path for alignment.
  --api-key TEXT                                                                 API key for LattifAI.
  --gemini-api-key TEXT                                                          Gemini API key for transcription fallback.
  -F, --output-format [srt|vtt|ass|ssa|sub|sbv|txt|json|TextGrid]              Subtitle output format.
  --help                                                                         Show this message and exit.
```

#### lai agent command

**Intelligent Agentic Workflow** - Process YouTube videos through an advanced multi-step workflow with automatic retries, smart file management, and comprehensive error handling.

```bash
# Basic usage
lai agent --youtube https://www.youtube.com/watch?v=VIDEO_ID

# Production workflow with retries, verbose logging, and force overwrite
lai agent --youtube --media-format mp4 --output-format TextGrid \
  --split-sentence --word-level --device mps --max-retries 2 --verbose --force \
  --output-dir ./outputs https://www.youtube.com/watch?v=VIDEO_ID
```

**Key Features**:
- **🔄 Automatic Retry Logic**: Configurable retry mechanism for failed steps
- **📁 Smart File Management**: Detects existing files and prompts for action
- **🎯 Intelligent Workflow**: Multi-step pipeline with dependency management
- **🛡️ Error Recovery**: Graceful handling of failures with detailed logging
- **📊 Rich Output**: Comprehensive results with metadata and file paths
- **⚡ Async Processing**: Efficient parallel execution of independent tasks

**Options**:
```
> lai agent --help
Usage: lattifai agent [OPTIONS] URL

  LattifAI Agentic Workflow Agent

  Process multimedia content through intelligent agent-based pipelines.

Options:
  --youtube, --yt                                          Process YouTube URL through agentic workflow.
  --gemini-api-key TEXT                                    Gemini API key for transcription.
  --media-format [mp3|wav|m4a|aac|opus|mp4|webm|mkv|...]  Media format for YouTube download.
  --output-format [srt|vtt|ass|ssa|sub|sbv|txt|json|...]  Subtitle output format.
  --output-dir PATH                                        Output directory (default: current directory).
  --max-retries INTEGER                                    Maximum retries for failed steps.
  -S, --split-sentence                                     Re-segment subtitles by semantics.
  --word-level                                             Include word-level alignment timestamps.
  --verbose, -v                                            Enable verbose logging.
  --force, -f                                              Force overwrite without confirmation.
  --help                                                   Show this message and exit.
```

**When to use `lai agent` vs `lai youtube`**:
- **Use `lai agent`**: For production workflows, batch processing, advanced error handling, and when you need retry logic
- **Use `lai youtube`**: For quick one-off downloads and alignment with minimal overhead

#### Understanding --split_sentence

The `--split_sentence` option performs intelligent sentence re-splitting based on punctuation and semantic boundaries. This is especially useful when processing subtitles that combine multiple semantic units in a single segment, such as:

- **Mixed content**: Non-speech elements (e.g., `[APPLAUSE]`, `[MUSIC]`) followed by actual dialogue
- **Natural punctuation boundaries**: Colons, periods, and other punctuation marks that indicate semantic breaks
- **Concatenated phrases**: Multiple distinct utterances joined together without proper separation

**Example transformations**:
```
Input:  "[APPLAUSE] >> MIRA MURATI: Thank you all"
Output: ["[APPLAUSE]", ">> MIRA MURATI: Thank you all"]

Input:  "[MUSIC] Welcome back. Today we discuss AI."
Output: ["[MUSIC]", "Welcome back.", "Today we discuss AI."]
```

This feature helps improve alignment accuracy by:
1. Respecting punctuation-based semantic boundaries
2. Separating distinct utterances for more precise timing
3. Maintaining semantic context for each independent phrase

**Usage**:
```bash
lai align --split_sentence audio.wav subtitle.srt output.srt
```

#### Understanding --word_level

The `--word_level` option enables word-level alignment, providing precise timing information for each individual word in the audio. When enabled, the output includes detailed word boundaries within each subtitle segment, allowing for fine-grained synchronization and analysis.

**Key features**:
- **Individual word timestamps**: Each word gets its own start and end time
- **Format-specific output**:
  - **JSON (Recommended)**: Full alignment details stored in `alignment.word` field of each segment, preserving all word-level timing information in a structured format
  - **TextGrid**: Separate "words" tier alongside the "utterances" tier for linguistic analysis
  - **TXT**: Each word on a separate line with timestamp range: `[start-end] word`
  - **Standard subtitle formats** (SRT, VTT, ASS, etc.): Each word becomes a separate subtitle event

> **💡 Recommended**: Use JSON format (`output.json`) to preserve complete word-level alignment data. Other formats may lose some structural information.

**Example output formats**:

**JSON format** (with word-level details):
```json
[
{
  "id": "6",
  "recording_id": "",
  "start": 24.52,
  "duration": 9.1,
  "channel": 0,
  "text": "We will start with why it is so important to us to have a product that we can make truly available and broadly available to everyone.",
  "custom": {
    "score": 0.8754
  },
  "alignment": {
    "word": [
      [
        "We",
        24.6,
        0.14,
        1.0
      ],
      [
        "will",
        24.74,
        0.14,
        1.0
      ],
      [
        "start",
        24.88,
        0.46,
        0.771
      ],
      [
        "with",
        25.34,
        0.28,
        0.9538
      ],
      [
        "why",
        26.2,
        0.36,
        1.0
      ],
      [
        "it",
        26.56,
        0.14,
        0.9726
      ],
      [
        "is",
        26.74,
        0.02,
        0.6245
      ],
      [
        "so",
        26.76,
        0.16,
        0.6615
      ],
      [
        "important",
        26.92,
        0.54,
        0.9257
      ],
      [
        "to",
        27.5,
        0.1,
        1.0
      ],
      [
        "us",
        27.6,
        0.34,
        0.7955
      ],
      [
        "to",
        28.04,
        0.08,
        0.8545
      ],
      [
        "have",
        28.16,
        0.46,
        0.9994
      ],
      [
        "a",
        28.76,
        0.06,
        1.0
      ],
      [
        "product",
        28.82,
        0.56,
        0.9975
      ],
      [
        "that",
        29.38,
        0.08,
        0.5602
      ],
      [
        "we",
        29.46,
        0.16,
        0.7017
      ],
      [
        "can",
        29.62,
        0.22,
        1.0
      ],
      [
        "make",
        29.84,
        0.32,
        0.9643
      ],
      [
        "truly",
        30.42,
        0.32,
        0.6737
      ],
      [
        "available",
        30.74,
        0.6,
        0.9349
      ],
      [
        "and",
        31.4,
        0.2,
        0.4114
      ],
      [
        "broadly",
        31.6,
        0.44,
        0.6726
      ],
      [
        "available",
        32.04,
        0.58,
        0.9108
      ],
      [
        "to",
        32.72,
        0.06,
        1.0
      ],
      [
        "everyone.",
        32.78,
        0.64,
        0.7886
      ]
    ]
  }
}
]
```

**TXT format** (word-level):
```
[0.50-1.20] Hello
[1.20-2.30] world
```

**TextGrid format** (Praat-compatible):
```
Two tiers created:
- "utterances" tier: Full segments with original text
- "words" tier: Individual words with precise boundaries
```

**Use cases**:
- **Linguistic analysis**: Study pronunciation patterns, speech timing, and prosody
- **Accessibility**: Create more granular captions for hearing-impaired users
- **Video/Audio editing**: Enable precise word-level subtitle synchronization
- **Karaoke applications**: Highlight individual words as they are spoken
- **Language learning**: Provide precise word boundaries for pronunciation practice

**Usage**:
```bash
# Generate word-level aligned JSON
lai align --word_level audio.wav subtitle.srt output.json

# Create TextGrid file for Praat analysis
lai align --word_level audio.wav subtitle.srt output.TextGrid

# Word-level TXT output
lai align --word_level audio.wav subtitle.srt output.txt

# Standard subtitle with word-level events
lai align --word_level audio.wav subtitle.srt output.srt
```

**Combined with --split_sentence**:
```bash
# Optimal alignment: semantic splitting + word-level details
lai align --split_sentence --word_level audio.wav subtitle.srt output.json
```

### Python API

```python
from lattifai import LattifAI

client = LattifAI()  # api_key will be read from LATTIFAI_API_KEY if not provided
alignments, output_path = client.alignment(
    audio="audio.wav",
    subtitle="subtitle.srt",
    output_subtitle_path="output.srt",
)
```

Need to run inside an async application? Use the drop-in asynchronous client:

```python
import asyncio
from lattifai import AsyncLattifAI


async def main():
    async with AsyncLattifAI() as client:
        alignments, output_path = await client.alignment(
            audio="audio.wav",
            subtitle="subtitle.srt",
            split_sentence=False,
            output_subtitle_path="output.srt",
        )


asyncio.run(main())
```

Both clients return a list of `Supervision` segments with timing information and, if provided, the path where the aligned subtitle was written.

## Supported Formats

**Audio**: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, AIFF

**Video**: MP4, MKV, MOV, WEBM, AVI

**Subtitle Input**: SRT, VTT, ASS, SSA, SUB, SBV, TXT (plain text), Gemini (Google Gemini transcript format)

**Subtitle Output**: All input formats plus TextGrid (Praat format for linguistic analysis)

## API Reference

### LattifAI (sync)

```python
LattifAI(
    api_key: Optional[str] = None,
    model_name_or_path: str = 'Lattifai/Lattice-1-Alpha',
    device: str = 'cpu',  # 'cpu', 'cuda', or 'mps'
)
```

### AsyncLattifAI (async)

```python
AsyncLattifAI(
    api_key: Optional[str] = None,
    model_name_or_path: str = 'Lattifai/Lattice-1-Alpha',
    device: str = 'cpu',
)
```

Use `async with AsyncLattifAI() as client:` or call `await client.close()` when you are done to release the underlying HTTP session.

### alignment()

```python
client.alignment(
    audio: str,                           # Path to audio file
    subtitle: str,                        # Path to subtitle/text file
    format: Optional[str] = None,         # Input format: 'srt', 'vtt', 'ass', 'txt', 'gemini', or 'auto' (auto-detect if None)
    split_sentence: bool = False,         # Smart sentence splitting based on punctuation semantics
    return_details: bool = False,         # Enable word-level alignment details
    output_subtitle_path: Optional[str] = None
) -> Tuple[List[Supervision], Optional[str]]  # await client.alignment(...) for AsyncLattifAI
```

**Parameters**:
- `audio`: Path to the audio file to be aligned
- `subtitle`: Path to the subtitle or text file
- `format`: Input subtitle format. Supported values: 'srt', 'vtt', 'ass', 'txt', 'gemini', 'auto'. When set to None or 'auto', the format is automatically detected from file extension. Additional formats (ssa, sub, sbv) are supported through automatic format detection
- `split_sentence`: Enable intelligent sentence re-splitting (default: False). Set to True when subtitles combine multiple semantic units (non-speech elements + dialogue, or multiple sentences) that would benefit from separate timing alignment
- `return_details`: Enable word-level alignment details (default: False). When True, each `Supervision` object includes an `alignment` field with word-level timestamps, accessible via `supervision.alignment['word']`. This provides precise timing for each individual word within the segment
- `output_subtitle_path`: Output path for aligned subtitle (optional)

**Returns**:
- A tuple containing:
  - `alignments`: List of aligned `Supervision` objects with timing information
  - `output_subtitle_path`: Path where the subtitle was written (if `output_subtitle_path` was provided)

## Examples

### Basic Text Alignment

```python
from lattifai import LattifAI

client = LattifAI()
alignments, output_path = client.alignment(
    audio="speech.wav",
    subtitle="transcript.txt",
    format="txt",
    split_sentence=False,
    output_subtitle_path="output.srt"
)
```

### Word-Level Alignment

```python
from lattifai import LattifAI

client = LattifAI()
alignments, output_path = client.alignment(
    audio="speech.wav",
    subtitle="transcript.srt",
    return_details=True,  # Enable word-level alignment
    output_subtitle_path="output.json"  # JSON format preserves word-level data
)

# Access word-level timestamps
for segment in alignments:
    print(f"Segment: {segment.text} ({segment.start:.2f}s - {segment.end:.2f}s)")
    if segment.alignment and 'word' in segment.alignment:
        for word in segment.alignment['word']:
            print(f"  Word: {word.symbol} ({word.start:.2f}s - {word.end:.2f}s)")
```

### Batch Processing

```python
from pathlib import Path
from lattifai import LattifAI

client = LattifAI()
audio_dir = Path("audio_files")
subtitle_dir = Path("subtitles")
output_dir = Path("aligned")

for audio in audio_dir.glob("*.wav"):
    subtitle = subtitle_dir / f"{audio.stem}.srt"
    if subtitle.exists():
        alignments, output_path = client.alignment(
            audio=audio,
            subtitle=subtitle,
            output_subtitle_path=output_dir / f"{audio.stem}_aligned.srt"
        )
```

### GPU Acceleration

```python
from lattifai import LattifAI

# NVIDIA GPU
client = LattifAI(device='cuda')

# Apple Silicon
client = LattifAI(device='mps')

# CLI
lai align --device mps audio.wav subtitle.srt output.srt
```

### YouTube Processing with Agent Workflow

```python
import asyncio
from lattifai.workflows import YouTubeSubtitleAgent

async def process_youtube():
    # Initialize agent with configuration
    agent = YouTubeSubtitleAgent(
        gemini_api_key="your-gemini-api-key",
        video_format="mp4",  # or "mp3", "wav", etc.
        output_format="srt",
        max_retries=2,
        split_sentence=True,
        word_level=True,
        force_overwrite=False
    )

    # Process YouTube URL
    result = await agent.process_youtube_url(
        url="https://www.youtube.com/watch?v=VIDEO_ID",
        output_dir="./output",
        output_format="srt"
    )

    # Access results
    print(f"Title: {result['metadata']['title']}")
    print(f"Duration: {result['metadata']['duration']} seconds")
    print(f"Subtitle count: {result['subtitle_count']}")

    # Access generated files
    for format_name, file_path in result['exported_files'].items():
        print(f"{format_name.upper()}: {file_path}")

# Run the async workflow
asyncio.run(process_youtube())
```

## Configuration

### API Key Setup

First, create your API key at [https://lattifai.com/dashboard/api-keys](https://lattifai.com/dashboard/api-keys)

**Recommended: Using .env file**

Create a `.env` file in your project root:
```bash
LATTIFAI_API_KEY=your-api-key
```

The library automatically loads the `.env` file (python-dotenv is included as a dependency).

**Alternative: Environment variable**
```bash
export LATTIFAI_API_KEY="your-api-key"
```

## Model Information

**[Lattice-1-Alpha](https://huggingface.co/Lattifai/Lattice-1-Alpha)** features:
- State-of-the-art alignment precision
- **Language Support**: Currently supports English only. The upcoming **Lattice-1** release will support English, Chinese, and mixed English-Chinese content.
- Handles noisy audio and imperfect transcripts
- Optimized for CPU and GPU (CUDA/MPS)

**Requirements**:
- Python 3.9+
- 4GB RAM recommended
- ~2GB storage for model files

## Development

### Setup

```bash
git clone https://github.com/lattifai/lattifai-python.git
cd lattifai-python
pip install -e ".[test]"
./scripts/install-hooks.sh  # Optional: install pre-commit hooks
```

### Testing

```bash
pytest                        # Run all tests
pytest --cov=src             # With coverage
pytest tests/test_basic.py   # Specific test
```

### Code Quality

```bash
ruff check src/ tests/       # Lint
ruff format src/ tests/      # Format
isort src/ tests/            # Sort imports
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run `pytest` and `ruff check`
5. Submit a pull request

## License

Apache License 2.0

## Support

- **Issues**: [GitHub Issues](https://github.com/lattifai/lattifai-python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lattifai/lattifai-python/discussions)
- **Discord**: [Join our community](https://discord.gg/kvF4WsBRK8)
