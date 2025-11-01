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

# Convert subtitle format
lai subtitle convert input.srt output.vtt
```

> **💡 Tip**: Use `lai` for faster typing in your daily workflow!

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
  - **JSON**: Full alignment details stored in `alignment.word` field of each segment
  - **TextGrid**: Separate "words" tier alongside the "utterances" tier for linguistic analysis
  - **TXT**: Each word on a separate line with timestamp range: `[start-end] word`
  - **Standard subtitle formats** (SRT, VTT, ASS, etc.): Each word becomes a separate subtitle event

**Example output formats**:

**JSON format** (with word-level details):
```json
[
  {
    "id": "segment-001",
    "start": 0.5,
    "end": 2.3,
    "text": "Hello world",
    "alignment": {
      "word": [
        {"start": 0.5, "end": 1.2, "symbol": "Hello"},
        {"start": 1.2, "end": 2.3, "symbol": "world"}
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
- **Karaoke applications**: Highlight individual words as they are spoken
- **Language learning**: Provide precise word boundaries for pronunciation practice
- **Accessibility**: Create more granular captions for hearing-impaired users
- **Video editing**: Enable precise word-level subtitle synchronization

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
