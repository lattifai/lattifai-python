# LattifAI Python

[![PyPI version](https://badge.fury.io/py/lattifai.svg)](https://badge.fury.io/py/lattifai)

<p align="center">
   🌐 <a href="https://lattifai.com"><b>Official Website</b></a> &nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://github.com/lattifai/lattifai-python">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/Lattifai/Lattice-1-Alpha">Model</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://lattifai.com/blogs">Blog</a> &nbsp&nbsp | &nbsp&nbsp <a href="https://discord.gg/gTZqdaBJ"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" style="vertical-align: middle;"></a>
</p>

Advanced forced alignment and subtitle generation powered by [Lattice-1-Alpha](https://huggingface.co/Lattifai/Lattice-1-Alpha) model.

## Installation

```bash
pip install install-k2
# The installation will automatically detect and use your already installed PyTorch version.
install-k2  # Install k2

pip install lattifai
```

> **⚠️ Important**: You must run `install-k2` before using the lattifai library.

## Quick Start

### Command Line

```bash
# Align audio with subtitle
lattifai align audio.wav subtitle.srt output.srt

# Convert subtitle format
lattifai subtitle convert input.srt output.vtt
```
#### lattifai align options
```
> lattifai align --help
Usage: lattifai align [OPTIONS] INPUT_AUDIO_PATH INPUT_SUBTITLE_PATH OUTPUT_SUBTITLE_PATH

  Command used to align audio with subtitles

Options:
  -F, --input_format [srt|vtt|ass|txt|auto]  Input Subtitle format.
  -D, --device [cpu|cuda|mps]                Device to use for inference.
  --split_sentence                           Smart sentence splitting based on punctuation semantics.
  --help                                     Show this message and exit.
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
lattifai align --split_sentence audio.wav subtitle.srt output.srt
```

### Python API

```python
from lattifai import LattifAI

# Initialize client
client = LattifAI(
    api_key=None,  # Or set LATTIFAI_API_KEY environment variable
    model_name_or_path='Lattifai/Lattice-1-Alpha',
    device='cpu',  # 'cpu', 'cuda', or 'mps'
)

# Perform alignment
alignments, output_path = client.alignment(
    audio="audio.wav",
    subtitle="subtitle.srt",
    split_sentence=False,
    output_subtitle_path="output.srt"
)
```

## Supported Formats

**Audio**: WAV, MP3, FLAC, M4A, OGG
**Subtitle**: SRT, VTT, ASS, TXT (plain text)

## API Reference

### LattifAI

```python
LattifAI(
    api_key: Optional[str] = None,
    model_name_or_path: str = 'Lattifai/Lattice-1-Alpha',
    device: str = 'cpu'  # 'cpu', 'cuda', or 'mps'
)
```

### alignment()

```python
client.alignment(
    audio: str,                           # Path to audio file
    subtitle: str,                        # Path to subtitle/text file
    format: Optional[str] = None,         # 'srt', 'vtt', 'ass', 'txt' (auto-detect if None)
    split_sentence: bool = False,         # Smart sentence splitting based on punctuation semantics
    output_subtitle_path: Optional[str] = None
) -> Tuple[List[Supervision], Optional[str]]
```

**Parameters**:
- `audio`: Path to the audio file to be aligned
- `subtitle`: Path to the subtitle or text file
- `format`: Subtitle format ('srt', 'vtt', 'ass', 'txt'). Auto-detected if None
- `split_sentence`: Enable intelligent sentence re-splitting (default: False). Set to True when subtitles combine multiple semantic units (non-speech elements + dialogue, or multiple sentences) that would benefit from separate timing alignment
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
lattifai align --device mps audio.wav subtitle.srt output.srt
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
- **Discord**: [Join our community](https://discord.gg/gTZqdaBJ)
