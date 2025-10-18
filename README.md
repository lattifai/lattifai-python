# LattifAI Python

<!-- <p align="center">
  <a href="https://badge.fury.io/py/lattifai"><img src="https://badge.fury.io/py/lattifai.svg" alt="PyPI version"></a>
</p> -->

[![PyPI version](https://badge.fury.io/py/lattifai.svg)](https://badge.fury.io/py/lattifai)

<p align="center">
   &nbsp&nbsp 🖥️ <a href="https://github.com/lattifai/lattifai-python">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Lattifai/Lattice-1-Alpha">Lattifai/Lattice-1-Alpha</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://lattifai.com/blogs">Blog</a> &nbsp&nbsp |  &nbsp&nbsp <a href="https://discord.gg/gTZqdaBJ"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" style="vertical-align: middle;"></a>&nbsp&nbsp
</p>

The official Python library for the LattifAI API - Advanced forced alignment and subtitle generation powered by `Lattice-1-Alpha` model.

## Installation

```bash
pip install lattifai

install-k2  # Required: This step must be executed to install k2 dependencies
```

> **⚠️ Important**: After installing `lattifai`, you **must** run `install-k2` to install the required k2 library. The library will not function properly without this step.

## GPU Support Status

> **🚧 GPU Support Coming Soon**: The current version supports CPU-only processing. NVIDIA GPU and Apple Silicon acceleration will be available in an upcoming release, providing significant performance improvements for large audio files.

## Quick Start

### Command Line Interface

The library provides powerful command-line tools for batch processing and automation.

#### Basic Alignment

```bash
# Align audio with subtitle file
lattifai-align input_audio.wav input_subtitle.srt output_aligned.srt

# Specify input format explicitly
lattifai-align -F srt input_audio.wav transcript.srt output.srt

# Auto-detect input/output format (default)
lattifai-align input_audio.wav subtitle_file.vtt output.vtt
```

#### Using the Main CLI

```bash
# Show available commands
lattifai --help

# Perform alignment (same as lattifai-align)
lattifai align input_audio.wav input_subtitle.srt output.srt

# Subtitle format conversion
lattifai subtitle convert input.srt output.vtt
```

### Python API

```python
import os
from lattifai import LattifAI

# Initialize client
client = LattifAI(
    api_key=os.environ.get("LATTIFAI_API_KEY"),  # Optional if set in environment
    model_name_or_path='Lattifai/Lattice-1-Alpha',
    device='cpu',  # Currently only CPU is supported, GPU support coming soon
)

# Perform alignment
result = client.alignment(
    audio="path/to/audio.wav",
    subtitle="path/to/subtitle.srt",  # or .txt, .vtt, .ass
    format="srt",  # auto-detect if not specified
    output_subtitle_path="output.srt"
)

print(f"Alignment complete! Output saved to: {result}")
```


#### Supported Input Formats

- **SRT**: SubRip Subtitle format (`.srt`)
- **VTT**: WebVTT format (`.vtt`)
- **ASS**: Advanced SubStation Alpha format (`.ass`)
- **TXT**: Plain text format (`.txt`)
- **AUTO**: Automatic format detection (default)

#### Command Line Options

- `-F, --input_format`: Specify input subtitle format (`srt`, `vtt`, `ass`, `txt`, `auto`)
- `--help`: Show help message and available options

## API Reference

### LattifAI Class

#### Constructor Parameters

```python
LattifAI(
    api_key: Optional[str] = None,           # API key (or set LATTIFAI_API_KEY env var)
    device: str = 'cpu',                     # Device for processing (currently only 'cpu' supported)
)
```

> **Note**: The `device` parameter currently only supports `'cpu'`. GPU support will be available in future releases.

#### Methods

##### `alignment()`

Perform forced alignment between audio and text/subtitles.

```python
def alignment(
    audio: Pathlike,                         # Path to audio file
    subtitle: Pathlike,                      # Path to subtitle/text file
    format: Optional[SubtitleFormat] = None, # Input format (auto-detect if None)
    output_subtitle_path: Optional[Pathlike] = None  # Output file path
) -> str:
```

**Parameters:**
- `audio`: Path to the audio file (WAV, MP3, etc.)
- `subtitle`: Path to subtitle or text file
- `format`: Input format (`'srt'`, `'vtt'`, `'ass'`, `'txt'`, or `None` for auto-detection)
- `output_subtitle_path`: Path where aligned subtitle will be saved

**Returns:**
- Path to output file (if `output_subtitle_path` specified) or alignment results

## Configuration

### Environment Variables

- `LATTIFAI_API_KEY`: Your LattifAI API key (required)

### Performance Tuning

For better performance with large files:

```python
client = LattifAI(
    device='cpu',         # Currently only CPU is supported, cuda/mps will be supported in upcoming releases.
)
```

> **GPU Acceleration**: CUDA and Apple Silicon(MPS) support is in active development and will significantly improve processing speed for large audio files. Expected in upcoming releases.

## Examples

### Basic Text Alignment

```python
from lattifai import LattifAI

client = LattifAI()

# Align plain text with audio
result = client.alignment(
    audio="speech.wav",
    subtitle="Hello world. This is a test.",
    format="txt",
    output_subtitle_path="aligned.srt"
)
```

### Subtitle Refinement

```python
# Improve existing subtitle timing
result = client.alignment(
    audio="movie.mp4",
    subtitle="rough_subtitles.srt",
    output_subtitle_path="precise_subtitles.srt"
)
```

### Batch Processing

```python
import os
from pathlib import Path

client = LattifAI()

# Process multiple files
audio_dir = Path("audio_files")
subtitle_dir = Path("subtitles")
output_dir = Path("aligned_output")

for audio_file in audio_dir.glob("*.wav"):
    subtitle_file = subtitle_dir / f"{audio_file.stem}.srt"
    output_file = output_dir / f"{audio_file.stem}_aligned.srt"

    if subtitle_file.exists():
        client.alignment(
            audio=audio_file,
            subtitle=subtitle_file,
            output_subtitle_path=output_file
        )
```

## Supported Formats

### Audio Formats
- **WAV**: Recommended for best quality
- **MP3**: Widely supported compressed format
- **FLAC**: Lossless compression
- **M4A**: Apple audio format
- **OGG**: Open source audio format

### Subtitle Formats
- **SRT**: SubRip Subtitle format (most common)
- **VTT**: WebVTT format (web standard)
- **ASS**: Advanced SubStation Alpha format (advanced styling)
- **TXT**: Plain text format (one sentence per line)

## Error Handling

```python
from lattifai import LattifAI, LattifAIError

try:
    client = LattifAI()
    result = client.alignment(
        audio="audio.wav",
        subtitle="subtitle.srt"
    )
except LattifAIError as e:
    print(f"LattifAI error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Common Error Scenarios

- **Missing API Key**: Set `LATTIFAI_API_KEY` environment variable
- **File Not Found**: Verify input file paths exist
- **Unsupported Format**: Check supported audio/subtitle formats
- **Network Issues**: Check internet connection and API endpoint
- **Device Configuration**: Currently only `device='cpu'` is supported (GPU support coming soon)

## Model Information

This library uses the **[Lattice-1-Alpha](https://huggingface.co/Lattifai/Lattice-1-Alpha)** model for high-quality forced alignment and subtitle generation.

### Model Features
- **High Accuracy**: State-of-the-art alignment precision
- **Multi-language Support**: Supports various languages and accents
- **Robust Processing**: Handles noisy audio and imperfect transcripts
- **Fast Processing**: Optimized for both CPU and GPU execution

### Model Requirements
- **Minimum Python**: 3.9+
- **Memory**: 4GB RAM recommended
- **Storage**: ~2GB for model files
- **GPU**: CUDA support coming soon (currently CPU-only)

## Requirements

- **Python**: 3.9-3.13
- **Core Dependencies**:
  - httpx (HTTP client)
  - lhotse (audio processing)
  - k2 (audio computation)
  - python-dotenv (environment management)
  - click (command line interface)

## Development

### Setting Up Development Environment

1. **Clone the repository:**
```bash
git clone https://github.com/lattifai/lattifai-python.git
cd lattifai-python
```

2. **Install development dependencies:**
```bash
pip install -e ".[test]"
```

3. **Install Git hooks (recommended):**
```bash
./scripts/install-hooks.sh
```

This will install pre-commit hooks that automatically run `isort` and `ruff` checks before each commit to ensure code quality.

### Code Quality Tools

We use the following tools to maintain code quality:

- **isort**: Automatically sorts and organizes imports
- **ruff**: Fast Python linter and formatter

To manually run these tools:

```bash
# Sort imports
isort src/ tests/ scripts/

# Run linter
ruff check src/ tests/ scripts/

# Run formatter
ruff format src/ tests/ scripts/

# Fix issues automatically
ruff check --fix src/ tests/ scripts/
```

### Running Tests

```bash
# Install test dependencies first
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_basic.py -v
```

### Building and Testing Locally

```bash
# Build package
python -m build

# Install locally
pip install -e .

# Test CLI commands
lattifai --help
lattifai-align --help
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Run code quality checks: `ruff check src/ tests/`
6. Commit your changes: `git commit -am 'Add some feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

### Bypassing Pre-commit Hooks

If you need to commit without running the hooks (not recommended):

```bash
git commit --no-verify
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [API Documentation](https://github.com/lattifai/lattifai-python/README.md)
- **Issues**: [GitHub Issues](https://github.com/lattifai/lattifai-python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lattifai/lattifai-python/discussions)
- **Changelog**: [CHANGELOG.md](https://github.com/lattifai/lattifai-python/CHANGELOG.md)
