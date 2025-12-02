<div align="center">
<img src="https://raw.githubusercontent.com/lattifai/lattifai-python/main/assets/logo.png" width=256>

[![PyPI version](https://badge.fury.io/py/lattifai.svg)](https://badge.fury.io/py/lattifai)
[![Python Versions](https://img.shields.io/pypi/pyversions/lattifai.svg)](https://pypi.org/project/lattifai)
[![PyPI Status](https://pepy.tech/badge/lattifai)](https://pepy.tech/project/lattifai)
</div>

<p align="center">
   🌐 <a href="https://lattifai.com"><b>Official Website</b></a> &nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://github.com/lattifai/lattifai-python">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/Lattifai/Lattice-1">Model</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://lattifai.com/blogs">Blog</a> &nbsp&nbsp | &nbsp&nbsp <a href="https://discord.gg/kvF4WsBRK8"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" style="vertical-align: middle;"></a>
</p>


# LattifAI: Precision Alignment, Infinite Possibilities

Advanced forced alignment and subtitle generation powered by [ 🤗 Lattice-1](https://huggingface.co/Lattifai/Lattice-1) model.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Command Line Interface](#command-line-interface)
  - [Python SDK (5 Lines of Code)](#python-sdk-5-lines-of-code)
- [CLI Reference](#cli-reference)
- [Python SDK Reference](#python-sdk-reference)
  - [Basic Alignment](#basic-alignment)
  - [YouTube Processing](#youtube-processing)
  - [Configuration Objects](#configuration-objects)
  - [Error Handling](#error-handling)
- [Advanced Features](#advanced-features)
- [Supported Formats](#supported-formats)
- [Roadmap](#roadmap)
- [Development](#development)

---

## Installation

### Step 1: Install SDK

**Using pip:**
```bash

pip install install-k2
install-k2  # Auto-detect PyTorch version and install compatible k2

pip install lattifai
```

**Using uv (Recommended - 10-100x faster):**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project with uv
uv init my-project
cd my-project

uv init
source .venv/bin/activate

# Install k2 (required dependency)
uv pip install install-k2
uv run install-k2

# Install LattifAI
uv pip install lattifai
```

> **Note**: `install-k2` automatically detects your PyTorch version (up to 2.9) and installs the compatible k2 wheel.

<details>
<summary><b>install-k2 options</b></summary>

```
usage: install-k2 [-h] [--system {linux,darwin,windows}] [--dry-run] [--torch-version TORCH_VERSION]

optional arguments:
  -h, help                      Show this help message and exit
  --system {linux,darwin,windows}  Override OS detection
  --dry-run                     Show what would be installed without making changes
  --torch-version TORCH_VERSION    Specify torch version (e.g., 2.8.0)
```
</details>

### Step 2: Get Your API Key

**LattifAI API Key (Required)**

Get your **free API key** at [https://lattifai.com/dashboard/api-keys](https://lattifai.com/dashboard/api-keys)

**Option A: Environment variable (recommended)**
```bash
export LATTIFAI_API_KEY="lf_your_api_key_here"
```

**Option B: `.env` file**
```bash
# .env
LATTIFAI_API_KEY=lf_your_api_key_here
```

**Gemini API Key (Optional - for transcription)**

If you want to use Gemini models for transcription (e.g., `gemini-2.5-pro`), get your **free Gemini API key** at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```bash
# Add to environment variable
export GEMINI_API_KEY="your_gemini_api_key_here"

# Or add to .env file
GEMINI_API_KEY=your_gemini_api_key_here  # AIzaSyxxxx
```

> **Note**: Gemini API key is only required if you use Gemini models for transcription. It's not needed for alignment or when using other transcription models.

---

## Quick Start

### Command Line Interface

```bash
# Align local audio with subtitle
lai alignment align audio.wav subtitle.srt output.srt

# Download and align YouTube video
lai alignment youtube "https://youtube.com/watch?v=VIDEO_ID"
```

### Python SDK (5 Lines of Code)

```python
from lattifai import LattifAI

client = LattifAI()
caption = client.alignment(
    input_media="audio.wav",
    input_caption="subtitle.srt",
    output_caption_path="aligned.srt",
)
```

That's it! Your aligned subtitles are saved to `aligned.srt`.

---

## CLI Reference

### Command Overview

| Command | Description |
|---------|-------------|
| `lai alignment align` | Align local audio/video with caption |
| `lai alignment youtube` | Download & align YouTube content |
| `lai transcribe file` | Transcribe audio/video to caption |
| `lai transcribe youtube` | Download & transcribe YouTube content |
| `lai caption convert`   | Convert between caption formats |
| `lai caption normalize` | Clean and normalize caption text |
| `lai caption shift`     | Shift caption timestamps |


### lai alignment align

```bash
# Basic usage
lai alignment align <audio> <caption> <output>

# Examples
lai alignment align audio.wav caption.srt output.srt
lai alignment align video.mp4 caption.vtt output.srt alignment.device=cuda
lai alignment align audio.wav caption.srt output.json \
    caption.split_sentence=true \
    caption.word_level=true
```

### lai alignment youtube

```bash
# Basic usage
lai alignment youtube <url>

# Examples
lai alignment youtube "https://youtube.com/watch?v=VIDEO_ID"
lai alignment youtube "https://youtube.com/watch?v=VIDEO_ID" \
    media.output_dir=~/Downloads \
    caption.output_path=aligned.srt \
    caption.split_sentence=true
```

### lai transcribe file

Perform automatic speech recognition (ASR) on audio/video files to generate timestamped transcriptions.

```bash
# Basic usage
lai transcribe file <audio> <output>

# Examples
lai transcribe file audio.wav output.srt

# Use specific transcription model
lai transcribe file audio.mp4 output.ass \
    transcription.model_name=nvidia/parakeet-tdt-0.6b-v3

# Use Gemini API for transcription
lai transcribe file audio.wav output.srt \
    transcription.model_name=gemini-2.5-pro \
    transcription.gemini_api_key=YOUR_GEMINI_API_KEY

# Specify language for transcription
lai transcribe file audio.wav output.srt \
    transcription.language=zh

# Full configuration with keyword arguments
lai transcribe file \
    input_media=audio.wav \
    output_caption=output.srt \
    transcription.device=cuda \
    transcription.model_name=iic/SenseVoiceSmall
```

**Supported Transcription Models (More Coming Soon):**
- `gemini-2.5-pro` - Google Gemini API (requires API key)
- `gemini-3-pro-preview` - Google Gemini API (requires API key)
- `nvidia/parakeet-tdt-0.6b-v3` - NVIDIA Parakeet model
- `iic/SenseVoiceSmall` - Alibaba SenseVoice model
- More models will be integrated in future releases


### lai transcribe youtube

Download YouTube video and perform automatic speech recognition without alignment.

```bash
# Basic usage
lai transcribe youtube <url> <output_dir>

# Examples
lai transcribe youtube "https://youtube.com/watch?v=VIDEO_ID" ./output

# Use Gemini API
lai transcribe youtube "https://youtube.com/watch?v=VIDEO_ID" ./output \
    transcription.model_name=gemini-2.5-pro \
    transcription.gemini_api_key=YOUR_GEMINI_API_KEY

# Use local model with specific device
lai transcribe youtube "https://youtube.com/watch?v=VIDEO_ID" ./output \
    transcription.model_name=nvidia/parakeet-tdt-0.6b-v3 \
    transcription.device=cuda

# Keyword argument syntax
lai transcribe youtube \
    url="https://youtube.com/watch?v=VIDEO_ID" \
    output_dir=./output \
    transcription.device=mps
```

**Note:** `lai transcribe youtube` performs transcription only. For transcription with alignment, use `lai alignment youtube` instead.


### lai caption convert

```bash
lai caption convert input.srt output.vtt
lai caption convert input.srt output.json normalize_text=true
```

### lai caption shift

```bash
lai caption shift input.srt output.srt 2.0    # Delay by 2 seconds
lai caption shift input.srt output.srt -1.5   # Advance by 1.5 seconds
```

---

## Python SDK Reference

### Basic Alignment

```python
from lattifai import LattifAI

# Initialize client (uses LATTIFAI_API_KEY from environment)
client = LattifAI()

# Align audio/video with subtitle
caption = client.alignment(
    input_media="audio.wav",           # Audio or video file
    input_caption="subtitle.srt",      # Input subtitle file
    output_caption_path="output.srt",  # Output aligned subtitle
    split_sentence=True,               # Enable smart sentence splitting
)

# Access alignment results
for segment in caption.supervisions:
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
```

### YouTube Processing

```python
from lattifai import LattifAI

client = LattifAI()

# Download YouTube video and align with auto-downloaded subtitles
caption = client.youtube(
    url="https://youtube.com/watch?v=VIDEO_ID",
    output_dir="./downloads",
    output_caption_path="aligned.srt",
    split_sentence=True,
)
```


### Configuration Objects

LattifAI uses a config-driven architecture for fine-grained control:

#### ClientConfig - API Settings

```python
from lattifai import LattifAI, ClientConfig

client = LattifAI(
    client_config=ClientConfig(
        api_key="lf_your_api_key",     # Or use LATTIFAI_API_KEY env var
        timeout=30.0,
        max_retries=3,
    )
)
```

#### AlignmentConfig - Model Settings

```python
from lattifai import LattifAI, AlignmentConfig

client = LattifAI(
    alignment_config=AlignmentConfig(
        model_name="Lattifai/Lattice-1",
        device="cuda",      # "cpu", "cuda", "cuda:0", "mps"
    )
)
```

#### CaptionConfig - Subtitle Settings

```python
from lattifai import LattifAI, CaptionConfig

client = LattifAI(
    caption_config=CaptionConfig(
        split_sentence=True,           # Smart sentence splitting
        word_level=True,               # Word-level timestamps
        normalize_text=True,           # Clean HTML entities
        include_speaker_in_text=False, # Include speaker labels
    )
)
```

#### Complete Configuration Example

```python
from lattifai import (
    LattifAI,
    ClientConfig,
    AlignmentConfig,
    CaptionConfig
)

client = LattifAI(
    client_config=ClientConfig(
        api_key="lf_your_api_key",
        timeout=60.0,
    ),
    alignment_config=AlignmentConfig(
        model_name="Lattifai/Lattice-1",
        device="cuda",
    ),
    caption_config=CaptionConfig(
        split_sentence=True,
        word_level=True,
        output_format="json",
    ),
)

caption = client.alignment(
    input_media="audio.wav",
    input_caption="subtitle.srt",
    output_caption_path="output.json",
)
```

### Available Exports

```python
from lattifai import (
    # Client classes
    LattifAI,
    # AsyncLattifAI,  # For async support

    # Config classes
    ClientConfig,
    AlignmentConfig,
    CaptionConfig,
    DiarizationConfig,
    MediaConfig,

    # I/O classes
    Caption,
)
```

---

## Advanced Features

### Word-Level Alignment

Enable `word_level=True` to get precise timestamps for each word:

```python
from lattifai import LattifAI, CaptionConfig

client = LattifAI(
    caption_config=CaptionConfig(word_level=True)
)

caption = client.alignment(
    input_media="audio.wav",
    input_caption="subtitle.srt",
    output_caption_path="output.json",  # JSON preserves word-level data
)

# Access word-level alignments
for segment in caption.alignments:
    if segment.alignment and "word" in segment.alignment:
        for word, start, duration, confidence in segment.alignment.word:
            print(f"{start:.2f}s: {word} (confidence: {confidence:.2f})")
```

### Smart Sentence Splitting

The `split_sentence` option intelligently separates:
- Non-speech elements (`[APPLAUSE]`, `[MUSIC]`) from dialogue
- Multiple sentences within a single subtitle
- Speaker labels from content

```python
caption = client.alignment(
    input_media="audio.wav",
    input_caption="subtitle.srt",
    split_sentence=True,
)
```

### WIP: Speaker Diarization

```python
from lattifai import LattifAI, DiarizationConfig

client = LattifAI(
    diarization_config=DiarizationConfig(enabled=True)
)

caption = client.alignment(
    input_media="audio.wav",
    input_caption="subtitle.srt",
    output_caption_path="output.srt",
)
```

### YAML Configuration Files

Create reusable configuration files:

```yaml
# config/alignment.yaml
model_name: "Lattifai/Lattice-1"
device: "cuda"
batch_size: 1
```

```bash
lai alignment align audio.wav subtitle.srt output.srt \
    alignment=config/alignment.yaml
```

---

## Supported Formats

LattifAI supports virtually all common media and subtitle formats:

| Type | Formats |
|------|---------|
| **Audio** | WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, AIFF, and more |
| **Video** | MP4, MKV, MOV, WEBM, AVI, and more |
| **Subtitle Input** | SRT, VTT, ASS, SSA, SUB, SBV, TXT, Gemini, and more |
| **Subtitle Output** | All input formats + TextGrid (Praat) |

> **Note**: If a format is not listed above but commonly used, it's likely supported. Feel free to try it or reach out if you encounter any issues.

---

## Roadmap

Visit our [LattifAI roadmap](https://lattifai.com/roadmap) for the latest updates.

| Date | Release | Features |
|------|---------|----------|
| **Oct 2025** | **Lattice-1-Alpha** | ✅ English forced alignment<br>✅ Multi-format support<br>✅ CPU/GPU optimization |
| **Nov 2025** | **Lattice-1** | ✅ English + Chinese + German<br>✅ Mixed languages alignment<br>🚀 Integrate Speaker Diarization |

---

## Development

### Setup

```bash
git clone https://github.com/lattifai/lattifai-python.git
cd lattifai-python

# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# Or using pip
pip install -e ".[test]"

pre-commit install
```

### Testing

```bash
pytest                        # Run all tests
pytest --cov=src              # With coverage
pytest tests/test_basic.py    # Specific test
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run `pytest` and `pre-commit run`
5. Submit a pull request

## License

Apache License 2.0

## Support

- **Issues**: [GitHub Issues](https://github.com/lattifai/lattifai-python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lattifai/lattifai-python/discussions)
- **Discord**: [Join our community](https://discord.gg/kvF4WsBRK8)
