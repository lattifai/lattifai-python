<div align="center">
<img src="https://raw.githubusercontent.com/lattifai/lattifai-python/main/assets/logo.png" width=256>

[![PyPI version](https://badge.fury.io/py/lattifai.svg)](https://badge.fury.io/py/lattifai)
[![Python Versions](https://img.shields.io/pypi/pyversions/lattifai.svg)](https://pypi.org/project/lattifai)
[![PyPI Status](https://pepy.tech/badge/lattifai)](https://pepy.tech/project/lattifai)
</div>

<p align="center">
   🌐 <a href="https://lattifai.com"><b>Official Website</b></a> &nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://github.com/lattifai/lattifai-python">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/Lattifai/Lattice-1-Alpha">Model</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://lattifai.com/blogs">Blog</a> &nbsp&nbsp | &nbsp&nbsp <a href="https://discord.gg/kvF4WsBRK8"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" style="vertical-align: middle;"></a>
</p>


# LattifAI: Precision Alignment, Infinite Possibilities

Advanced forced alignment and subtitle generation powered by [Lattice-1-Alpha](https://huggingface.co/Lattifai/Lattice-1-Alpha) model.

## Installation

```bash
pip install install-k2
# The installation will automatically detect and use your already installed PyTorch version(up to 2.8).
install-k2  # Install k2

pip install lattifai
```
> **⚠️ Important**: You must run `install-k2` before using the lattifai library.
```
> install-k2 help
usage: install-k2 [-h] [--system {linux,darwin,windows}] [--dry-run] [--torch-version TORCH_VERSION]

Auto-install the latest k2 wheel for your environment.

optional arguments:
  -h, help            show this help message and exit
  system {linux,darwin,windows}
                        Override OS detection. Valid values: linux, darwin (macOS), windows. Default: auto-detect
  dry-run             Show what would be installed without making changes.
  torch-version TORCH_VERSION
                        Specify torch version (e.g., 2.8.0). If not specified, will auto-detect or use latest available.
```


## Quick Start

### Command Line Interface

LattifAI provides a powerful CLI powered by [NeMo Run](https://github.com/lattifai/Run), offering flexible configuration management and execution capabilities.

```bash
# Align audio with subtitle
lai alignment align media.input_path=audio.wav \
                    subtitle.input_path=subtitle.srt \
                    subtitle.output_path=output.srt

# Download and align YouTube content
lai alignment youtube media.input_path="https://youtube.com/watch?v=VIDEO_ID" \
                      subtitle.input_path=subtitle.srt

# Run intelligent YouTube workflow (download → transcribe → align → export)
lai agent workflow media.input_path="https://youtube.com/watch?v=VIDEO_ID" \
                   transcription.api_key=YOUR_GEMINI_KEY

# Convert subtitle formats
lai subtitle convert input_path=input.srt output_path=output.vtt

# Normalize subtitle text (clean HTML entities)
lai subtitle normalize input_path=input.srt output_path=output.srt
```

> **💡 New to NeMo Run?** Check out the [Configuration Guide](#advanced-configuration-with-nemo-run) below to learn about powerful features like YAML configs, config reuse, and parameter sweeps.

#### Command Quick Reference

| Command | Use Case | Best For |
|---------|----------|----------|
| `lai alignment align` | Align existing audio + subtitle files | Local files, custom workflows |
| `lai alignment youtube` | Download & align YouTube content | Quick YouTube processing with existing subtitles |
| `lai subtitle convert` | Convert subtitle formats | Format conversion only |
| `lai subtitle normalize` | Clean and normalize subtitle text | Text preprocessing |
| `lai agent workflow(under construction)` | Intelligent YouTube workflow with transcription | Production, batch jobs, full automation |


#### lai alignment align

Align audio/video with subtitle files using forced alignment.

**Basic Usage:**
```bash
# Simple alignment
lai alignment align media.input_path=audio.wav \
                    subtitle.input_path=subtitle.srt \
                    subtitle.output_path=output.srt

# With GPU acceleration and word_level alignment
lai alignment align media.input_path=audio.mp4 \
                    subtitle.input_path=subtitle.srt \
                    subtitle.output_path=output.json \
                    alignment.device=cuda \
                    subtitle.word_level=true

# Smart sentence splitting with custom output format
lai alignment align media.input_path=audio.wav \
                    subtitle.input_path=subtitle.srt \
                    subtitle.output_path=output.vtt \
                    subtitle.split_sentence=true \
                    subtitle.output_format=vtt
```

**Common Options:**
- `media.input_path`: Path to audio/video file (required)
- `subtitle.input_path`: Path to subtitle file (required)
- `subtitle.output_path`: Output path for aligned subtitle
- `subtitle.split_sentence`: Enable intelligent sentence splitting
- `subtitle.word_level`: Include word_level timestamps
- `alignment.device`: Device to use (`cpu`, `cuda`, or `mps`)
- `alignment.model_name_or_path`: Model to use for alignment

#### lai alignment youtube

Download media and optionally subtitles from YouTube, then perform forced alignment.

**Basic Usage:**
```bash
# Basic YouTube alignment with existing subtitle
lai alignment youtube media.input_path="https://youtu.be/VIDEO_ID" \
                      subtitle.input_path=subtitle.srt

# Full configuration with custom output
lai alignment youtube media.input_path="https://youtu.be/VIDEO_ID" \
                      media.output_dir=/tmp/youtube \
                      media.output_format=wav \
                      subtitle.input_path=subtitle.srt \
                      subtitle.output_path=aligned.srt \
                      subtitle.split_sentence=true \
                      alignment.device=mps
```

**Common Options:**
- `media.input_path`: YouTube URL or local file path (required)
- `subtitle.input_path`: Path to subtitle file
- `media.output_dir`: Directory for downloaded media
- `media.output_format`: Media format (`mp3`, `wav`, `mp4`, etc.)
- `subtitle.output_path`: Output path for aligned subtitle
- `alignment.device`: Device to use for alignment

#### lai agent workflow (under construction)

Run the intelligent agentic YouTube workflow with automatic transcription, alignment, and export.

This command provides a complete end-to-end pipeline:
1. Download media from YouTube URL
2. Transcribe audio using Gemini API (if no subtitle provided)
3. Align transcription with audio
4. Export results in the desired format

**Basic Usage:**
```bash
# Basic YouTube video processing
lai agent workflow media.input_path="https://youtu.be/VIDEO_ID" \
                   transcription.api_key=YOUR_GEMINI_KEY

# Enable speaker diarization and sentence splitting
lai agent workflow media.input_path="https://youtu.be/VIDEO_ID" \
                   transcription.enable_diarization=true \
                   subtitle.split_sentence=true \
                   transcription.api_key=YOUR_GEMINI_KEY

# Full configuration with retries
lai agent workflow \
    media.input_path="https://youtu.be/VIDEO_ID" \
    media.output_dir=/tmp/youtube \
    media.output_format=wav \
    subtitle.output_format=json \
    subtitle.word_level=true \
    subtitle.split_sentence=true \
    alignment.device=cuda \
    transcription.api_key=YOUR_GEMINI_KEY \
    transcription.enable_diarization=true \
    max_retries=3
```

**Common Options:**
_ `media.input_path`: YouTube URL (required)
_ `transcription.api_key`: Gemini API key for transcription (required)
_ `transcription.enable_diarization`: Enable speaker diarization
_ `subtitle.word_level`: Include word_level timestamps
_ `subtitle.split_sentence`: Enable smart sentence splitting
_ `max_retries`: Maximum retries for failed operations

> **� Tip**: Get your free Gemini API key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

#### lai subtitle convert

Convert subtitle files between different formats.

**Basic Usage:**
```bash
# Basic format conversion
lai subtitle convert input.srt output.vtt

# Convert to TextGrid with speaker info
lai subtitle convert input.srt output.TextGrid \
    subtitle.include_speaker_in_text=true

# Convert with text normalization
lai subtitle convert input.json output.srt \
    subtitle.normalize_text=true
```

**Arguments:**
- First argument: Input subtitle file path
- Second argument: Output subtitle file path
- `subtitle.include_speaker_in_text`: Include speaker labels in text
- `subtitle.normalize_text`: Normalize text during conversion

#### lai subtitle normalize

Normalize subtitle text by cleaning HTML entities, tags, and whitespace.

**Basic Usage:**
```bash
# Normalize in-place
lai subtitle normalize input.srt input.srt

# Normalize and save to new file
lai subtitle normalize input.srt output.srt

# Normalize with format conversion
lai subtitle normalize input.vtt output.srt
```

**Arguments:**
- First argument: Input subtitle file path
- Second argument: Output subtitle file path

#### Understanding split_sentence

The `split_sentence` option performs intelligent sentence re-splitting based on punctuation and semantic boundaries. This is especially useful when processing subtitles that combine multiple semantic units in a single segment, such as:

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
lai alignment align subtitle.split_sentence=true \
                    media.input_path=audio.wav \
                    subtitle.input_path=subtitle.srt \
                    subtitle.output_path=output.srt
```

#### Understanding word_level

The `word_level` option enables word_level alignment, providing precise timing information for each individual word in the audio. When enabled, the output includes detailed word boundaries within each subtitle segment, allowing for fine-grained synchronization and analysis.

**Key features**:
- **Individual word timestamps**: Each word gets its own start and end time
- **Format-specific output**:
  - **JSON (Recommended)**: Full alignment details stored in `alignment.word` field of each segment, preserving all word_level timing information in a structured format
  - **TextGrid**: Separate "words" tier alongside the "utterances" tier for linguistic analysis
  - **TXT**: Each word on a separate line with timestamp range: `[start-end] word`
  - **Standard subtitle formats** (SRT, VTT, ASS, etc.): Each word becomes a separate subtitle event

> **💡 Recommended**: Use JSON format (`output.json`) to preserve complete word_level alignment data. Other formats may lose some structural information.

**Example output formats**:

**JSON format** (with word_level details):
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

**TXT format** (word_level):
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
- **Video/Audio editing**: Enable precise word_level subtitle synchronization
- **Karaoke applications**: Highlight individual words as they are spoken
- **Language learning**: Provide precise word boundaries for pronunciation practice

**Usage**:
```bash
# Generate word_level aligned JSON
lai alignment align subtitle.word_level=true \
                 media.input_path=audio.wav \
                 subtitle.input_path=subtitle.srt \
                 subtitle.output_path=output.json

# Create TextGrid file for Praat analysis
lai alignment align subtitle.word_level=true \
                 media.input_path=audio.wav \
                 subtitle.input_path=subtitle.srt \
                 subtitle.output_path=output.TextGrid
```

**Combined with split_sentence**:
```bash
# Optimal alignment: semantic splitting
lai alignment align subtitle.split_sentence=true \
                    media.input_path=audio.wav \
                    subtitle.input_path=subtitle.srt \
                    subtitle.output_path=output.json
```

### Advanced Configuration with LattifAI Run

LattifAI CLI is built on [NeMo Run](https://github.com/lattifai/Run), a powerful configuration and execution framework that provides:

- **📋 YAML-based Configurations**: Define reusable configuration files
- **🔧 Pythonic Config Objects**: Use `run.Config` for type-safe parameter management
- **🎯 Config Composition**: Combine multiple configuration files
- **🔄 Parameter Sweeps**: Run experiments with different parameter combinations
- **📦 Config Reuse**: Share configurations across different commands

> **Note**: We use a customized version of [NVIDIA NeMo Run](https://github.com/NVIDIA-NeMo/Run) tailored for LattifAI's workflow requirements. For full documentation, visit [https://github.com/lattifai/Run](https://github.com/lattifai/Run).

#### Using Configuration Files

**1. Create a YAML configuration file:**

```yaml
# config/media.yaml
input_path: "audio.wav"
output_dir: "/tmp/output"
output_format: "wav"
prefer_audio: true
force_overwrite: false

# config/subtitle.yaml
input_path: "subtitle.srt"
output_path: "output.srt"
output_format: "srt"
split_sentence: true
word_level: true
normalize_text: true

# config/alignment.yaml
model_name_or_path: "Lattifai/Lattice-1-Alpha"
device: "mps"
batch_size: 1
```

**2. Use the configuration file:**

```bash
# Load configuration from YAML files
lai alignment align media=config/media.yaml \
                    subtitle=config/subtitle.yaml \
                    alignment=config/alignment.yaml

# Override specific parameters
lai alignment align media=config/media.yaml \
                    media.input_path=different_audio.wav \
                    subtitle=config/subtitle.yaml \
                    alignment.device=cuda
```

#### Using Python Configuration Files

Create reusable configuration objects in Python:

```python
# configs/my_config.py
import nemo_run as run
from lattifai.config import MediaConfig, SubtitleConfig, AlignmentConfig

# Define configurations
media_config = run.Config(
    MediaConfig,
    input_path="audio.wav",
    output_dir="/tmp/output",
    prefer_audio=True,
)

subtitle_config = run.Config(
    SubtitleConfig,
    input_path="subtitle.srt",
    output_path="output.srt",
    split_sentence=True,
    word_level=True,
)

alignment_config = run.Config(
    AlignmentConfig,
    device="mps",
    model_name_or_path="Lattifai/Lattice-1-Alpha",
)
```

**Use the Python config:**

```bash
# Import configs from Python file
lai alignment align media=configs.my_config.media_config \
                    subtitle=configs.my_config.subtitle_config \
                    alignment=configs.my_config.alignment_config
```

#### Configuration Composition

Combine multiple configuration sources:

```bash
# Base config from YAML + overrides from command line
lai alignment align media=config/base_media.yaml \
                    media.input_path=new_audio.wav \
                    subtitle.split_sentence=true \
                    subtitle.word_level=true \
                    alignment.device=cuda
```

#### Configuration for YouTube Workflow

**Create a complete workflow configuration:**

```yaml
# config/youtube_workflow.yaml
media:
  input_path: "https://youtu.be/VIDEO_ID"
  output_dir: "/tmp/youtube"
  output_format: "wav"
  prefer_audio: true

subtitle:
  output_format: "json"
  split_sentence: true
  word_level: true
  include_speaker_in_text: true

alignment:
  device: "cuda"
  model_name_or_path: "Lattifai/Lattice-1-Alpha"

transcription:
  api_key: "${GEMINI_API_KEY}"  # Reference environment variable
  enable_diarization: true
  language: "en"
```

**Run the workflow:**

```bash
# Use the complete workflow config
lai agent workflow media=config/youtube_workflow.yaml

# Or load from Python
# configs/workflows.py
import os
import nemo_run as run
from lattifai.config import (
    MediaConfig,
    SubtitleConfig,
    AlignmentConfig,
    TranscriptionConfig,
)

youtube_media = run.Config(
    MediaConfig,
    input_path="https://youtu.be/VIDEO_ID",
    output_dir="/tmp/youtube",
    prefer_audio=True,
)

youtube_subtitle = run.Config(
    SubtitleConfig,
    output_format="json",
    split_sentence=True,
    word_level=True,
)

youtube_transcription = run.Config(
    TranscriptionConfig,
    api_key=os.getenv("GEMINI_API_KEY"),
    enable_diarization=True,
)

# Run with Python configs
lai agent workflow media=configs.workflows.youtube_media \
                      subtitle=configs.workflows.youtube_subtitle \
                      transcription=configs.workflows.youtube_transcription
```

#### Benefits of Configuration Files

1. **Reusability**: Define once, use across multiple experiments
2. **Reproducibility**: Version control your configurations
3. **Documentation**: Configs serve as self-documenting code
4. **Team Collaboration**: Share standardized configurations
5. **Environment Flexibility**: Easy switching between dev/prod settings

For more advanced features like parameter sweeps and experiment management, see the [LattifAI Run documentation](https://github.com/lattifai/Run/tree/main/docs).

### Python API

```python
from lattifai import LattifAI

client = LattifAI()
alignments, output_path = client.alignment(
            input_media_path="audio.wav",
            input_subtitle_path="subtitle.srt",
            split_sentence=False,
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
            input_media_path="audio.wav",
            input_subtitle_path="subtitle.srt",
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

```

### AsyncLattifAI (async)

```python

```

Use `async with AsyncLattifAI() as client:` or call `await client.close()` when you are done to release the underlying HTTP session.


## Examples

### GPU Acceleration

```python
from lattifai import LattifAI

# NVIDIA GPU
lai alignment align alignment.device=cuda:0 \
                    media.input_path=audio.wav \
                    subtitle.input_path=subtitle.srt \
                    subtitle.output_path=output.srt

# Apple Silicon
lai alignment align alignment.device=mps \
                    media.input_path=audio.wav \
                    subtitle.input_path=subtitle.srt \
                    subtitle.output_path=output.srt
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
- Python 3.10 - 3.13 (3.14 support coming soon)
- 4GB RAM recommended
- ~2GB storage for model files

## Development

### Setup

```bash
git clone https://github.com/lattifai/lattifai-python.git
cd lattifai-python
pip install -e ".[test]"
pre-commit install
```

### Testing

```bash
pytest                        # Run all tests
pytest cov=src             # With coverage
pytest tests/test_basic.py   # Specific test
```

### Code Quality

```bash
pre-commit run
```

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
