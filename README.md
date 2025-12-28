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
  - [Web Interface](#web-interface)
- [CLI Reference](#cli-reference)
  - [lai alignment align](#lai-alignment-align)
  - [lai alignment youtube](#lai-alignment-youtube)
  - [lai transcribe run](#lai-transcribe-run)
  - [lai caption convert](#lai-caption-convert)
  - [lai caption shift](#lai-caption-shift)
- [Python SDK Reference](#python-sdk-reference)
  - [Basic Alignment](#basic-alignment)
  - [YouTube Processing](#youtube-processing)
  - [Configuration Objects](#configuration-objects)
- [Advanced Features](#advanced-features)
  - [Word-Level Alignment](#word-level-alignment)
  - [Smart Sentence Splitting](#smart-sentence-splitting)
  - [Speaker Diarization](#speaker-diarization)
  - [YAML Configuration Files](#yaml-configuration-files)
- [Supported Formats](#supported-formats)
- [Roadmap](#roadmap)
- [Development](#development)

---

## Installation

### Step 1: Install SDK

**Using pip:**
```bash

pip install install-k2
install-k2 --torch-version 2.9.1  # if not set will auto-detect PyTorch version and install compatible k2

pip install lattifai
```

**Using uv (Recommended - 10-100x faster):**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project with uv
uv init my-project
cd my-project
source .venv/bin/activate

# Install k2 (required dependency)
uv pip install install-k2
uv pip install pip
uv run install-k2 --torch-version 2.9.1

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

![CLI Demo](assets/cli.png)

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

### Web Interface

![web Demo](assets/web.png)

1. **Install the web application (one-time setup):**
   ```bash
   lai-app-install
   ```

   This command will:
   - Check if Node.js/npm is installed (and install if needed)
   - Install frontend dependencies
   - Build the application
   - Setup the `lai-app` command globally

2. **Start the backend server:**
   ```bash
   lai-server

   # Custom port (default: 8001)
   lai-server --port 9000

   # Custom host
   lai-server --host 127.0.0.1 --port 9000

   # Production mode (disable auto-reload)
   lai-server --no-reload
   ```

   **Backend Server Options:**
   - `-p, --port` - Server port (default: 8001)
   - `--host` - Host address (default: 0.0.0.0)
   - `--no-reload` - Disable auto-reload for production
   - `-h, --help` - Show help message

3. **Start the frontend application:**
   ```bash
   lai-app

   # Custom port (default: 5173)
   lai-app --port 8080

   # Custom backend URL
   lai-app --backend http://localhost:9000

   # Don't auto-open browser
   lai-app --no-open
   ```

   **Frontend Application Options:**
   - `-p, --port` - Frontend server port (default: 5173)
   - `--backend` - Backend API URL (default: http://localhost:8001)
   - `--no-open` - Don't automatically open browser
   - `-h, --help` - Show help message

   The web interface will automatically open in your browser at `http://localhost:5173`.

**Features:**
- ✅ Automatic backend server status detection
- ✅ Visual file upload with drag-and-drop
- ✅ Real-time alignment progress
- ✅ Multiple subtitle format support
- ✅ Built-in transcription with multiple models
- ✅ API key management interface
- ✅ Download aligned subtitles in various formats

---

## CLI Reference

### Command Overview

| Command | Description |
|---------|-------------|
| `lai alignment align` | Align local audio/video with caption |
| `lai alignment youtube` | Download & align YouTube content |
| `lai transcribe run` | Transcribe audio/video or YouTube URL to caption |
| `lai transcribe align` | Transcribe audio/video and align with generated transcript |
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

### lai transcribe run

Perform automatic speech recognition (ASR) on audio/video files or YouTube URLs to generate timestamped transcriptions.

```bash
# Basic usage - local file
lai transcribe run <input> <output>

# Basic usage - YouTube URL
lai transcribe run <url> <output_dir>

# Examples - Local files
lai transcribe run audio.wav output.srt
lai transcribe run audio.mp4 output.ass \
    transcription.model_name=nvidia/parakeet-tdt-0.6b-v3

# Examples - YouTube URLs
lai transcribe run "https://youtube.com/watch?v=VIDEO_ID" output_dir=./output
lai transcribe run "https://youtube.com/watch?v=VIDEO_ID" output.ass output_dir=./output \
    transcription.model_name=gemini-2.5-pro \
    transcription.gemini_api_key=YOUR_GEMINI_API_KEY

# Full configuration with keyword arguments
lai transcribe run \
    input=audio.wav \
    output_caption=output.srt \
    channel_selector=average \
    transcription.device=cuda \
    transcription.model_name=iic/SenseVoiceSmall
```

**Parameters:**
- `input`: Path to audio/video file or YouTube URL (required)
- `output_caption`: Path for output caption file (for local files)
- `output_dir`: Directory for output files (for YouTube URLs, defaults to current directory)
- `media_format`: Media format for YouTube downloads (default: mp3)
- `channel_selector`: Audio channel selection - "average", "left", "right", or channel index (default: "average")
  - Note: Ignored when transcribing YouTube URLs with Gemini models
- `transcription`: Transcription configuration (model_name, device, language, gemini_api_key)

**Supported Transcription Models (More Coming Soon):**
- `gemini-2.5-pro` - Google Gemini API (requires API key)
  - Languages: 100+ languages including English, Chinese, Spanish, French, German, Japanese, Korean, Arabic, and more
- `gemini-3-pro-preview` - Google Gemini API (requires API key)
  - Languages: 100+ languages (same as gemini-2.5-pro)
- `nvidia/parakeet-tdt-0.6b-v3` - NVIDIA Parakeet model
  - Languages: Bulgarian (bg), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), English (en), Estonian (et), Finnish (fi), French (fr), German (de), Greek (el), Hungarian (hu), Italian (it), Latvian (lv), Lithuanian (lt), Maltese (mt), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk), Slovenian (sl), Spanish (es), Swedish (sv), Russian (ru), Ukrainian (uk)
- `iic/SenseVoiceSmall` - Alibaba SenseVoice model
  - Languages: Chinese/Mandarin (zh), English (en), Japanese (ja), Korean (ko), Cantonese (yue)
- More models will be integrated in future releases

**Note:** For transcription with alignment on local files, use `lai transcribe align` instead.

### lai transcribe align

Transcribe audio/video file and automatically align the generated transcript with the audio.

This command combines transcription and alignment in a single step, producing precisely aligned captions.

```bash
# Basic usage
lai transcribe align <input_media> <output_caption>

# Examples
lai transcribe align audio.wav output.srt
lai transcribe align audio.mp4 output.ass \
    transcription.model_name=nvidia/parakeet-tdt-0.6b-v3 \
    alignment.device=cuda

# Using Gemini transcription with alignment
lai transcribe align audio.wav output.srt \
    transcription.model_name=gemini-2.5-pro \
    transcription.gemini_api_key=YOUR_KEY \
    caption.split_sentence=true

# Full configuration
lai transcribe align \
    input_media=audio.wav \
    output_caption=output.srt \
    transcription.device=mps \
    transcription.model_name=iic/SenseVoiceSmall \
    alignment.device=cuda \
    caption.word_level=true
```

**Parameters:**
- `input_media`: Path to input audio/video file (required)
- `output_caption`: Path for output aligned caption file (required)
- `transcription`: Transcription configuration (model_name, device, language, gemini_api_key)
- `alignment`: Alignment configuration (model_name, device)
- `caption`: Caption formatting options (split_sentence, word_level, etc.)


### lai caption convert

```bash
lai caption convert input.srt output.vtt
lai caption convert input.srt output.json
# Enable normalization to clean HTML entities and special characters:
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
        split_sentence=True,           # Smart sentence splitting (default: False)
        word_level=True,               # Word-level timestamps (default: False)
        normalize_text=True,           # Clean HTML entities (default: True)
        include_speaker_in_text=False, # Include speaker labels (default: True)
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

### Long-Form Audio Support

LattifAI now supports processing long audio files (up to 20 hours) through streaming mode. Enable streaming by setting the `streaming_chunk_secs` parameter:

**Python SDK:**
```python
from lattifai import LattifAI

client = LattifAI()

# Enable streaming for long audio files
caption = client.alignment(
    input_media="long_audio.wav",
    input_caption="subtitle.srt",
    output_caption_path="output.srt",
    streaming_chunk_secs=600.0,  # Process in 30-second chunks
)
```

**CLI:**
```bash
# Enable streaming with chunk size
lai alignment align long_audio.wav subtitle.srt output.srt \
    media.streaming_chunk_secs=300.0

# For YouTube videos
lai alignment youtube "https://youtube.com/watch?v=VIDEO_ID" \
    media.streaming_chunk_secs=300.0
```

**MediaConfig:**
```python
from lattifai import LattifAI, MediaConfig

client = LattifAI(
    media_config=MediaConfig(
        streaming_chunk_secs=600.0,  # Chunk duration in seconds (1-1800), default: 600 (10 minutes)
    )
)
```

**Notes:**
- Chunk duration must be between 1 and 1800 seconds (minimum 1 second, maximum 30 minutes)
- Default value: 600 seconds (10 minutes)
- **Recommended: Use 60 seconds or larger for optimal performance**
- Set to `None` to disable streaming
- **Thanks to our precise implementation, streaming has virtually no impact on alignment accuracy**
- Smaller chunks reduce memory usage with minimal quality trade-off
- Recommended chunk size: 300-900 seconds (5-15 minutes) for optimal balance

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
        for word_item in segment.alignment["word"]:
            print(f"{word_item.start:.2f}s: {word_item.symbol} (confidence: {word_item.score:.2f})")
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

### Speaker Diarization

Speaker diarization automatically identifies and labels different speakers in audio. When enabled, the system will:
- Detect speaker changes in the audio
- Assign speaker labels (e.g., SPEAKER_00, SPEAKER_01) to each segment
- Update subtitle segments with speaker information

**Speaker Name Handling:**
- **Existing speaker labels in subtitles**: If your input captions already contain speaker names (e.g., `[Alice]`, `>> Bob:`, or `SPEAKER_01:`), the system will preserve them as much as possible during alignment
- **Gemini Transcriber**: When using Gemini models for transcription (e.g., `gemini-2.5-pro`), the model can intelligently identify and extract speaker names from dialogue context, making it easier to generate speaker-aware transcripts

**CLI:**
```bash
# Enable speaker diarization during alignment
lai alignment align audio.wav subtitle.srt output.srt \
    diarization.enabled=true

# With additional diarization settings
lai alignment align audio.wav subtitle.srt output.srt \
    diarization.enabled=true \
    diarization.device=cuda \
    diarization.min_speakers=2 \
    diarization.max_speakers=4

# For YouTube videos with diarization
lai alignment youtube "https://youtube.com/watch?v=VIDEO_ID" \
    diarization.enabled=true
```

**Python SDK:**
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

# Access speaker information
for segment in caption.supervisions:
    print(f"[{segment.speaker}] {segment.text}")
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
| **Caption/Subtitle Input** | SRT, VTT, ASS, SSA, SUB, SBV, TXT, Gemini, and more |
| **Caption/Subtitle Output** | All input formats + TextGrid (Praat) |

**Tabular Formats:**
- **TSV**: Tab-separated values with optional speaker column
- **CSV**: Comma-separated values with optional speaker column
- **AUD**: Audacity labels format with `[[speaker]]` notation

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
