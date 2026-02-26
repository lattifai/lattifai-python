<div align="center">
<img src="https://raw.githubusercontent.com/lattifai/lattifai-python/main/assets/logo.png" width=256>

[![PyPI version](https://badge.fury.io/py/lattifai.svg)](https://badge.fury.io/py/lattifai)
[![Python Versions](https://img.shields.io/pypi/pyversions/lattifai.svg)](https://pypi.org/project/lattifai)
[![PyPI Status](https://pepy.tech/badge/lattifai)](https://pepy.tech/project/lattifai)
</div>

<p align="center">
   🌐 <a href="https://lattifai.com"><b>Official Website</b></a> &nbsp;&nbsp; | &nbsp;&nbsp; 🖥️ <a href="https://github.com/lattifai/lattifai-python">GitHub</a> &nbsp;&nbsp; | &nbsp;&nbsp; 🤗 <a href="https://huggingface.co/LattifAI/Lattice-1">Model</a> &nbsp;&nbsp; | &nbsp;&nbsp; 📑 <a href="https://lattifai.com/blogs">Blog</a> &nbsp;&nbsp; | &nbsp;&nbsp; <a href="https://discord.gg/kvF4WsBRK8"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" style="vertical-align: middle;"></a>
</p>


# LattifAI: Precision Alignment, Infinite Possibilities

Advanced forced alignment and subtitle generation powered by [ 🤗 Lattice-1](https://huggingface.co/LattifAI/Lattice-1) model.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python SDK](#python-sdk)
- [Advanced Features](#advanced-features)
- [Text Processing](#text-processing)
- [Supported Formats & Languages](#supported-formats--languages)
- [Roadmap](#roadmap)
- [Development](#development)

---

## Features

| Feature | Description |
|---------|-------------|
| **Forced Alignment** | Word-level and segment-level audio-text synchronization powered by [Lattice-1](https://huggingface.co/LattifAI/Lattice-1) |
| **Multi-Model Transcription** | Gemini (100+ languages), Parakeet (24 languages), SenseVoice (5 languages) |
| **Speaker Diarization** | Multi-speaker identification with label preservation |
| **Streaming Mode** | Process audio up to 20 hours with minimal memory |
| **Universal Format Support** | 30+ caption/subtitle formats |

### Alignment Models

| Model | Links | Languages | Description |
|-------|-------|-----------|-------------|
| **Lattice-1** | [🤗 HF](https://huggingface.co/LattifAI/Lattice-1) • [🤖 MS](https://modelscope.cn/models/LattifAI/Lattice-1) | English, Chinese, German | Production model with mixed-language alignment support |
| **Lattice-1-Alpha** | [🤗 HF](https://huggingface.co/LattifAI/Lattice-1-Alpha) • [🤖 MS](https://modelscope.cn/models/LattifAI/Lattice-1-Alpha) | English | Initial release with English forced alignment |

**Model Hub**: Models can be downloaded from `huggingface` (default) or `modelscope` (recommended for users in China):

```bash
# Use ModelScope (faster in China)
lai alignment align audio.wav caption.srt output.srt alignment.model_hub=modelscope
```

```python
from lattifai.client import LattifAI
from lattifai.config import AlignmentConfig

client = LattifAI(alignment_config=AlignmentConfig(model_hub="modelscope"))
```

---

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager (10-100x faster than pip). **No extra configuration needed** - uv automatically uses our package index.

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project and add lattifai
uv init my-project && cd my-project
uv add "lattifai[all]" --extra-index-url https://lattifai.github.io/pypi/simple/

# Or add to an existing project
uv add "lattifai[all]" --extra-index-url https://lattifai.github.io/pypi/simple/

# Run CLI without installing (quick test)
uvx --from lattifai --extra-index-url https://lattifai.github.io/pypi/simple/ lai --help
```

### Using pip

```bash
# Full installation (recommended)
pip install "lattifai[all]" --extra-index-url https://lattifai.github.io/pypi/simple/
```

**Configure pip globally** (optional, to avoid `--extra-index-url` each time):

```bash
# Add to ~/.pip/pip.conf (Linux/macOS) or %APPDATA%\pip\pip.ini (Windows)
[global]
extra-index-url = https://lattifai.github.io/pypi/simple/
```

### Installation Options

| Extra | Includes |
|-------|----------|
| (base) | Forced alignment (Lattice-1, k2py, ONNX, captions and YouTube) |
| `all` | Base + transcription + youtube |
| `transcription` | ASR models (Gemini, Parakeet, SenseVoice) |
| `diarization` | Speaker diarization (NeMo, pyannote) |
| `event` | Audio event detection |

**Note:** Base installation includes full alignment functionality. Use `[all]` for transcription and YouTube features.

### Caption Format Support

Caption/subtitle format parsing is provided by [lattifai-captions](https://github.com/lattifai/captions), a separate package supporting 30+ formats (SRT, VTT, ASS, TTML, TextGrid, NLE formats, etc.). It is automatically installed with `lattifai[core]` or `lattifai[all]`.

### API Keys

**LattifAI API Key (Required)** - Get your free key at [lattifai.com/dashboard/api-keys](https://lattifai.com/dashboard/api-keys)

```bash
export LATTIFAI_API_KEY="lf_your_api_key_here"
```

**Gemini API Key (Optional)** - For transcription with Gemini models, get key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

Or use a `.env` file:
```bash
LATTIFAI_API_KEY=lf_your_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Quick Start

### Command Line

```bash
# Align audio with subtitle
lai alignment align audio.wav subtitle.srt output.srt

# YouTube video
lai alignment youtube "https://youtube.com/watch?v=VIDEO_ID"
```

### Python SDK

```python
from lattifai.client import LattifAI

client = LattifAI()
caption = client.alignment(
    input_media="audio.wav",
    input_caption="subtitle.srt",
    output_caption_path="aligned.srt",
)
```

---

## CLI Reference

| Command | Description | Example |
|---------|-------------|---------|
| `lai alignment align` | Align audio/video with caption | `lai alignment align audio.wav caption.srt output.srt` |
| `lai alignment youtube` | Download & align YouTube | `lai alignment youtube "https://youtube.com/watch?v=ID"` |
| `lai transcribe run` | Transcribe audio/video | `lai transcribe run audio.wav output.srt` |
| `lai transcribe align` | Transcribe and align | `lai transcribe align audio.wav output.srt` |
| `lai caption convert` | Convert caption formats | `lai caption convert input.srt output.vtt` |
| `lai caption shift` | Shift timestamps | `lai caption shift input.srt output.srt 2.0` |

### Common Options

```bash
# Device selection
alignment.device=cuda          # cuda, mps, cpu

# Caption options
caption.split_sentence=true    # Smart sentence splitting
caption.word_level=true        # Word-level timestamps

# Streaming for long audio
media.streaming_chunk_secs=600

# Channel selection
media.channel_selector=left    # left, right, average, or index
```

### Transcription Models

```bash
# Gemini (100+ languages, requires GEMINI_API_KEY)
transcription.model_name=gemini-2.5-pro

# Parakeet (24 European languages)
transcription.model_name=nvidia/parakeet-tdt-0.6b-v3

# SenseVoice (zh, en, ja, ko, yue)
transcription.model_name=iic/SenseVoiceSmall
```

### lai transcribe run

Transcribe audio/video files or YouTube URLs to generate timestamped captions.

```bash
# Local file
lai transcribe run audio.wav output.srt

# YouTube URL
lai transcribe run "https://youtube.com/watch?v=VIDEO_ID" output_dir=./output

# With model selection
lai transcribe run audio.wav output.srt \
    transcription.model_name=gemini-2.5-pro \
    transcription.device=cuda
```

**Parameters:**
- `input`: Path to audio/video file or YouTube URL
- `output_caption`: Output caption file path (for local files)
- `output_dir`: Output directory (for YouTube URLs, defaults to current directory)
- `channel_selector`: Audio channel - `average` (default), `left`, `right`, or channel index

### lai transcribe align

Transcribe and align in a single step - produces precisely aligned captions.

```bash
# Basic usage
lai transcribe align audio.wav output.srt

# With options
lai transcribe align audio.wav output.srt \
    transcription.model_name=nvidia/parakeet-tdt-0.6b-v3 \
    alignment.device=cuda \
    caption.split_sentence=true \
    caption.word_level=true
```

---

## Python SDK

### Configuration Objects

```python
from lattifai.client import LattifAI
from lattifai.config import (
    ClientConfig,
    AlignmentConfig,
    CaptionConfig,
    DiarizationConfig,
    MediaConfig,
)

client = LattifAI(
    client_config=ClientConfig(api_key="lf_xxx", timeout=60.0),
    alignment_config=AlignmentConfig(device="cuda"),
    caption_config=CaptionConfig(split_sentence=True, word_level=True),
)

caption = client.alignment(
    input_media="audio.wav",
    input_caption="subtitle.srt",
    output_caption_path="output.json",
)

# Access results
for segment in caption.supervisions:
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
```

### YouTube Processing

```python
caption = client.youtube(
    url="https://youtube.com/watch?v=VIDEO_ID",
    output_dir="./downloads",
    output_caption_path="aligned.srt",
)
```

### CaptionConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `split_sentence` | `False` | Smart sentence splitting, separates non-speech elements |
| `word_level` | `False` | Include word-level timestamps in output |
| `normalize_text` | `True` | Clean HTML entities and special characters |
| `include_speaker_in_text` | `True` | Include speaker labels in text output |

```python
from lattifai.client import LattifAI
from lattifai.config import CaptionConfig

client = LattifAI(
    caption_config=CaptionConfig(
        split_sentence=True,
        word_level=True,
        normalize_text=True,
        include_speaker_in_text=False,
    )
)
```

---

## Advanced Features

### Streaming Mode (Long Audio)

Process audio up to 20 hours with minimal memory:

```python
caption = client.alignment(
    input_media="long_audio.wav",
    input_caption="subtitle.srt",
    streaming_chunk_secs=600.0,  # 10-minute chunks
)
```

### Word-Level Alignment

```python
from lattifai.client import LattifAI
from lattifai.config import CaptionConfig

client = LattifAI(caption_config=CaptionConfig(word_level=True))
caption = client.alignment(
    input_media="audio.wav",
    input_caption="subtitle.srt",
    output_caption_path="output.json",  # JSON preserves word-level data
)
```

### Speaker Diarization

Automatically identify and label different speakers in audio.

**Capabilities:**
- **Multi-Speaker Detection**: Automatically detect speaker changes
- **Smart Labeling**: Assign labels (SPEAKER_00, SPEAKER_01, etc.)
- **Label Preservation**: Maintain existing speaker names from input captions
- **Gemini Integration**: Extract speaker names from transcription context

**Label Handling:**
- Without existing labels → Generic labels (SPEAKER_00, SPEAKER_01)
- With existing labels (`[Alice]`, `>> Bob:`, `SPEAKER_01:`) → Preserved during alignment
- Gemini transcription → Names extracted from context (e.g., "Hi, I'm Alice" → `Alice`)

```python
from lattifai.client import LattifAI
from lattifai.config import DiarizationConfig

client = LattifAI(
    diarization_config=DiarizationConfig(
        enabled=True,
        device="cuda",
        min_speakers=2,
        max_speakers=4,
    )
)
caption = client.alignment(...)

for segment in caption.supervisions:
    print(f"[{segment.speaker}] {segment.text}")
```

**CLI:**
```bash
lai alignment align audio.wav subtitle.srt output.srt \
    diarization.enabled=true \
    diarization.device=cuda
```

### Data Flow

```
Input Media → AudioLoader → Aligner → (Diarizer) → Caption
                              ↑
Input Caption → Reader → Tokenizer
```

---

## Text Processing

The tokenizer handles various text patterns for forced alignment.

### Bracket/Caption Handling

Visual captions and annotations in brackets are treated specially - they get **two pronunciation paths** so the aligner can choose:
1. **Silence path** - skip when content doesn't appear in audio
2. **Inner text pronunciation** - match if someone actually says the words

| Bracket Type | Symbol | Example | Alignment Behavior |
|--------------|--------|---------|-------------------|
| Half-width square | `[]` | `[APPLAUSE]` | Skip or match "applause" |
| Half-width paren | `()` | `(music)` | Skip or match "music" |
| Full-width square | `【】` | `【笑声】` | Skip or match "笑声" |
| Full-width paren | `（）` | `（音乐）` | Skip or match "音乐" |
| Angle brackets | `<>` | `<intro>` | Skip or match "intro" |
| Book title marks | `《》` | `《开场白》` | Skip or match "开场白" |

This allows proper handling of:
- **Visual descriptions**: `[Barret adjusts the camera and smiles]` → skipped if not spoken
- **Sound effects**: `[APPLAUSE]`, `(music)` → matched if audible
- **Chinese annotations**: `【笑声】`, `（鼓掌）` → flexible alignment

### Multilingual Text

| Pattern | Handling | Example |
|---------|----------|---------|
| CJK characters | Split individually | `你好` → `["你", "好"]` |
| Latin words | Grouped with accents | `Kühlschrank` → `["Kühlschrank"]` |
| Contractions | Kept together | `I'm`, `don't`, `we'll` |
| Punctuation | Attached to words | `Hello,` `world!` |

### Speaker Labels

Recognized speaker patterns are preserved during alignment:

| Format | Example | Output |
|--------|---------|--------|
| Arrow prefix | `>> Alice:` or `&gt;&gt; Alice:` | `[Alice]` |
| LattifAI format | `[SPEAKER_01]:` | `[SPEAKER_01]` |
| Uppercase name | `SPEAKER NAME:` | `[SPEAKER NAME]` |

---

## Supported Formats & Languages

### Media Formats

| Type | Formats |
|------|---------|
| **Audio** | WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, AIFF, and more |
| **Video** | MP4, MKV, MOV, WEBM, AVI, and more |
| **Caption** | SRT, VTT, ASS, SSA, SRV3, JSON, TextGrid, TSV, CSV, LRC, TTML, and more |

> **Note**: Caption format handling is provided by [lattifai-captions](https://github.com/lattifai/captions), which is automatically installed as a dependency. For standalone caption processing without alignment features, install `pip install lattifai-captions`.

### JSON Format

JSON is the most flexible format for storing caption data with full word-level timing support:

```json
[
    {
        "text": "Hello beautiful world",
        "start": 0.0,
        "end": 2.5,
        "speaker": "Speaker 1",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "beautiful", "start": 0.6, "end": 1.4},
            {"word": "world", "start": 1.5, "end": 2.5}
        ]
    }
]
```

**Features:**
- Word-level timestamps preserved in `words` array
- Round-trip compatible (read/write without data loss)
- Optional `speaker` field for multi-speaker content

### Word-Level and Karaoke Output

| Format | `word_level=True` | `word_level=True` + `karaoke=True` |
|--------|-------------------|-----------------------------------|
| **JSON** | Includes `words` array | Same as word_level=True |
| **SRT** | One word per segment | One word per segment |
| **VTT** | One word per segment | YouTube VTT style: `<00:00:00.000><c> word</c>` |
| **ASS** | One word per segment | `{\kf}` karaoke tags (sweep effect) |
| **LRC** | One word per line | Enhanced `<timestamp>` tags |
| **TTML** | One word per `<p>` element | `<span>` with `itunes:timing="Word"` |

### VTT Format (YouTube VTT Support)

The VTT format handler supports both standard WebVTT and YouTube VTT with word-level timestamps.

**Reading**: VTT automatically detects YouTube VTT format (with `<timestamp><c>` tags) and extracts word-level alignment data:

```
WEBVTT

00:00:00.000 --> 00:00:02.000
<00:00:00.000><c> Hello</c><00:00:00.500><c> world</c>
```

**Writing**: Use `word_level=True` with `karaoke_config` to output YouTube VTT style:

```python
from lattifai.caption import Caption
from lattifai.caption.config import KaraokeConfig

caption = Caption.read("input.vtt")
caption.write(
    "output.vtt",
    word_level=True,
    karaoke_config=KaraokeConfig(enabled=True)
)
```

```bash
# CLI: Convert to YouTube VTT with word-level timestamps
lai caption convert input.json output.vtt \
    caption.word_level=true \
    caption.karaoke.enabled=true
```

### Transcription Language Support

#### Gemini Models (100+ Languages)

**Models**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-pro-preview`

English, Chinese (Mandarin & Cantonese), Spanish, French, German, Italian, Portuguese, Japanese, Korean, Arabic, Russian, Hindi, Bengali, Turkish, Dutch, Polish, Swedish, Danish, Norwegian, Finnish, Greek, Hebrew, Thai, Vietnamese, Indonesian, Malay, Filipino, Ukrainian, Czech, Romanian, Hungarian, and 70+ more.

> Requires Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

#### NVIDIA Parakeet (24 European Languages)

**Model**: `nvidia/parakeet-tdt-0.6b-v3`

| Region | Languages |
|--------|-----------|
| Western Europe | English (en), French (fr), German (de), Spanish (es), Italian (it), Portuguese (pt), Dutch (nl) |
| Nordic | Danish (da), Swedish (sv), Norwegian (no), Finnish (fi) |
| Eastern Europe | Polish (pl), Czech (cs), Slovak (sk), Hungarian (hu), Romanian (ro), Bulgarian (bg), Ukrainian (uk), Russian (ru) |
| Others | Croatian (hr), Estonian (et), Latvian (lv), Lithuanian (lt), Slovenian (sl), Maltese (mt), Greek (el) |

#### Alibaba SenseVoice (5 Asian Languages)

**Model**: `iic/SenseVoiceSmall`

Chinese/Mandarin (zh), English (en), Japanese (ja), Korean (ko), Cantonese (yue)

---

## Roadmap

Visit [lattifai.com/roadmap](https://lattifai.com/roadmap) for updates.

| Date | Release | Features |
|------|---------|----------|
| **Oct 2025** | Lattice-1-Alpha | ✅ English forced alignment, multi-format support |
| **Nov 2025** | Lattice-1 | ✅ EN+ZH+DE, speaker diarization, multi-model transcription |
| **Q1 2026** | Lattice-2 | ✅ Streaming mode, 🔮 40+ languages, real-time alignment |

---

## Development

```bash
git clone https://github.com/lattifai/lattifai-python.git
cd lattifai-python

# Using uv (recommended, auto-configures extra index)
uv sync && source .venv/bin/activate

# Or pip (requires extra-index-url for lattifai-core)
pip install -e ".[all,dev]" --extra-index-url https://lattifai.github.io/pypi/simple/

# Run tests
pytest

# Install pre-commit hooks
pre-commit install
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run `pytest` and `pre-commit run --all-files`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Support

- **Issues**: [GitHub Issues](https://github.com/lattifai/lattifai-python/issues)
- **Discord**: [Join our community](https://discord.gg/kvF4WsBRK8)

## License

Apache License 2.0
