# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LattifAI is a Python SDK for precision audio-text forced alignment powered by the Lattice-1 model. The project provides:
- **Forced alignment**: Word-level and segment-level audio-text synchronization
- **Multi-model transcription**: Gemini (100+ languages), Parakeet (24 languages), SenseVoice (5 languages)
- **Speaker diarization**: Multi-speaker identification with label preservation
- **Universal format support**: 30+ caption/subtitle formats
- **Multiple interfaces**: CLI, Python SDK, and Web UI

## Essential Commands

### Development Setup

```bash
# Using uv (recommended - 10-100x faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# Or using pip
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_basic.py

# Run specific test
pytest tests/caption/test_formats.py::test_format_name
```

### Code Quality

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Format code with black (line length: 120)
black src/ tests/ --line-length=120

# Sort imports with isort
isort src/ tests/ --profile=black --line-length=120

# Lint with flake8
flake8 src/ tests/ --config=.flake8 --ignore=E203
```

### CLI Commands

The project uses `nemo_run` for CLI entry points. All CLI commands are in `src/lattifai/cli/`:

```bash
# Alignment
lai alignment align <audio> <caption> <output>
lai alignment youtube <url>

# Transcription
lai transcribe run <input> <output>
lai transcribe align <input_media> <output_caption>

# Caption utilities
lai caption convert <input> <output>
lai caption shift <input> <output> <offset_seconds>

# Server
lai-server --port 8001
lai-app-install  # Install web UI
lai-app          # Run web UI
```

## Architecture

### Core Design Pattern: Config-Driven

LattifAI uses a **config-driven architecture** with `nemo_run` for maximum flexibility and reproducibility:

```
Configuration Layer (nemo_run Config objects)
├── ClientConfig       # API client settings
├── AlignmentConfig    # Lattice-1 model & device
├── CaptionConfig      # I/O formats & processing
├── TranscriptionConfig # ASR model selection
├── DiarizationConfig  # Speaker detection
└── MediaConfig        # Audio/video loading
```

### Module Structure

```
src/lattifai/
├── client.py              # Main LattifAI client (config-driven)
├── config/                # Configuration classes (all inherit from nemo_run.Config)
│   ├── client.py          # API settings
│   ├── alignment.py       # Model & device config
│   ├── caption.py         # I/O format config
│   ├── transcription.py   # ASR model config
│   ├── diarization.py     # Speaker detection config
│   └── media.py           # Audio loading config
├── alignment/             # Forced alignment engine
│   ├── lattice1_aligner.py    # Main aligner using Lattice-1 model
│   ├── lattice1_worker.py     # Low-level alignment worker
│   ├── tokenizer.py           # Text preprocessing
│   ├── sentence_splitter.py   # Smart sentence splitting
│   ├── segmenter.py           # Audio segmentation
│   └── text_align.py          # Supervision-transcription alignment
├── caption/               # Caption I/O and data structures
│   ├── caption.py         # Caption dataclass (container for all results)
│   ├── supervision.py     # Supervision segment dataclass
│   ├── formats/           # Format handlers (30+ formats)
│   │   ├── base.py        # Base reader/writer classes
│   │   ├── pysubs2.py     # SRT, VTT, ASS, SSA via pysubs2
│   │   ├── gemini.py      # Gemini API format
│   │   ├── textgrid.py    # Praat TextGrid
│   │   ├── tabular.py     # TSV, CSV, AUD
│   │   └── nle/           # NLE formats (Premiere, FCPXML, etc.)
│   └── parsers/           # Text parsing utilities
├── transcription/         # Multi-model transcription
│   ├── base.py            # BaseTranscriber interface
│   ├── gemini.py          # Gemini API (100+ languages)
│   ├── lattifai.py        # Local models (Parakeet, SenseVoice)
│   └── __init__.py        # create_transcriber() factory
├── diarization/           # Speaker diarization
│   └── lattifai.py        # LattifAIDiarizer using pyannote.audio
├── youtube/               # YouTube download/processing
│   └── client.py          # YouTube downloader
├── workflow/              # Agentic workflow system
│   ├── base.py            # WorkflowAgent, WorkflowStep
│   └── file_manager.py    # File existence handling
├── cli/                   # CLI entry points (nemo_run)
│   ├── alignment.py       # lai alignment commands
│   ├── transcribe.py      # lai transcribe commands
│   ├── caption.py         # lai caption commands
│   └── youtube.py         # lai youtube commands
└── server/                # FastAPI web server
    └── app.py
```

### Key Data Structures

**Caption** (`caption/caption.py`):
- Central dataclass containing all results
- Fields: `supervisions`, `transcription`, `audio_events`, `speaker_diarization`, `alignments`
- Handles reading/writing 30+ caption formats via format handlers
- Supports metadata (language, kind, source_format, source_path)

**Supervision** (`caption/supervision.py`):
- Individual subtitle segment with timing and text
- Fields: `start`, `end`, `text`, `speaker`, `custom_dict`
- Can contain word-level alignment data (`alignment` field with AlignmentItem list)

**AlignmentConfig** (`config/alignment.py`):
- Model selection and device configuration
- `strategy`: "entire" (single-pass) vs "transcription"/"subtitle" (segmented)
- `trust_caption_timestamps`: Whether to segment based on input timestamps

### Data Flow

**Standard Alignment Flow:**
```
Input Media → AudioLoader → Lattice1Aligner → Caption (with alignments)
                                  ↓
Input Caption → Reader → supervisions → Tokenizer → align() → supervisions
                                                                   ↓
                                                          (Optional) Diarizer
```

**Transcription + Alignment Flow:**
```
Input Media → AudioLoader → Transcriber → supervisions
                                  ↓                ↓
                         Lattice1Aligner ← align() → Caption
```

**YouTube Flow:**
```
YouTube URL → Downloader → Media File + Auto Caption
                                  ↓
                           AudioLoader → Aligner → Caption
```

### Critical Implementation Details

1. **Alignment Strategies**:
   - `entire`: Single-pass alignment (default for short audio)
   - `transcription`: Segment based on transcription timestamps (requires ASR)
   - `subtitle`: Segment based on input caption timestamps (respects user timing)

2. **Streaming Mode** (`media.streaming_chunk_secs`):
   - Enables processing of long audio (up to 20 hours)
   - Default chunk size: 600 seconds (10 minutes)
   - Preserves alignment accuracy with minimal memory overhead

3. **Sentence Splitting** (`caption.split_sentence`):
   - Uses `wtpsplit` for intelligent sentence segmentation
   - Separates non-speech elements (`[MUSIC]`, `[APPLAUSE]`) from dialogue
   - Splits long segments into natural sentence boundaries

4. **Word-Level Output** (`caption.word_level`):
   - Returns `AlignmentItem` objects with per-word timestamps
   - Stored in `Supervision.alignment` field
   - Best exported to JSON or TextGrid formats

5. **Format Detection**:
   - Auto-detects input format from file extension/content
   - Format handlers in `caption/formats/` registry pattern
   - Each format has `Reader` and `Writer` classes

6. **Speaker Label Preservation**:
   - Diarizer recognizes speaker labels from input: `[Alice]`, `>> Bob:`, `SPEAKER_01:`
   - Matches detected speakers with existing labels
   - Gemini transcription can extract speaker names from context

## Testing Conventions

- Test data in `tests/data/`
- Test structure mirrors `src/lattifai/` structure
- Caption format tests in `tests/caption/test_formats.py`
- Config tests in `tests/config/`
- Use `pytest` fixtures for shared test data
- Coverage target: Include tests for new features

## Common Patterns

### Creating a New Caption Format Handler

1. Implement `Reader` and `Writer` in `src/lattifai/caption/formats/your_format.py`
2. Inherit from `BaseReader` and `BaseWriter`
3. Register format in `formats/__init__.py` `FORMAT_HANDLERS` dict
4. Add format extension to `SUPPORTED_FORMATS`
5. Add tests in `tests/caption/test_formats.py`

### Adding a New Transcription Model

1. Check if model fits Gemini pattern (API) or local pattern (HuggingFace)
2. For API models: Extend `GeminiTranscriber` or create new API client
3. For local models: Use `LattifAITranscriber` (supports any HF ASR model)
4. Update `create_transcriber()` factory logic if needed
5. Add model to README supported models table

### Extending Configuration

1. All config classes inherit from `nemo_run.Config`
2. Add new fields with type annotations and default values
3. Update corresponding component to consume config
4. Update CLI argument parsing in relevant `cli/*.py` file
5. Document in README and update examples

## Environment Variables

- `LATTIFAI_API_KEY`: LattifAI API key (required for alignment)
- `GEMINI_API_KEY`: Google Gemini API key (required for Gemini transcription)
- `TOKENIZERS_PARALLELISM=false`: Suppress tokenizers warning (set in `__init__.py`)

## Dependencies

**Core:**
- `lattifai-core`: Core API client
- `k2py`: K2 bindings for lattice operations
- `lhotse`: Audio data handling
- `nemo_run`: Configuration and CLI system

**Alignment:**
- `onnxruntime`: Lattice-1 model inference
- `wtpsplit`: Sentence splitting

**Transcription:**
- `OmniSenseVoice`: SenseVoice model wrapper
- `nemo_toolkit_asr`: NVIDIA Parakeet models
- `google-genai`: Gemini API

**Diarization:**
- `pyannote-audio-notorchdeps`: Speaker diarization

**Format Support:**
- `pysubs2`: SRT, VTT, ASS, SSA formats
- `praatio`: TextGrid format
- `tgt`: Alternative TextGrid support

## Performance Considerations

- Use `device="cuda"` for GPU acceleration (10x faster than CPU)
- Use `device="mps"` for Apple Silicon GPU
- Enable `streaming_chunk_secs` for audio >10 minutes
- Word-level alignment increases output size significantly
- Diarization adds ~10-30% processing time

## Key Conventions

- Line length: 120 characters (enforced by black/flake8)
- Import style: `isort` with `--profile=black`
- Type hints: Use for public APIs, optional for internal helpers
- Docstrings: Google style for public classes/methods
- Error handling: Custom exceptions in `errors.py`, raise specific errors with context
