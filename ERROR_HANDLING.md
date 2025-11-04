# LattifAI Error Handling System

This document describes the comprehensive error handling system implemented for the LattifAI Python SDK.

## Overview

The LattifAI error handling system provides:
- **Clear, actionable error messages** with context information
- **Direct support links** to GitHub issues and Discord community
- **Proper error hierarchy** for specific error handling
- **Helpful guidance** for common issues and solutions

## Error Hierarchy

```
LattifAIError (base)
├── AudioProcessingError
│   ├── AudioLoadError
│   └── AudioFormatError
├── SubtitleProcessingError
│   └── SubtitleParseError
├── AlignmentError
│   ├── LatticeEncodingError
│   └── LatticeDecodingError
├── ModelLoadError
├── DependencyError
├── APIError
└── ConfigurationError
```

## Error Classes

### Base Error: `LattifAIError`
The base exception class for all LattifAI-related errors.

**Features:**
- Includes error codes for categorization
- Provides context information
- Automatically includes support links in error messages

### Audio Processing Errors

#### `AudioLoadError`
Raised when audio files cannot be loaded or read.

**Common causes:**
- File not found
- Unsupported audio format
- File corruption
- Permission issues

#### `AudioFormatError`
Raised when there are issues with audio format or codec.

**Common causes:**
- Unsupported codec
- Invalid sample rate
- Channel configuration issues

### Text/Subtitle Processing Errors

#### `SubtitleProcessingError`
General error for subtitle/text processing operations.

#### `SubtitleParseError`
Raised when subtitle or text files cannot be parsed.

**Common causes:**
- Invalid file format
- Encoding issues
- Malformed subtitle files

### Alignment Errors

#### `AlignmentError`
General error during audio-text alignment process.

#### `LatticeEncodingError`
Raised when lattice graph generation fails.

**Common causes:**
- Invalid text content
- Tokenization issues
- API communication problems

#### `LatticeDecodingError`
Raised when lattice decoding fails.

**Common causes:**
- Invalid lattice structure
- Decoding algorithm issues
- Memory constraints

### System Errors

#### `ModelLoadError`
Raised when AI models fail to load.

**Common causes:**
- Missing model files
- Invalid model format
- Insufficient memory
- Hardware compatibility issues

#### `DependencyError`
Raised when required dependencies are missing.

**Common causes:**
- Missing k2 installation
- Missing audio processing libraries
- Version incompatibilities

#### `ConfigurationError`
Raised for client configuration issues.

**Common causes:**
- Missing API key
- Invalid configuration parameters
- Environment setup issues

#### `APIError`
Raised for API communication issues.

**Common causes:**
- Network connectivity
- Authentication failures
- API rate limits
- Server errors

## Usage Examples

### Basic Error Handling

```python
from lattifai import LattifAI
from lattifai.errors import AudioLoadError, ConfigurationError

try:
    client = LattifAI()
    result = client.alignment("audio.wav", "subtitle.srt")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    # Handle missing API key, etc.
except AudioLoadError as e:
    print(f"Audio loading failed: {e}")
    print(f"Audio path: {e.context['audio_path']}")
    # Handle audio file issues
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Specific Error Handling

```python
from lattifai.errors import (
    LatticeEncodingError,
    LatticeDecodingError,
    DependencyError
)

try:
    # Alignment process
    result = client.alignment("audio.wav", "text.txt")
except LatticeEncodingError as e:
    print(f"Text processing failed: {e}")
    print(f"Text preview: {e.context['text_preview']}")
    # Handle text-specific issues
except LatticeDecodingError as e:
    print(f"Alignment decoding failed: {e}")
    print(f"Lattice ID: {e.context['lattice_id']}")
    # Handle decoding issues
except DependencyError as e:
    print(f"Missing dependency: {e.context['dependency_name']}")
    if e.context['install_command']:
        print(f"Install with: {e.context['install_command']}")
    # Handle missing dependencies
```

## Support Information

All LattifAI errors automatically include support information:

### GitHub Issues
Users are directed to create GitHub issues at:
https://github.com/lattifai/lattifai-python/issues

**Include in issues:**
- Audio file format and duration
- Text/subtitle content being aligned
- Complete error message and stack trace
- Environment details (OS, Python version)

### Discord Community
Real-time support available at:
https://discord.gg/vzmTzzZgNu

## Error Context Information

Each error includes relevant context:

- **File paths** (audio, subtitle)
- **Text content** (truncated for privacy)
- **Technical details** (model names, lattice IDs)
- **System information** (device, format details)
- **Original error messages** (from underlying libraries)

## Best Practices

### For Users
1. **Read the full error message** - it contains helpful guidance
2. **Check the context information** - provides debugging details
3. **Use the support links** - GitHub issues and Discord community
4. **Include all requested information** when reporting issues

### For Developers
1. **Catch specific errors** rather than generic exceptions
2. **Use error context** for debugging and logging
3. **Provide fallback handling** for dependency errors
4. **Log errors appropriately** while preserving user privacy

## Testing

The error handling system includes comprehensive tests:

```bash
# Run error handling tests
python test_error_handling.py

# Run error demonstration
python demo_error_handling.py
```

## Error Codes

Each error includes a code for categorization:
- `AudioLoadError` → Error loading audio
- `ConfigurationError` → Configuration issues
- `LatticeEncodingError` → Text processing issues
- `DependencyError` → Missing dependencies
- etc.

## Integration

The error system is fully integrated into:
- **Client initialization** (`LattifAI` class)
- **Audio processing** (`Lattice1AlphaWorker`)
- **Text processing** (tokenization and lattice generation)
- **Alignment operations** (full pipeline)
- **I/O operations** (file reading/writing)

This comprehensive error handling ensures users get clear guidance when issues occur and can easily get help from the community or development team.
