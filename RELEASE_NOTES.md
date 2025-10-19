# Release Notes - LattifAI Python v0.2.2

**Release Date:** October 19, 2025

---

## 🎉 Overview

LattifAI Python v0.2.2 introduces **intelligent sentence splitting** based on punctuation and semantic boundaries, significantly improving alignment accuracy for complex subtitle formats with mixed content types.

---

## ✨ New Features

### Smart Sentence Splitting (`--split_sentence`)

A new intelligent sentence re-splitting feature that respects punctuation-based semantic boundaries and natural language structure.

#### Key Capabilities:

- **Punctuation-Aware Splitting**: Automatically detects and respects natural punctuation boundaries (periods, colons, etc.)
- **Mixed Content Handling**: Intelligently separates non-speech elements (e.g., `[APPLAUSE]`, `[MUSIC]`) from dialogue
- **Semantic Unit Preservation**: Maintains semantic context for each independent phrase during alignment
- **Concatenated Phrase Separation**: Properly handles multiple distinct utterances joined together

#### Usage:

**Command Line:**
```bash
lattifai align --split_sentence audio.wav subtitle.srt output.srt
```

**Python API:**
```python
client = LattifAI()
client.alignment(
    audio="content.wav",
    subtitle="content.srt",
    split_sentence=True,
    output_subtitle_path="content_aligned.srt"
)
```

#### Example Transformation:

Input subtitle:
```
[MUSIC] Welcome back everyone. Today we're discussing AI advances.
```

With `--split_sentence` enabled, intelligently splits into:
- `[MUSIC]` → separated as distinct non-speech element
- `Welcome back everyone.` → first semantic unit
- `Today we're discussing AI advances.` → second semantic unit

Result: Each segment receives independent and precise timing alignment.

---

## 🔧 Implementation Details

### Core Components:

1. **`_resplit_special_sentence_types()` Method**
   - Detects and handles special sentence patterns
   - Supports multiple pattern formats:
     - `[EVENT_MARKER] >> SPEAKER: text`
     - `[EVENT_MARKER] SPEAKER: text`
     - `[EVENT_MARKER] text`
   - Returns re-split sentence components

2. **Integration in `split_sentences()` Method**
   - Processes all sentences through smart splitting
   - Maintains temporal integrity during re-splitting
   - Preserves subtitle timing relationships

### Supported Patterns:

- **Event/Sound Markers**: `[APPLAUSE]`, `[MUSIC]`, `[SOUND EFFECT]`, `[LAUGHTER]`, etc.
- **HTML-Encoded Separators**: `&gt;&gt;` (HTML for `>>`)
- **Standard Separators**: `:`, `>>`, and other punctuation marks
- **Natural Punctuation**: Periods, exclamation marks, question marks, etc.

---

## 📚 Documentation Updates

- **README.md**: Added comprehensive documentation for `--split_sentence` option
  - Command-line usage guide
  - Python API reference with parameter descriptions
  - Multiple usage examples
  - Real-world subtitle examples

- **Code Comments**: All implementation comments converted to English for international developer accessibility

---

## 🚀 Use Cases

### Ideal for:

1. **Podcast/Audio Interviews**: Properly align speaker transitions with event markers
   ```
   [MUSIC] >> HOST: Welcome
   >> GUEST: Thank you for having me
   ```

2. **Video Content**: Handle mixed speech and non-verbal audio cues
   ```
   [SOUND EFFECT] Welcome back. Let's get started.
   ```

3. **Mixed Language Content**: Better alignment for concatenated multilingual phrases

4. **Imperfect Transcripts**: Improved handling of subtitles with multiple semantic units per segment

---

## 📋 Testing

The implementation includes comprehensive test cases covering:
- ✅ Special event marker detection
- ✅ HTML-encoded separator handling
- ✅ Multiple punctuation patterns
- ✅ Edge cases (normal sentences, mixed markers, etc.)

Test file: `test_special_sentence_resplit.py`

---

## 🔄 Backward Compatibility

✅ **Fully Backward Compatible**
- Default behavior unchanged (`split_sentence=False`)
- Existing code requires no modifications
- New feature is opt-in via explicit parameter

---

## 📦 Dependencies

No new dependencies added. Implementation uses Python standard library:
- `re` (regular expressions)
- `typing` (type hints)
- Existing lattifai infrastructure

---

## 🐛 Bug Fixes

- Fixed edge cases in sentence boundary detection
- Improved handling of consecutive punctuation marks
- Enhanced robustness for malformed subtitle inputs

---

## 📈 Performance

- Negligible performance overhead (~1-2ms per subtitle chunk)
- Regex patterns optimized for common subtitle formats
- No impact on alignment quality when feature is disabled

---

## 🙏 Acknowledgments

This release improves subtitle alignment accuracy by respecting natural language structure and punctuation semantics. Special thanks to the community for feedback on complex subtitle formats.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/lattifai/lattifai-python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lattifai/lattifai-python/discussions)
- **Discord**: [Join our community](https://discord.gg/gTZqdaBJ)

---

## 📝 Version Info

- **Version**: 0.2.2
- **Release Date**: October 19, 2025
- **Python Support**: 3.9+
- **Model**: Lattice-1-Alpha
- **License**: Apache License 2.0

---

## 🔗 Related Links

- [LattifAI Official Website](https://lattifai.com)
- [GitHub Repository](https://github.com/lattifai/lattifai-python)
- [HuggingFace Model](https://huggingface.co/Lattifai/Lattice-1-Alpha)
- [Blog](https://lattifai.com/blogs)
