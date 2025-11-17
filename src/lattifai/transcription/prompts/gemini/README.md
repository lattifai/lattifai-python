# Gemini Workflow Prompts

This directory contains system prompts for Gemini-based workflows.

## Files

### `transcription_gem.txt` [@dotey](https://x.com/dotey/status/1971810075867046131)

System prompt extracted from the Audio/Video Transcription Gem:
- **Gem URL**: https://gemini.google.com/gem/1870ly7xvW2hU_umtv-LedGsjywT0sQiN
- **Model**: Gemini 2.5 Pro
- **Purpose**: Audio/Video transcription with accuracy and natural formatting

## Updating the Prompt

To update the transcription behavior:

1. Edit `transcription_gem.txt` directly
2. Changes take effect on the next workflow run
3. No code changes needed

## Gem Information

The original Gem configuration is preserved in the `GeminiTranscriber.GEM_URL` constant and can be accessed via the `get_gem_info()` method.
