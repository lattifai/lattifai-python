# Workflow Prompts

This directory contains system prompts and instructions for various AI models used in workflows.

## Structure

- `gemini/` - Prompts for Gemini models
  - `transcription_gem.txt` - System prompt from the Audio Transcription Gem

## Usage

Prompts are loaded automatically by their respective workflow modules. To update a prompt:

1. Edit the corresponding `.txt` file
2. The changes will be picked up on the next workflow run
3. No code changes needed

## Adding New Prompts

1. Create a subdirectory for the model/service (e.g., `openai/`, `anthropic/`)
2. Add prompt files as `.txt` files with descriptive names
3. Update the loader in the corresponding workflow module
