lai-transcribe -Y tests/data/SA1.wav transcription.model_name=gemini-3-pro-preview -v
lai transcribe align -Y -v tests/data/SA1.wav transcription.model_name=nvidia/parakeet-tdt-0.6b-v3

uv run lai-transcribe -Y tests/data/SA1.wav transcription.model_name=gemini-3-pro-preview -v
uv run lai transcribe align -Y -v tests/data/SA1.wav transcription.model_name=gemini-3-flash-preview
