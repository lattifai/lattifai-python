"""
Subtitle Agents

An agentic workflow for processing YouTube(or more) videos through:
1. URL processing and audio download
2. Gemini 2.5 Pro transcription
3. LattifAI alignment
"""

from .youtube import YouTubeSubtitleAgent

__all__ = ["YouTubeSubtitleAgent"]
