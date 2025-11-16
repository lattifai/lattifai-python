"""Agent workflow CLI entry point with nemo_run."""

import asyncio
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.client import AsyncLattifAI
from lattifai.config import (
    AlignmentConfig,
    ClientConfig,
    MediaConfig,
    SubtitleConfig,
    TranscriptionConfig,
)
from lattifai.transcription.gemini import GeminiTranscriber
from lattifai.workflow.youtube import YouTubeDownloader, YouTubeSubtitleAgent


def _build_async_client(
    client_config: ClientConfig,
    alignment_config: AlignmentConfig,
    subtitle_config: SubtitleConfig,
) -> AsyncLattifAI:
    return AsyncLattifAI(
        client_config=client_config,
        alignment_config=alignment_config,
        subtitle_config=subtitle_config,
    )


def _create_agent(
    transcription_config: TranscriptionConfig,
    client_config: ClientConfig,
    alignment_config: AlignmentConfig,
    subtitle_config: SubtitleConfig,
    max_retries: int,
) -> YouTubeSubtitleAgent:
    downloader = YouTubeDownloader()
    transcriber = GeminiTranscriber(transcription_config=transcription_config)
    aligner = _build_async_client(client_config, alignment_config, subtitle_config)
    return YouTubeSubtitleAgent(
        downloader=downloader,
        transcriber=transcriber,
        aligner=aligner,
        max_retries=max_retries,
    )


@run.cli.entrypoint(name="agent", namespace="agent")
def agent(
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    max_retries: int = 0,
):
    """
    Run the agentic YouTube workflow (download → transcribe → align → export).

    This workflow performs a complete end-to-end processing pipeline:
    1. Download media from YouTube URL
    2. Transcribe audio using Gemini API
    3. Align transcription with audio
    4. Export results in the desired format

    Args:
        media: Media configuration for YouTube download and output handling.
            Fields: input_path (YouTube URL), media_format, sample_rate, channels,
                    output_dir, output_path, output_format, prefer_audio,
                    default_audio_format, default_video_format, force_overwrite
        client: API client configuration.
            Fields: api_key, base_url, timeout, max_retries, default_headers
        alignment: Alignment configuration (model selection and inference settings).
            Fields: model_name_or_path, device, batch_size
        subtitle: Subtitle I/O configuration (file reading/writing and formatting).
            Fields: input_format, input_path, output_format, output_path,
                    normalize_text, split_sentence, word_level,
                    include_speaker_in_text, encoding
        transcription: Transcription configuration for Gemini API.
            Fields: api_key, model_name, prompt_text, enable_diarization, language
        max_retries: Maximum number of retries for failed operations (default: 0)

    Examples:
        # Basic YouTube video processing
        lai agent workflow media.input_path="https://youtu.be/VIDEO_ID" \\
                          transcription.api_key=YOUR_GEMINI_KEY

        # Download as audio only with word-level alignment
        lai agent workflow media.input_path="https://youtu.be/VIDEO_ID" \\
                          --media.prefer-audio=true \\
                          --subtitle.word-level=true \\
                          transcription.api_key=YOUR_GEMINI_KEY

        # Enable speaker diarization and sentence splitting
        lai agent workflow media.input_path="https://youtu.be/VIDEO_ID" \\
                          --transcription.enable-diarization=true \\
                          subtitle.split_sentence=true \\
                          transcription.api_key=YOUR_GEMINI_KEY

        # Full configuration with custom output directory
        lai agent workflow \\
            media.input_path="https://youtu.be/VIDEO_ID" \\
            media.output_dir=/tmp/youtube \\
            --media.output-format=wav \\
            --subtitle.output-format=json \\
            --subtitle.word-level=true \\
            subtitle.split_sentence=true \\
            --alignment.device=cuda \\
            --alignment.model-name-or-path=Lattifai/Lattice-1-Alpha \\
            transcription.api_key=YOUR_GEMINI_KEY \\
            --transcription.enable-diarization=true \\
            max_retries=3
    """

    media_config = media or MediaConfig()
    subtitle_config = subtitle or SubtitleConfig()
    transcription_config = transcription or TranscriptionConfig()
    client_config = client or ClientConfig()
    alignment_config = alignment or AlignmentConfig()

    # Normalize media preferences before launching the workflow
    media_format = media_config.normalize_format()
    output_dir = media_config.output_dir

    workflow_agent = _create_agent(
        transcription_config=transcription_config,
        client_config=client_config,
        alignment_config=alignment_config,
        subtitle_config=subtitle_config,
        max_retries=max_retries,
    )

    workflow_result = asyncio.run(
        workflow_agent.execute(
            url=media_config.input_path,
            output_dir=str(output_dir),
            media_format=media_format,
            force_overwrite=media_config.force_overwrite,
            output_format=subtitle_config.output_format,
            split_sentence=subtitle_config.split_sentence,
            word_level=subtitle_config.word_level,
            include_speaker_in_text=subtitle_config.include_speaker_in_text,
        )
    )

    if not workflow_result.is_success:
        if workflow_result.exception:
            raise workflow_result.exception
        raise RuntimeError(workflow_result.error or "Workflow execution failed")

    export_payload = workflow_result.data.get("export_results_result", {})
    return {
        "media_format": media_format,
        "output_dir": str(output_dir),
        "results": export_payload,
    }


def main():
    run.cli.main(agent)


if __name__ == "__main__":
    main()
