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


@run.cli.entrypoint(name="workflow", namespace="agent")
def agent(
    url: str,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    max_retries: int = 0,
):
    """Run the agentic YouTube workflow (download → transcribe → align → export)."""

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
            url=url,
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
