"""Translation CLI entry point with nemo_run."""

import asyncio
from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.cli._shared import ensure_parent_dir, resolve_caption_paths, resolve_media_input, run_youtube_workflow
from lattifai.cli.entrypoint import LattifAIEntrypoint
from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    EventConfig,
    MediaConfig,
    TranscriptionConfig,
)
from lattifai.config.translation import TranslationConfig


def _should_continue_with_refined(translation_config: TranslationConfig) -> bool:
    """Decide whether to continue from normal mode to refined review."""
    import sys

    if translation_config.mode != "normal":
        return False
    if translation_config.auto_refine_after_normal:
        return True
    if not translation_config.ask_refine_after_normal:
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False

    try:
        answer = input("Normal translation complete. Continue with refined review? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _resolve_translation_output_path(
    *,
    input_path: Optional[Path],
    explicit_output: Optional[str],
    source_path: Optional[str],
    target_lang: str,
) -> Path:
    """Resolve the final translation output path."""
    from lattifai.translation.prompts import get_language_name

    if explicit_output:
        return Path(explicit_output)

    lang_name = get_language_name(target_lang)
    if input_path is not None:
        return input_path.with_name(f"{input_path.stem}_{lang_name}{input_path.suffix}")
    if source_path:
        src = Path(source_path)
        return src.with_name(f"{src.stem}_{lang_name}{src.suffix}")
    return Path(f"translated_{lang_name}.srt")


def _translate_caption_in_place(cap, translation_config: TranslationConfig):
    """Run translation and optional refined review in place."""
    from lattifai.theme import theme
    from lattifai.translation import create_translator
    from lattifai.translation.prompts import get_language_name
    from lattifai.utils import safe_print

    translator = create_translator(translation_config)
    lang_name = get_language_name(translation_config.target_lang)
    safe_print(
        theme.step(
            f"Translating {len(cap.supervisions)} segments to {lang_name} "
            f"[mode={translation_config.mode}, provider={translator.name}]"
        )
    )

    source_texts = [sup.text or "" for sup in cap.supervisions]
    asyncio.run(translator.translate_captions(cap.supervisions, translation_config))

    if _should_continue_with_refined(translation_config):
        safe_print(theme.step("Continuing with refined review pass..."))
        asyncio.run(
            translator.refine_existing_draft(
                cap.supervisions,
                translation_config,
                source_texts=source_texts,
            )
        )


@run.cli.entrypoint(name="caption", namespace="translate", entrypoint_cls=LattifAIEntrypoint)
def translate(
    input: Optional[str] = None,
    output: Optional[str] = None,
    translation: Annotated[Optional[TranslationConfig], run.Config[TranslationConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
):
    """
    Translate caption file to target language.

    Supports three modes:
    - quick: Direct batch translation (fast, informal use)
    - normal: Analyze content first, then translate with terminology consistency
    - refined: Analyze + translate + global review for publication quality

    Args:
        input: Path to input caption file (positional argument)
        output: Path for output caption file (positional argument)
        translation: Translation configuration.
            Fields: llm.model_name, llm.provider, llm.api_base_url, target_lang,
                    mode, bilingual, style, approach, batch_size, glossary_file,
                    save_artifacts, ask_refine_after_normal, auto_refine_after_normal
        caption: Caption I/O configuration.
            Fields: input_format, output_format

    Examples:
        # Quick mode
        lai translate caption input.srt output.srt translation.target_lang=zh translation.mode=quick

        # Normal mode (default) with bilingual output
        lai translate caption input.srt output.srt translation.bilingual=true

        # Refined mode with artifacts
        lai translate caption input.srt output.srt \\
            translation.mode=refined \\
            translation.save_artifacts=true \\
            translation.artifacts_dir=/tmp/artifacts

        # Using OpenAI-compatible API
        lai translate caption input.srt output.srt \\
            translation.llm.provider=openai \\
            translation.llm.api_base_url=http://localhost:8000/v1 \\
            translation.llm.model_name=qwen3

        # With custom glossary
        lai translate caption input.srt output.srt \\
            translation.glossary_file=glossary.yaml
    """
    from lattifai.theme import theme
    from lattifai.utils import safe_print

    translation_config = translation or TranslationConfig()
    caption_config = caption or CaptionConfig()

    if not input:
        raise ValueError("Input caption file is required.")

    input_path = Path(input)
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input}")

    from lattifai.caption import Caption

    safe_print(theme.step(f"Loading: {input_path}"))
    cap = Caption.read(str(input_path))

    if not cap.supervisions:
        raise ValueError(f"No caption segments found in: {input}")

    output_path = _resolve_translation_output_path(
        input_path=input_path,
        explicit_output=output,
        source_path=cap.source_path,
        target_lang=translation_config.target_lang,
    )

    # Always set artifacts_dir so checkpoints land next to output, not in cwd
    if not translation_config.artifacts_dir:
        translation_config.artifacts_dir = str(output_path.parent)

    _translate_caption_in_place(cap, translation_config)
    ensure_parent_dir(output_path)
    cap.write(str(output_path), translation_first=caption_config.translation_first)

    safe_print(theme.ok(f"Translation saved: {output_path}"))

    return cap


@run.cli.entrypoint(name="youtube", namespace="translate", entrypoint_cls=LattifAIEntrypoint)
def translate_youtube(
    yt_url: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
    event: Annotated[Optional[EventConfig], run.Config[EventConfig]] = None,
    translation: Annotated[Optional[TranslationConfig], run.Config[TranslationConfig]] = None,
    use_transcription: bool = False,
):
    """
    Download YouTube video, align captions, and translate in one step.

    Combines the youtube align workflow with caption translation.
    Downloads media, transcribes/downloads captions, performs forced alignment,
    then translates the aligned captions to the target language.

    Args:
        yt_url: YouTube video URL or video ID (positional argument)
        media: Media configuration.
            Fields: output_dir, output_format, force_overwrite, audio_track_id, quality
        client: API client configuration.
        alignment: Alignment configuration.
            Fields: model_name, device, batch_size
        caption: Caption I/O configuration.
            Fields: output_format, output_path, split_sentence, word_level
        transcription: Transcription configuration.
            Fields: gemini_api_key, model_name, language
        diarization: Speaker diarization configuration.
        event: Event tracking configuration.
        translation: Translation configuration.
            Fields: target_lang, mode, bilingual, llm.provider, llm.model_name, glossary_file
        use_transcription: Skip YouTube caption download and transcribe directly.

    Examples:
        # Download + align + translate to Chinese
        lai translate youtube "https://www.youtube.com/watch?v=VIDEO_ID" \\
            translation.target_lang=zh

        # With word-level alignment and bilingual Japanese output
        lai translate youtube "dQw4w9WgXcQ" \\
            translation.target_lang=ja \\
            translation.bilingual=true \\
            caption.word_level=true

        # Refined translation with custom glossary
        lai translate youtube "VIDEO_ID" \\
            translation.target_lang=zh \\
            translation.mode=refined \\
            translation.glossary_file=glossary.yaml

        # Using OpenAI-compatible API for translation
        lai translate youtube "VIDEO_ID" \\
            translation.target_lang=ko \\
            translation.llm.provider=openai \\
            translation.llm.api_base_url=http://localhost:8000/v1
    """
    from lattifai.theme import theme
    from lattifai.utils import safe_print

    media_config = resolve_media_input(
        media,
        yt_url,
        positional_name="yt_url",
        required_message="YouTube URL is required.",
    )
    caption_config = resolve_caption_paths(caption)
    translation_config = translation or TranslationConfig()

    cap = run_youtube_workflow(
        media=media_config,
        caption=caption_config,
        client=client,
        alignment=alignment,
        transcription=transcription,
        diarization=diarization,
        event=event,
        use_transcription=use_transcription,
    )

    if not cap or not cap.supervisions:
        raise RuntimeError("YouTube alignment produced no caption segments.")

    _translate_caption_in_place(cap, translation_config)
    output_path = _resolve_translation_output_path(
        input_path=None,
        explicit_output=caption_config.output_path,
        source_path=cap.source_path,
        target_lang=translation_config.target_lang,
    )
    ensure_parent_dir(output_path)
    cap.write(str(output_path), translation_first=caption_config.translation_first)

    safe_print(theme.ok(f"🎉 Translation saved: {output_path}"))
    return cap


def main():
    """Entry point for lai-translate command."""
    run.cli.main(translate)


if __name__ == "__main__":
    main()
