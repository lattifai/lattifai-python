"""Translation CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

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


@run.cli.entrypoint(name="caption", namespace="translate")
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
            Fields: model_name, provider, target_lang, mode, bilingual, style,
                    batch_size, glossary_file, save_artifacts,
                    ask_refine_after_normal, auto_refine_after_normal
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
            translation.provider=openai \\
            translation.api_base_url=http://localhost:8000/v1 \\
            translation.model_name=qwen3

        # With custom glossary
        lai translate caption input.srt output.srt \\
            translation.glossary_file=glossary.yaml
    """
    import asyncio
    from pathlib import Path

    from lattifai.theme import theme
    from lattifai.translation import create_translator
    from lattifai.translation.prompts import get_language_name
    from lattifai.utils import safe_print

    # Initialize config
    translation_config = translation or TranslationConfig()

    # Validate input
    if not input:
        raise ValueError("Input caption file is required.")

    input_path = Path(input)
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input}")

    # Load caption
    from lattifai.caption import Caption, MarkdownReader

    safe_print(theme.step(f"Loading: {input_path}"))

    if input_path.suffix.lower() == ".md":
        supervisions = MarkdownReader.extract_for_alignment(str(input_path))
        cap = Caption.from_supervisions(supervisions)
    else:
        cap = Caption.read(str(input_path))

    if not cap.supervisions:
        raise ValueError(f"No caption segments found in: {input}")

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        lang_name = get_language_name(translation_config.target_lang)
        suffix = input_path.suffix
        output_path = input_path.with_name(f"{input_path.stem}_{lang_name}{suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set artifacts dir if saving
    if translation_config.save_artifacts and not translation_config.artifacts_dir:
        translation_config.artifacts_dir = str(output_path.parent)

    # Create translator
    translator = create_translator(translation_config)

    lang_name = get_language_name(translation_config.target_lang)
    mode = translation_config.mode
    safe_print(
        theme.step(
            f"Translating {len(cap.supervisions)} segments to {lang_name} " f"[mode={mode}, provider={translator.name}]"
        )
    )

    source_texts = [sup.text or "" for sup in cap.supervisions]

    # Run translation (quick / normal / refined)
    asyncio.run(translator.translate_captions(cap.supervisions, translation_config))

    # Progressive upgrade: normal -> refined without retranslating draft
    if _should_continue_with_refined(translation_config):
        safe_print(theme.step("Continuing with refined review pass..."))
        asyncio.run(
            translator.refine_existing_draft(
                cap.supervisions,
                translation_config,
                source_texts=source_texts,
            )
        )

    # Write output
    from lattifai.caption import MarkdownWriter

    caption_config = caption or CaptionConfig()

    if output_path.suffix.lower() == ".md":
        MarkdownWriter.write(cap.supervisions, str(output_path))
    else:
        cap.write(str(output_path), translation_first=caption_config.translation_first)

    safe_print(theme.ok(f"Translation saved: {output_path}"))

    return cap


@run.cli.entrypoint(name="youtube", namespace="translate")
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

    Combines the youtube alignment workflow with caption translation.
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
            Fields: target_lang, mode, bilingual, provider, model_name, glossary_file
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
            translation.provider=openai \\
            translation.api_base_url=http://localhost:8000/v1
    """
    import asyncio
    from pathlib import Path

    from lattifai.client import LattifAI
    from lattifai.theme import theme
    from lattifai.translation import create_translator
    from lattifai.translation.prompts import get_language_name
    from lattifai.utils import safe_print

    # Initialize configs
    media_config = media or MediaConfig()
    caption_config = caption or CaptionConfig()
    translation_config = translation or TranslationConfig()

    # Validate URL
    if yt_url and media_config.input_path:
        raise ValueError("Cannot specify both positional yt_url and media.input_path.")
    if not yt_url and not media_config.input_path:
        raise ValueError("YouTube URL is required.")
    if yt_url:
        media_config.set_input_path(yt_url)

    # Step 1: YouTube workflow (download + transcribe + align)
    lattifai_client = LattifAI(
        client_config=client,
        alignment_config=alignment,
        caption_config=caption_config,
        transcription_config=transcription,
        diarization_config=diarization,
        event_config=event,
    )

    cap = lattifai_client.youtube(
        url=media_config.input_path,
        output_dir=media_config.output_dir,
        output_caption_path=caption_config.output_path,
        media_format=media_config.normalize_format() if media_config.output_format else None,
        force_overwrite=media_config.force_overwrite,
        split_sentence=caption_config.split_sentence,
        channel_selector=media_config.channel_selector,
        streaming_chunk_secs=media_config.streaming_chunk_secs,
        use_transcription=use_transcription,
        audio_track_id=media_config.audio_track_id,
        quality=media_config.quality,
    )

    if not cap or not cap.supervisions:
        raise RuntimeError("YouTube alignment produced no caption segments.")

    # Step 2: Translate the aligned captions
    lang_name = get_language_name(translation_config.target_lang)
    safe_print(
        theme.step(
            f"🌐 Translating {len(cap.supervisions)} segments to {lang_name} "
            f"[mode={translation_config.mode}, provider={translation_config.provider}]"
        )
    )

    translator = create_translator(translation_config)
    source_texts = [sup.text or "" for sup in cap.supervisions]
    asyncio.run(translator.translate_captions(cap.supervisions, translation_config))

    # Progressive upgrade: normal -> refined
    if _should_continue_with_refined(translation_config):
        safe_print(theme.step("Continuing with refined review pass..."))
        asyncio.run(
            translator.refine_existing_draft(
                cap.supervisions,
                translation_config,
                source_texts=source_texts,
            )
        )

    # Step 3: Write translated output
    # Determine output path: use caption.output_path or generate from aligned output
    if caption_config.output_path:
        output_path = Path(caption_config.output_path)
    elif cap.source_path:
        src = Path(cap.source_path)
        output_path = src.with_name(f"{src.stem}_{lang_name}{src.suffix}")
    else:
        output_path = Path(f"translated_{lang_name}.srt")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from lattifai.caption import MarkdownWriter

    if output_path.suffix.lower() == ".md":
        MarkdownWriter.write(cap.supervisions, str(output_path))
    else:
        cap.write(str(output_path), translation_first=caption_config.translation_first)

    safe_print(theme.ok(f"🎉 Translation saved: {output_path}"))
    return cap


def main():
    """Entry point for lai-translate command."""
    run.cli.main(translate)


if __name__ == "__main__":
    main()
