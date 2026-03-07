"""Translation CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.config import CaptionConfig
from lattifai.config.translation import TranslationConfig


@run.cli.entrypoint(name="run", namespace="translate")
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
                    batch_size, glossary_file, save_artifacts
        caption: Caption I/O configuration.
            Fields: input_format, output_format

    Examples:
        # Quick mode
        lai translate run input.srt output.srt translation.target_lang=zh translation.mode=quick

        # Normal mode (default) with bilingual output
        lai translate run input.srt output.srt translation.bilingual=true

        # Refined mode with artifacts
        lai translate run input.srt output.srt \\
            translation.mode=refined \\
            translation.save_artifacts=true \\
            translation.artifacts_dir=/tmp/artifacts

        # Using OpenAI-compatible API
        lai translate run input.srt output.srt \\
            translation.provider=openai \\
            translation.api_base_url=http://localhost:8000/v1 \\
            translation.model_name=qwen3

        # With custom glossary
        lai translate run input.srt output.srt \\
            translation.glossary_file=glossary.yaml
    """
    import asyncio
    from pathlib import Path

    import colorful

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
    from lattifai.caption import Caption, GeminiReader

    safe_print(colorful.cyan(f"Loading: {input_path}"))

    if input_path.suffix.lower() == ".md":
        supervisions = GeminiReader.extract_for_alignment(str(input_path))
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
        colorful.cyan(
            f"Translating {len(cap.supervisions)} segments to {lang_name} " f"[mode={mode}, provider={translator.name}]"
        )
    )

    # Run translation
    asyncio.run(translator.translate_captions(cap.supervisions, translation_config))

    # Write output
    from lattifai.caption import GeminiWriter

    if output_path.suffix.lower() == ".md":
        GeminiWriter.write(cap.supervisions, str(output_path))
    else:
        cap.write(str(output_path))

    safe_print(colorful.green(f"Translation saved: {output_path}"))

    return cap


def main():
    """Entry point for lai-translate command."""
    run.cli.main(translate)


if __name__ == "__main__":
    main()
