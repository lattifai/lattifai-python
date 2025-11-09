"""Subtitle CLI entry point with nemo_run."""

from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.config import SubtitleConfig


@run.cli.entrypoint(name="convert", namespace="lai")
def convert(
    input_path: Path,
    output_path: Path,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
):
    """
    Convert subtitle file to another format.

    Args:
        input_path: Path to input subtitle file
        output_path: Path to output subtitle file
        subtitle: Subtitle configuration

    Examples:
        # Basic format conversion
        lai subtitle convert input.srt output.vtt

        # Convert to TextGrid with speaker info
        lai subtitle convert input.srt output.TextGrid \\
            --subtitle.include-speaker-in-text=true

        # Convert without speaker info
        lai subtitle convert input.srt output.vtt \\
            --subtitle.include-speaker-in-text=false
    """
    from lattifai.subtitle import Subtitler

    subtitler = Subtitler(config=subtitle)

    supervisions = subtitler.read(input_path)
    output_path = subtitler.write(supervisions, output_path)

    print(f"✅ Converted {input_path} -> {output_path}")
    return output_path


def main():
    run.cli.main(convert)


if __name__ == "__main__":
    main()
