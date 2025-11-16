"""Subtitle CLI entry point with nemo_run."""

from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.config import SubtitleConfig


@run.cli.entrypoint(name="convert", namespace="subtitle")
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


@run.cli.entrypoint(name="normalize", namespace="subtitle")
def normalize(
    input_path: Path,
    output_path: Path,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
):
    """
    Normalize subtitle text by cleaning HTML entities and whitespace.

    This command reads a subtitle file, normalizes all text content by:
    - Decoding common HTML entities (&amp;, &lt;, &gt;, &quot;, &#39;, &nbsp;)
    - Removing HTML tags (e.g., <i>, <font>, <b>, <br>)
    - Collapsing multiple whitespace into single spaces
    - Converting curly apostrophes to straight ones in contractions

    Args:
        input_path: Path to input subtitle file
        output_path: Path to output subtitle file (default: overwrite input file)
        subtitle: Subtitle configuration

    Examples:
        # Normalize subtitle in-place
        lai subtitle normalize input.srt

        # Normalize and save to new file
        lai subtitle normalize input.srt output.srt

        # Normalize with format conversion
        lai subtitle normalize input.vtt output.srt
    """
    from lattifai.subtitle import Subtitler

    # Create config with normalize_text enabled
    if subtitle is None:
        subtitle = SubtitleConfig(normalize_text=True)
    else:
        subtitle.normalize_text = True

    subtitler = Subtitler(config=subtitle)

    # Read with normalization enabled
    supervisions = subtitler.read(input_path.expanduser(), normalize_text=True)
    output_path = subtitler.write(supervisions, output_path.expanduser())

    if output_path == input_path:
        print(f"✅ Normalized {input_path} (in-place)")
    else:
        print(f"✅ Normalized {input_path} -> {output_path}")

    return output_path


def main():
    run.cli.main(convert, normalize)


if __name__ == "__main__":
    main()
