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

    This command reads a subtitle file from one format and writes it to another format,
    preserving all timing information, text content, and speaker labels (if present).
    Supports common subtitle formats including SRT, VTT, JSON, and Praat TextGrid.

    Shortcut: invoking ``lai-subtitle-convert`` is equivalent to running ``lai subtitle convert``.

    Args:
        input_path: Path to input subtitle file (supports SRT, VTT, JSON, TextGrid formats)
        output_path: Path to output subtitle file (format determined by file extension)
        subtitle: Subtitle configuration for controlling text normalization and formatting.
            Fields: input_format, output_format, normalize_text, split_sentence,
                    word_level, include_speaker_in_text, encoding

    Examples:
        # Basic format conversion
        lai subtitle convert input.srt output.vtt

        # Convert to TextGrid with speaker info
        lai subtitle convert input.srt output.TextGrid \\
            --subtitle.include-speaker-in-text=true

        # Convert without speaker info
        lai subtitle convert input.srt output.vtt \\
            --subtitle.include-speaker-in-text=false

        # Convert with text normalization
        lai subtitle convert input.json output.srt \\
            --subtitle.normalize-text=true
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

    This command reads a subtitle file and normalizes all text content by applying
    the following transformations:
    - Decode common HTML entities (&amp;, &lt;, &gt;, &quot;, &#39;, &nbsp;)
    - Remove HTML tags (e.g., <i>, <font>, <b>, <br>)
    - Collapse multiple whitespace characters into single spaces
    - Convert curly apostrophes to straight ones in contractions
    - Strip leading and trailing whitespace from each segment

    Shortcut: invoking ``lai-subtitle-normalize`` is equivalent to running ``lai subtitle normalize``.

    Args:
        input_path: Path to input subtitle file to normalize
        output_path: Path to output subtitle file (defaults to overwriting input file)
        subtitle: Subtitle configuration for text normalization.
            Fields: input_format, output_format, normalize_text (automatically enabled),
                    encoding

    Examples:
        # Normalize subtitle in-place
        lai subtitle normalize input.srt

        # Normalize and save to new file
        lai subtitle normalize input.srt output.srt

        # Normalize with format conversion
        lai subtitle normalize input.vtt output.srt

        # Normalize and specify encoding
        lai subtitle normalize input.srt output.srt \\
            --subtitle.encoding=utf-8
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


def main_convert():
    run.cli.main(convert)


def main_normalize():
    run.cli.main(normalize)


if __name__ == "__main__":
    main_convert()
