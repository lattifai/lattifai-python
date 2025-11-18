"""Subtitle CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from lhotse.utils import Pathlike
from typing_extensions import Annotated

from lattifai.config import SubtitleConfig


@run.cli.entrypoint(name="convert", namespace="subtitle")
def convert(
    input_path: Pathlike,
    output_path: Pathlike,
    include_speaker_in_text: bool = True,
    normalize_text: bool = False,
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
        include_speaker_in_text: Preserve speaker labels in subtitle text content.
        normalize_text: Whether to normalize subtitle text during conversion.
            This applies text cleaning such as removing HTML tags, decoding entities,
            collapsing whitespace, and standardizing punctuation.

    Examples:
        # Basic format conversion (positional arguments)
        lai subtitle convert input.srt output.vtt

        # Convert with text normalization
        lai subtitle convert input.srt output.json normalize_text=true

        # Mixing positional and keyword arguments
        lai subtitle convert input.srt output.vtt \\
            include_speaker_in_text=false \\
            normalize_text=true

        # Using keyword arguments (traditional syntax)
        lai subtitle convert \\
            input_path=input.srt \\
            output_path=output.TextGrid
    """
    from lattifai.subtitle import Subtitler

    subtitle = SubtitleConfig(
        include_speaker_in_text=include_speaker_in_text,
        normalize_text=normalize_text,
    )

    subtitler = Subtitler(config=subtitle)

    supervisions = subtitler.read(input_path)
    output_path = subtitler.write(supervisions, output_path)

    print(f"✅ Converted {input_path} -> {output_path}")
    return output_path


@run.cli.entrypoint(name="normalize", namespace="subtitle")
def normalize(
    input_path: Pathlike,
    output_path: Pathlike,
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
        # Normalize and save to new file (positional arguments)
        lai subtitle normalize input.srt output.srt

        # Normalize with format conversion
        lai subtitle normalize input.vtt output.srt

        # Normalize with custom subtitle config
        lai subtitle normalize input.srt output.srt \\
            subtitle.encoding=utf-8

        # Using keyword arguments (traditional syntax)
        lai subtitle normalize \\
            input_path=input.srt \\
            output_path=output.srt
    """
    from lattifai.subtitle import Subtitler

    # Create config with normalize_text enabled
    if subtitle is None:
        subtitle = SubtitleConfig(normalize_text=True)
    else:
        subtitle.normalize_text = True

    subtitler = Subtitler(config=subtitle)

    # Read with normalization enabled
    from pathlib import Path

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    supervisions = subtitler.read(input_path, normalize_text=True)
    output_path = subtitler.write(supervisions, output_path)

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
