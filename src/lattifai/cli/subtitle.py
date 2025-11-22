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

    Shortcut: invoking ``laisub-convert`` is equivalent to running ``lai subtitle convert``.

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

    Shortcut: invoking ``laisub-normalize`` is equivalent to running ``lai subtitle normalize``.

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


@run.cli.entrypoint(name="shift", namespace="subtitle")
def shift(
    input_path: Pathlike,
    output_path: Pathlike,
    seconds: float,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
):
    """
    Shift subtitle timestamps by a specified number of seconds.

    This command reads a subtitle file and adjusts all timestamps by adding or
    subtracting a specified offset. Use positive values to delay subtitles and
    negative values to make them appear earlier.

    Shortcut: invoking ``laisub-shift`` is equivalent to running ``lai subtitle shift``.

    Args:
        input_path: Path to input subtitle file
        output_path: Path to output subtitle file (can be same as input for in-place modification)
        seconds: Number of seconds to shift timestamps. Positive values delay subtitles,
                negative values make them appear earlier.
        subtitle: Subtitle configuration for reading/writing.
            Fields: input_format, output_format, encoding

    Examples:
        # Delay subtitles by 2 seconds (positional arguments)
        lai subtitle shift input.srt output.srt 2.0

        # Make subtitles appear 1.5 seconds earlier
        lai subtitle shift input.srt output.srt -1.5

        # Shift and convert format
        lai subtitle shift input.vtt output.srt seconds=0.5

        # Using keyword arguments (traditional syntax)
        lai subtitle shift \\
            input_path=input.srt \\
            output_path=output.srt \\
            seconds=3.0
    """
    from pathlib import Path

    from lattifai.subtitle import Subtitler

    if subtitle is None:
        subtitle = SubtitleConfig()

    subtitler = Subtitler(config=subtitle)

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    # Read subtitles
    supervisions = subtitler.read(input_path)

    # Shift timestamps
    for sup in supervisions:
        sup.start = max(0.0, sup.start + seconds)

    # Write shifted subtitles
    output_path = subtitler.write(supervisions, output_path)

    if seconds >= 0:
        direction = f"delayed by {seconds}s"
    else:
        direction = f"advanced by {abs(seconds)}s"

    if output_path == input_path:
        print(f"✅ Shifted timestamps {direction} in {input_path} (in-place)")
    else:
        print(f"✅ Shifted timestamps {direction}: {input_path} -> {output_path}")

    return output_path


def main_convert():
    run.cli.main(convert)


def main_normalize():
    run.cli.main(normalize)


def main_shift():
    run.cli.main(shift)


if __name__ == "__main__":
    main_convert()
