"""Caption CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from lhotse.utils import Pathlike
from typing_extensions import Annotated

from lattifai.config import CaptionConfig


@run.cli.entrypoint(name="convert", namespace="caption")
def convert(
    input_path: Pathlike,
    output_path: Pathlike,
    include_speaker_in_text: bool = True,
    normalize_text: bool = False,
):
    """
    Convert caption file to another format.

    This command reads a caption file from one format and writes it to another format,
    preserving all timing information, text content, and speaker labels (if present).
    Supports common caption formats including SRT, VTT, JSON, and Praat TextGrid.

    Shortcut: invoking ``laisub-convert`` is equivalent to running ``lai caption convert``.

    Args:
        input_path: Path to input caption file (supports SRT, VTT, JSON, TextGrid formats)
        output_path: Path to output caption file (format determined by file extension)
        include_speaker_in_text: Preserve speaker labels in caption text content.
        normalize_text: Whether to normalize caption text during conversion.
            This applies text cleaning such as removing HTML tags, decoding entities,
            collapsing whitespace, and standardizing punctuation.

    Examples:
        # Basic format conversion (positional arguments)
        lai caption convert input.srt output.vtt

        # Convert with text normalization
        lai caption convert input.srt output.json normalize_text=true

        # Mixing positional and keyword arguments
        lai caption convert input.srt output.vtt \\
            include_speaker_in_text=false \\
            normalize_text=true

        # Using keyword arguments (traditional syntax)
        lai caption convert \\
            input_path=input.srt \\
            output_path=output.TextGrid
    """
    from lattifai.caption import Caption

    caption = Caption.read(input_path, normalize_text=normalize_text)
    caption.write(output_path, include_speaker_in_text=include_speaker_in_text)

    print(f"✅ Converted {input_path} -> {output_path}")
    return output_path


@run.cli.entrypoint(name="normalize", namespace="caption")
def normalize(
    input_path: Pathlike,
    output_path: Pathlike,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
):
    """
    Normalize caption text by cleaning HTML entities and whitespace.

    This command reads a caption file and normalizes all text content by applying
    the following transformations:
    - Decode common HTML entities (&amp;, &lt;, &gt;, &quot;, &#39;, &nbsp;)
    - Remove HTML tags (e.g., <i>, <font>, <b>, <br>)
    - Collapse multiple whitespace characters into single spaces
    - Convert curly apostrophes to straight ones in contractions
    - Strip leading and trailing whitespace from each segment

    Shortcut: invoking ``laisub-normalize`` is equivalent to running ``lai caption normalize``.

    Args:
        input_path: Path to input caption file to normalize
        output_path: Path to output caption file (defaults to overwriting input file)
        caption: Caption configuration for text normalization.
            Fields: input_format, output_format, normalize_text (automatically enabled),
                    encoding

    Examples:
        # Normalize and save to new file (positional arguments)
        lai caption normalize input.srt output.srt

        # Normalize with format conversion
        lai caption normalize input.vtt output.srt

        # Normalize with custom caption config
        lai caption normalize input.srt output.srt \\
            caption.encoding=utf-8

        # Using keyword arguments (traditional syntax)
        lai caption normalize \\
            input_path=input.srt \\
            output_path=output.srt
    """
    from pathlib import Path

    from lattifai.caption import Caption

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    caption_obj = Caption.read(input_path, normalize_text=True)
    caption_obj.write(output_path, include_speaker_in_text=True)

    if output_path == input_path:
        print(f"✅ Normalized {input_path} (in-place)")
    else:
        print(f"✅ Normalized {input_path} -> {output_path}")

    return output_path


@run.cli.entrypoint(name="shift", namespace="caption")
def shift(
    input_path: Pathlike,
    output_path: Pathlike,
    seconds: float,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
):
    """
    Shift caption timestamps by a specified number of seconds.

    This command reads a caption file and adjusts all timestamps by adding or
    subtracting a specified offset. Use positive values to delay captions and
    negative values to make them appear earlier.

    Shortcut: invoking ``laisub-shift`` is equivalent to running ``lai caption shift``.

    Args:
        input_path: Path to input caption file
        output_path: Path to output caption file (can be same as input for in-place modification)
        seconds: Number of seconds to shift timestamps. Positive values delay captions,
                 negative values advance them earlier.
        caption: Caption configuration for reading/writing.
            Fields: input_format, output_format, encoding

    Examples:
        # Delay captions by 2 seconds (positional arguments)
        lai caption shift input.srt output.srt 2.0

        # Make captions appear 1.5 seconds earlier
        lai caption shift input.srt output.srt -1.5

        # Shift and convert format
        lai caption shift input.vtt output.srt seconds=0.5

        # Using keyword arguments (traditional syntax)
        lai caption shift \\
            input_path=input.srt \\
            output_path=output.srt \\
            seconds=3.0
    """
    from pathlib import Path

    from lattifai.caption import Caption

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    # Read captions
    caption_obj = Caption.read(input_path)

    # Shift timestamps
    shifted_caption = caption_obj.shift_time(seconds)

    # Write shifted captions
    shifted_caption.write(output_path, include_speaker_in_text=True)

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
