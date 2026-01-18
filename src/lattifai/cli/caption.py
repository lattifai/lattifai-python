"""Caption CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from lhotse.utils import Pathlike
from typing_extensions import Annotated

from lattifai.config import CaptionConfig
from lattifai.config.caption import KaraokeConfig
from lattifai.utils import safe_print


@run.cli.entrypoint(name="convert", namespace="caption")
def convert(
    input_path: Pathlike,
    output_path: Pathlike,
    include_speaker_in_text: bool = False,
    normalize_text: bool = False,
    word_level: bool = False,
    karaoke: bool = False,
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
        word_level: Use word-level output format if supported.
            When True without karaoke: outputs word-per-segment (each word as separate segment).
            JSON format will include a 'words' field with word-level timestamps.
        karaoke: Enable karaoke styling (requires word_level=True).
            When True: outputs karaoke format (ASS \\kf tags, enhanced LRC, etc.).

    Examples:
        # Basic format conversion (positional arguments)
        lai caption convert input.srt output.vtt

        # Convert with text normalization
        lai caption convert input.srt output.json normalize_text=true

        # Convert to word-per-segment output (if input has alignment)
        lai caption convert input.json output.srt word_level=true

        # Convert to karaoke format (ASS with \\kf tags)
        lai caption convert input.json output.ass word_level=true karaoke=true

        # Export JSON with word-level timestamps
        lai caption convert input.srt output.json word_level=true

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

    # Create karaoke_config if karaoke flag is set
    karaoke_config = KaraokeConfig(enabled=True) if karaoke else None

    caption = Caption.read(input_path, normalize_text=normalize_text)
    caption.write(
        output_path,
        include_speaker_in_text=include_speaker_in_text,
        word_level=word_level,
        karaoke_config=karaoke_config,
    )

    safe_print(f"✅ Converted {input_path} -> {output_path}")
    return output_path


@run.cli.entrypoint(name="normalize", namespace="caption")
def normalize(
    input_path: Pathlike,
    output_path: Pathlike,
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

    Examples:
        # Normalize and save to new file (positional arguments)
        lai caption normalize input.srt output.srt

        # Normalize with format conversion
        lai caption normalize input.vtt output.srt

        # Using keyword arguments (traditional syntax)
        lai caption normalize \
            input_path=input.srt \
            output_path=output.srt
    """
    from pathlib import Path

    from lattifai.caption import Caption

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    caption_obj = Caption.read(input_path, normalize_text=True)
    caption_obj.write(output_path, include_speaker_in_text=True)

    if output_path == input_path:
        safe_print(f"✅ Normalized {input_path} (in-place)")
    else:
        safe_print(f"✅ Normalized {input_path} -> {output_path}")

    return output_path


@run.cli.entrypoint(name="shift", namespace="caption")
def shift(
    input_path: Pathlike,
    output_path: Pathlike,
    seconds: float,
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
        safe_print(f"✅ Shifted timestamps {direction} in {input_path} (in-place)")
    else:
        safe_print(f"✅ Shifted timestamps {direction}: {input_path} -> {output_path}")

    return output_path


@run.cli.entrypoint(name="diff", namespace="caption")
def diff(
    ref_path: Pathlike,
    hyp_path: Pathlike,
    split_sentence: bool = True,
    verbose: bool = True,
):
    """
    Compare and align caption supervisions with transcription segments.

    This command reads a reference caption file and a hypothesis file, then performs
    text alignment to show how they match up. It's useful for comparing
    original subtitles against ASR (Automatic Speech Recognition) results.

    Args:
        ref_path: Path to reference caption file (ground truth)
        hyp_path: Path to hypothesis file (e.g., ASR results)
        split_sentence: Enable sentence splitting before alignment (default: True)
        verbose: Enable verbose output to show detailed alignment info (default: True)

    Examples:
        # Compare reference with hypothesis (positional arguments)
        lai caption diff subtitles.srt transcription.json

        # Disable sentence splitting
        lai caption diff subtitles.srt transcription.json split_sentence=false

        # Disable verbose output
        lai caption diff subtitles.srt transcription.json verbose=false
    """
    from pathlib import Path

    from lattifai.alignment.sentence_splitter import SentenceSplitter
    from lattifai.alignment.text_align import align_supervisions_and_transcription
    from lattifai.caption import Caption

    ref_path = Path(ref_path).expanduser()
    hyp_path = Path(hyp_path).expanduser()

    # Read reference caption (supervisions)
    caption_obj = Caption.read(ref_path)

    # Read hypothesis
    hyp_obj = Caption.read(hyp_path)

    # Apply sentence splitting if enabled
    if split_sentence:
        splitter = SentenceSplitter(device="cpu", lazy_init=True)
        caption_obj.supervisions = splitter.split_sentences(caption_obj.supervisions)
        hyp_obj.supervisions = splitter.split_sentences(hyp_obj.supervisions)

    # Set transcription on caption object
    caption_obj.transcription = hyp_obj.supervisions

    safe_print(f"📖  Reference: {len(caption_obj.supervisions)} segments from {ref_path}")
    safe_print(f"🎤 Hypothesis: {len(caption_obj.transcription)} segments from {hyp_path}")
    if split_sentence:
        safe_print("✂️  Sentence splitting: enabled")
    safe_print("")

    # Perform alignment
    results = align_supervisions_and_transcription(
        caption=caption_obj,
        verbose=verbose,
    )

    # # Print summary
    # safe_print("")
    # safe_print("=" * 72)
    # safe_print(f"📊 Alignment Summary: {len(results)} groups")
    # for idx, (sub_align, asr_align, quality, timestamp, typing) in enumerate(results):
    #     sub_count = len(sub_align) if sub_align else 0
    #     asr_count = len(asr_align) if asr_align else 0
    #     safe_print(f"  Group {idx + 1}: ref={sub_count}, hyp={asr_count}, {quality.info}, typing={typing}")

    return results


def main_diff():
    run.cli.main(diff)


def main_convert():
    run.cli.main(convert)


def main_normalize():
    run.cli.main(normalize)


def main_shift():
    run.cli.main(shift)


if __name__ == "__main__":
    main_convert()
