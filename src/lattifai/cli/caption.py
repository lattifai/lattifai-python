"""Caption CLI entry point with nemo_run."""

import re
from typing import List, Optional

import nemo_run as run

from lattifai.caption.config import KaraokeConfig
from lattifai.types import Pathlike
from lattifai.utils import safe_print


def align_timestamps_from_ref(
    supervisions: List,
    ref_supervisions: List,
) -> List:
    """Align timestamps of supervisions using a reference caption as timing source.

    Matches each input supervision to reference supervisions by text similarity,
    using both word-level keys (Latin scripts) and character-level keys (CJK).
    Searches sequentially through the reference stream to handle repeated text.
    Preserves original text and speaker labels from input; only timestamps change.

    Args:
        supervisions: Input supervisions (good text/speaker, coarse timestamps).
        ref_supervisions: Reference supervisions (accurate timestamps).

    Returns:
        Updated supervisions with reference-aligned timestamps.
    """
    if not supervisions or not ref_supervisions:
        return supervisions

    def _normalize(text: str) -> str:
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # strip md links
        text = re.sub(r"[^\w\s]", "", text.lower())
        return " ".join(text.split())

    # Build normalized text stream from reference, skipping empty entries
    ref_entries = []
    for s in ref_supervisions:
        norm = _normalize(s.text)
        if norm:
            ref_entries.append((s.start, s.start + s.duration, norm))

    if not ref_entries:
        return supervisions

    ref_stream = " ".join(norm for _, _, norm in ref_entries)
    ref_end_time = max(end for _, end, _ in ref_entries if end > 0) if ref_entries else 0.0

    # Build character-offset to timestamp mapping
    char_offsets = []
    pos = 0
    for start, _end, norm in ref_entries:
        char_offsets.append((pos, start))
        pos += len(norm) + 1  # +1 for space separator

    def _lookup_timestamp(char_idx: int) -> float:
        """Binary search for the timestamp at a character position."""
        lo, hi = 0, len(char_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if char_offsets[mid][0] <= char_idx:
                lo = mid
            else:
                hi = mid - 1
        return char_offsets[lo][1]

    def _make_keys(norm_text: str) -> List[str]:
        """Generate match keys of decreasing specificity.

        Word-based keys work well for Latin scripts (space-separated words).
        Character-based keys handle CJK and serve as fallback for Latin.
        """
        keys = []
        words = norm_text.split()

        # Word-based keys (Latin scripts)
        for n in [8, 5, 3]:
            if len(words) >= n:
                keys.append(" ".join(words[:n]))

        # Character-based keys (CJK + universal fallback)
        for n in [20, 12, 6]:
            if len(norm_text) >= n:
                key = norm_text[:n]
                if key not in keys:
                    keys.append(key)

        return keys

    def _find_time(text: str, search_from: int) -> tuple:
        """Find timestamp for text, searching forward from given position.

        Returns (timestamp, matched_char_position) or (None, search_from).
        """
        norm = _normalize(text)
        if not norm:
            return None, search_from

        keys = _make_keys(norm)
        for key in keys:
            idx = ref_stream.find(key, search_from)
            if idx >= 0:
                return _lookup_timestamp(idx), idx

        return None, search_from

    # Align each supervision with sequential search
    matched = 0
    prev_start = 0.0
    search_pos = 0
    results = []
    for sup in supervisions:
        found, match_pos = _find_time(sup.text, search_pos)
        if found is not None:
            sup.start = found
            search_pos = match_pos + 1  # advance past this match
            matched += 1
        else:
            # Fallback: keep original timestamp but ensure monotonicity
            sup.start = max(sup.start, prev_start)
        prev_start = sup.start
        results.append(sup)

    # Recalculate durations from next segment's start
    for i in range(len(results) - 1):
        results[i].duration = max(results[i + 1].start - results[i].start, 0.0)
    if results:
        last_dur = ref_end_time - results[-1].start
        results[-1].duration = max(last_dur, 0.1) if last_dur > 0 else max(results[-1].duration, 0.1)

    safe_print(f"   Aligned {matched}/{len(results)} segments from reference")
    return results


@run.cli.entrypoint(name="convert", namespace="caption")
def convert(
    input_path: Pathlike,
    output_path: Pathlike,
    reference: Optional[Pathlike] = None,
    input_format: Optional[str] = None,
    include_speaker_in_text: bool = False,
    normalize_text: bool = False,
    word_level: bool = False,
    karaoke: bool = False,
    translation_first: bool = False,
):
    """
    Convert caption file to another format.

    This command reads a caption file from one format and writes it to another format,
    preserving all timing information, text content, and speaker labels (if present).
    Supports common caption formats including SRT, VTT, JSON, and Praat TextGrid.

    When ``reference`` is provided, timestamps are aligned from the reference caption
    via text matching. This is useful for combining human-edited text (with coarse
    timestamps) with ASR subtitles (with accurate timestamps).

    Shortcut: invoking ``laisub-convert`` is equivalent to running ``lai caption convert``.

    Args:
        input_path: Path to input caption file (supports SRT, VTT, JSON, TextGrid formats)
        output_path: Path to output caption file (format determined by file extension)
        reference: Optional reference caption for timestamp alignment.
        input_format: Explicitly specify input format (e.g., 'markdown', 'srt').
            If None (default), auto-detect from file extension/content.
            When provided, timestamps are matched from the reference via text similarity.
            Input keeps its text and speaker labels; only timestamps are updated.
        include_speaker_in_text: Preserve speaker labels in caption text content.
        normalize_text: Whether to normalize caption text during conversion.
            This applies text cleaning such as removing HTML tags, decoding entities,
            collapsing whitespace, and standardizing punctuation.
        word_level: Use word-level output format if supported.
            When True without karaoke: outputs word-per-segment (each word as separate segment).
            JSON format will include a 'words' field with word-level timestamps.
        karaoke: Enable karaoke styling (requires word_level=True).
            When True: outputs karaoke format (ASS \\kf tags, enhanced LRC, etc.).
        translation_first: Place translation text above original text in bilingual output.
            When True: translation appears on the first line, original on the second line.

    Examples:
        # Basic format conversion (positional arguments)
        lai caption convert input.srt output.vtt

        # Convert with text normalization
        lai caption convert input.srt output.json normalize_text=true

        # Align timestamps from a reference subtitle
        lai caption convert transcript.md output.srt reference=youtube.en.srt

        # Convert to karaoke format (ASS with \\kf tags)
        lai caption convert input.json output.ass word_level=true karaoke=true

        # Using keyword arguments (traditional syntax)
        lai caption convert \\
            input_path=input.srt \\
            output_path=output.TextGrid
    """
    from pathlib import Path

    from lattifai.data import Caption

    # Create karaoke_config if karaoke flag is set
    karaoke_config = KaraokeConfig(enabled=True) if karaoke else None

    try:
        caption = Caption.read(input_path, normalize_text=normalize_text, format=input_format)
    except Exception:
        caption = Caption()

    # Fallback: if .md file yields 0 supervisions, try _parse_transcript_html
    if not caption.supervisions and str(input_path).endswith(".md"):
        try:
            from lattifai.youtube.client import YouTubeDownloader

            md_text = Path(input_path).read_text(encoding="utf-8")
            parsed = YouTubeDownloader._parse_transcript_html(md_text)
            if parsed:
                caption = Caption.from_string(parsed, format="markdown")
                # Strip markdown links [text](url) → text from supervisions
                _md_link = re.compile(r"\[([^\]]+)\]\([^)]+\)")
                for sup in caption.supervisions:
                    sup.text = _md_link.sub(r"\1", sup.text)
                safe_print(f"   Parsed transcript markdown ({len(caption.supervisions)} segments)")
        except Exception:
            pass

    # Align timestamps from reference if provided
    if reference:
        ref_caption = Caption.read(reference)
        caption.supervisions = align_timestamps_from_ref(caption.supervisions, ref_caption.supervisions)

    caption.write(
        output_path,
        include_speaker_in_text=include_speaker_in_text,
        word_level=word_level,
        karaoke_config=karaoke_config,
        translation_first=translation_first,
    )

    safe_print(f"Converted {input_path} -> {output_path}")
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

    from lattifai.data import Caption

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

    from lattifai.data import Caption

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
    reference: Pathlike,
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
        reference: Path to reference caption file (ground truth)
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

    from lattifai.alignment.text_align import align_supervisions_and_transcription
    from lattifai.caption import SentenceSplitter
    from lattifai.data import Caption

    reference = Path(reference).expanduser()
    hyp_path = Path(hyp_path).expanduser()

    # Read reference caption (supervisions)
    caption_obj = Caption.read(reference)

    # Read hypothesis
    hyp_obj = Caption.read(hyp_path)

    # Apply sentence splitting if enabled
    if split_sentence:
        splitter = SentenceSplitter(device="cpu", lazy_init=True)
        caption_obj.supervisions = splitter.split_sentences(caption_obj.supervisions)
        hyp_obj.supervisions = splitter.split_sentences(hyp_obj.supervisions)

    # Set transcription on caption object
    caption_obj.transcription = hyp_obj.supervisions

    safe_print(f"📖  Reference: {len(caption_obj.supervisions)} segments from {reference}")
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
