"""Test timing preservation in sentence splitting."""

import zipfile
from pathlib import Path

from lhotse.supervision import AlignmentItem

from lattifai.alignment.tokenizer import LatticeTokenizer
from lattifai.caption import Caption, Supervision


def test_split_preserves_timing_single_supervision():
    """Test that splitting a single supervision preserves timing based on character distribution."""
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.init_sentence_splitter()

    supervisions = [
        Supervision(
            id="sup-0",
            recording_id="rec",
            start=0.0,
            duration=10.0,
            text="Hello world. This is a test.",
            speaker=None,
        )
    ]

    result = tokenizer.split_sentences(supervisions)

    # Should be split into two sentences
    assert len(result) >= 2

    # First sentence should start at 0.0
    assert result[0].start == 0.0

    # All supervisions should have positive duration
    for sup in result:
        assert sup.duration > 0

    # Total duration should approximately equal original (allowing rounding errors from character-based distribution)
    total_duration = sum(sup.duration for sup in result)
    assert abs(total_duration - 10.0) < 0.5  # Allow up to 0.5s error for character distribution

    # Later sentences should start after earlier ones
    for i in range(1, len(result)):
        assert result[i].start >= result[i - 1].start


def test_split_preserves_timing_multiple_supervisions():
    """Test timing preservation across multiple supervisions."""
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.init_sentence_splitter()

    supervisions = [
        Supervision(
            id="sup-0",
            recording_id="rec",
            start=0.0,
            duration=5.0,
            text="First part.",
            speaker=None,
        ),
        Supervision(
            id="sup-1",
            recording_id="rec",
            start=5.0,
            duration=5.0,
            text="Second part here.",
            speaker=None,
        ),
    ]

    result = tokenizer.split_sentences(supervisions)

    # Should preserve overall timing
    assert result[0].start >= 0.0
    last_end = result[-1].start + result[-1].duration
    assert abs(last_end - 10.0) < 0.1

    # Verify recording_id is preserved
    for sup in result:
        assert sup.recording_id == "rec"


def test_split_with_word_alignment_precise_timing():
    """Test that word-level alignment produces more precise timing."""
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.init_sentence_splitter()

    # Create supervision with word-level alignment
    supervisions = [
        Supervision(
            id="sup-0",
            recording_id="rec",
            start=0.0,
            duration=6.0,
            text="Hello world. Goodbye world.",
            speaker=None,
            alignment={
                "word": [
                    AlignmentItem(symbol="Hello", start=0.0, duration=1.0),
                    AlignmentItem(symbol="world", start=1.0, duration=1.0),
                    AlignmentItem(symbol=".", start=2.0, duration=0.5),
                    AlignmentItem(symbol="Goodbye", start=2.5, duration=1.5),
                    AlignmentItem(symbol="world", start=4.0, duration=1.5),
                    AlignmentItem(symbol=".", start=5.5, duration=0.5),
                ]
            },
        )
    ]

    result = tokenizer.split_sentences(supervisions)

    # Should split into two sentences
    assert len(result) == 2

    # First sentence should roughly cover "Hello world."
    assert result[0].text == "Hello world."
    assert result[0].start == 0.0
    # Duration should be close to 2.5 (up to ".")
    assert 2.0 <= result[0].duration <= 3.0

    # Second sentence should start around 2.5
    assert result[1].text == "Goodbye world."
    assert 2.0 <= result[1].start <= 3.0
    assert result[1].duration >= 2.5


def test_split_character_based_timing_distribution():
    """Test character-based timing distribution when no word alignment."""
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.init_sentence_splitter()

    # 30 characters: "Hello world. Goodbye world."
    # Should split roughly 50/50 in timing
    supervisions = [
        Supervision(
            id="sup-0",
            recording_id="rec",
            start=0.0,
            duration=10.0,
            text="Hello world. Goodbye world.",
            speaker=None,
        )
    ]

    result = tokenizer.split_sentences(supervisions)

    assert len(result) == 2

    # First sentence "Hello world." is 13 chars out of 28 total (excluding extra space)
    # Should take roughly 13/28 ≈ 46% of duration
    expected_first_duration = 10.0 * (13 / 27)  # 27 = 28 - 1 space
    assert abs(result[0].duration - expected_first_duration) < 1.0

    # Second sentence starts where first ends
    assert abs(result[1].start - result[0].duration) < 0.5


def test_split_timing_with_speaker_changes():
    """Test that timing is preserved correctly across speaker changes."""
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.init_sentence_splitter()

    supervisions = [
        Supervision(
            id="sup-0",
            recording_id="rec",
            start=0.0,
            duration=5.0,
            text="Alice speaks first.",
            speaker="Alice",
        ),
        Supervision(
            id="sup-1",
            recording_id="rec",
            start=5.0,
            duration=5.0,
            text="Bob replies now.",
            speaker="Bob",
        ),
    ]

    result = tokenizer.split_sentences(supervisions)

    # Find Alice and Bob's supervisions
    alice_sups = [sup for sup in result if sup.speaker == "Alice"]
    bob_sups = [sup for sup in result if sup.speaker == "Bob"]

    assert len(alice_sups) >= 1
    assert len(bob_sups) >= 1

    # Alice should start at 0.0
    assert alice_sups[0].start == 0.0

    # Bob should start at or after 5.0
    assert bob_sups[0].start >= 5.0

    # All should have positive duration
    for sup in result:
        assert sup.duration > 0


def test_split_timing_text_integrity():
    """Test that all text is accounted for with proper timing."""
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.init_sentence_splitter()

    supervisions = [
        Supervision(
            id="sup-0",
            recording_id="rec",
            start=0.0,
            duration=3.0,
            text="One.",
            speaker=None,
        ),
        Supervision(
            id="sup-1",
            recording_id="rec",
            start=3.0,
            duration=3.0,
            text="Two.",
            speaker=None,
        ),
        Supervision(
            id="sup-2",
            recording_id="rec",
            start=6.0,
            duration=3.0,
            text="Three.",
            speaker=None,
        ),
    ]

    result = tokenizer.split_sentences(supervisions)

    # Verify text integrity
    original_text = " ".join(sup.text for sup in supervisions)
    result_text = " ".join(sup.text for sup in result)
    assert original_text == result_text

    # Verify timing bounds
    assert result[0].start >= 0.0
    last_end = result[-1].start + result[-1].duration
    assert abs(last_end - 9.0) < 0.1

    # All durations should be positive
    for sup in result:
        assert sup.duration > 0
        assert sup.start >= 0.0


def test_split_timing_real_captions():
    """Test timing preservation with real caption files."""
    import pytest

    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.init_sentence_splitter()

    test_data_dir = Path(__file__).parent.parent / "data" / "captions"
    caption_files = [
        test_data_dir / "_xYSQe9oq6c.en.vtt.zip",
        test_data_dir / "7nv1snJRCEI.en.vtt.zip",
        test_data_dir / "eIUqw3_YcCI.en.vtt.zip",
    ]

    for caption_file in caption_files:
        if not caption_file.exists():
            pytest.skip(f"Caption file not found: {caption_file}")

        # Extract and read VTT file
        with zipfile.ZipFile(caption_file, "r") as z:
            vtt_filename = caption_file.stem  # Remove .zip extension
            temp_vtt = Path("/tmp") / vtt_filename
            z.extract(vtt_filename, "/tmp")

        # Parse caption
        caption = Caption.read(temp_vtt, format="vtt")
        original_supervisions = caption.supervisions

        # Skip if no supervisions
        if not original_supervisions:
            continue

        # Split sentences
        result = tokenizer.split_sentences(original_supervisions)

        # Verify basic properties
        assert len(result) > 0, f"No results for {caption_file.name}"

        # Verify text integrity
        original_text = " ".join(sup.text for sup in original_supervisions)
        result_text = " ".join(sup.text for sup in result)
        assert original_text == result_text, f"Text mismatch for {caption_file.name}"

        # Verify timing properties
        assert result[0].start >= 0.0, f"First supervision has negative start for {caption_file.name}"

        # All durations should be positive
        for i, sup in enumerate(result):
            assert sup.duration > 0, f"Supervision {i} has non-positive duration for {caption_file.name}"
            assert sup.start >= 0.0, f"Supervision {i} has negative start for {caption_file.name}"

        # Verify timing order - each supervision should start at or after the previous one
        for i in range(1, len(result)):
            assert (
                result[i].start >= result[i - 1].start
            ), f"Supervision {i} starts before {i-1} for {caption_file.name}"

        # Verify total timing bounds
        original_start = original_supervisions[0].start
        original_end = original_supervisions[-1].start + original_supervisions[-1].duration
        result_start = result[0].start
        result_end = result[-1].start + result[-1].duration

        assert result_start >= original_start - 0.01, f"Result starts before original for {caption_file.name}"
        assert result_end <= original_end + 0.5, f"Result ends too late for {caption_file.name}"

        # Detailed timing verification
        # 1. Check for time gaps between consecutive supervisions
        max_gap = 0.0
        gap_count = 0
        for i in range(1, len(result)):
            prev_end = result[i - 1].start + result[i - 1].duration
            curr_start = result[i].start
            gap = curr_start - prev_end
            if gap > 0.01:  # Allow small floating point errors
                gap_count += 1
                max_gap = max(max_gap, gap)

        # 2. Check for time overlaps
        overlap_count = 0
        max_overlap = 0.0
        for i in range(1, len(result)):
            prev_end = result[i - 1].start + result[i - 1].duration
            curr_start = result[i].start
            overlap = prev_end - curr_start
            if overlap > 0.01:  # Allow small floating point errors
                overlap_count += 1
                max_overlap = max(max_overlap, overlap)

        # 3. Calculate timing coverage (how much of original time is covered)
        total_result_duration = sum(sup.duration for sup in result)
        original_total_duration = original_end - original_start
        coverage_ratio = total_result_duration / original_total_duration if original_total_duration > 0 else 0

        # 4. Check duration distribution
        min_duration = min(sup.duration for sup in result)
        max_duration = max(sup.duration for sup in result)
        avg_duration = total_result_duration / len(result)

        # 5. Verify no supervision extends beyond original bounds
        for i, sup in enumerate(result):
            sup_end = sup.start + sup.duration
            assert (
                sup.start >= original_start - 0.01
            ), f"Supervision {i} starts before original start for {caption_file.name}"
            assert sup_end <= original_end + 0.5, f"Supervision {i} ends after original end for {caption_file.name}"

        # Print statistics for debugging
        print(f"\n{caption_file.name}:")
        print(f"  Original supervisions: {len(original_supervisions)}")
        print(f"  Split supervisions: {len(result)}")
        print(
            f"  Original time range: {original_start:.2f}s - {original_end:.2f}s ({original_end - original_start:.2f}s)"
        )
        print(f"  Result time range: {result_start:.2f}s - {result_end:.2f}s ({result_end - result_start:.2f}s)")
        print(f"  Timing coverage: {coverage_ratio:.2%}")
        print(f"  Duration stats: min={min_duration:.2f}s, max={max_duration:.2f}s, avg={avg_duration:.2f}s")
        if gap_count > 0:
            print(f"  Time gaps: {gap_count} gaps found, max gap={max_gap:.3f}s")
        if overlap_count > 0:
            print(f"  Time overlaps: {overlap_count} overlaps found, max overlap={max_overlap:.3f}s")

        # Assert reasonable timing properties
        assert coverage_ratio >= 0.85, f"Coverage ratio too low ({coverage_ratio:.2%}) for {caption_file.name}"
        assert coverage_ratio <= 1.15, f"Coverage ratio too high ({coverage_ratio:.2%}) for {caption_file.name}"
        assert min_duration > 0, f"Found zero or negative duration for {caption_file.name}"
        assert max_duration < 60.0, f"Found unreasonably long duration ({max_duration:.2f}s) for {caption_file.name}"

        # Clean up temp file
        if temp_vtt.exists():
            temp_vtt.unlink()
