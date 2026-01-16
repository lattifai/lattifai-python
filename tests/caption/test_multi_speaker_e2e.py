#!/usr/bin/env python3
"""
End-to-end tests for multi-speaker caption processing.

Tests that speaker labels are correctly preserved through the CLI alignment pipeline.
Test data covers various speaker label combinations:
- Line with speaker label followed by line without
- Consecutive lines with same speaker
- Different speaker label formats (NAME:, [NAME]:, >> NAME:)
- Speaker appearing again after others
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

from lattifai.caption import Caption, Supervision

load_dotenv(find_dotenv(usecwd=True))

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
MULTI_SPEAKER_AUDIO = TEST_DATA_DIR / "SA1_multi_speaker.mp3"
MULTI_SPEAKER_VTT = TEST_DATA_DIR / "SA1_multi_speaker.vtt"
MULTI_SPEAKER_SRT = TEST_DATA_DIR / "SA1_multi_speaker.srt"

# Expected speaker labels after parsing (format depends on parser behavior)
EXPECTED_SPEAKERS = [
    "ALICE:",  # Line 1: ALICE:
    None,  # Line 2: no speaker
    "BOB:",  # Line 3: BOB:
    "BOB:",  # Line 4: BOB: (consecutive)
    "[SPEAKER_02]:",  # Line 5: [SPEAKER_02]:
    None,  # Line 6: no speaker
    ">> DAVID:",  # Line 7: >> DAVID:
    "ALICE:",  # Line 8: ALICE: (reappears)
]

EXPECTED_TEXT = "She had your dark suit in greasy wash water all year."


class TestMultiSpeakerParsing:
    """Test that multi-speaker captions are correctly parsed."""

    def test_vtt_multi_speaker_parsing(self):
        """Test VTT multi-speaker file is correctly parsed."""
        caption = Caption.read(MULTI_SPEAKER_VTT)

        assert len(caption.supervisions) == 8, f"Expected 8 supervisions, got {len(caption.supervisions)}"

        for i, (sup, expected_speaker) in enumerate(zip(caption.supervisions, EXPECTED_SPEAKERS)):
            # Verify text is correct (without speaker prefix)
            assert (
                EXPECTED_TEXT in sup.text or sup.text == EXPECTED_TEXT
            ), f"Line {i + 1}: Expected text '{EXPECTED_TEXT}', got '{sup.text}'"

            # Verify speaker is correctly parsed
            if expected_speaker is None:
                assert (
                    sup.speaker is None or sup.speaker == ""
                ), f"Line {i + 1}: Expected no speaker, got '{sup.speaker}'"
            else:
                assert (
                    sup.speaker == expected_speaker
                ), f"Line {i + 1}: Expected speaker '{expected_speaker}', got '{sup.speaker}'"

    def test_srt_multi_speaker_parsing(self):
        """Test SRT multi-speaker file is correctly parsed."""
        caption = Caption.read(MULTI_SPEAKER_SRT)

        assert len(caption.supervisions) == 8, f"Expected 8 supervisions, got {len(caption.supervisions)}"

        for i, (sup, expected_speaker) in enumerate(zip(caption.supervisions, EXPECTED_SPEAKERS)):
            # Verify speaker is correctly parsed
            if expected_speaker is None:
                assert (
                    sup.speaker is None or sup.speaker == ""
                ), f"Line {i + 1}: Expected no speaker, got '{sup.speaker}'"
            else:
                assert (
                    sup.speaker == expected_speaker
                ), f"Line {i + 1}: Expected speaker '{expected_speaker}', got '{sup.speaker}'"


class TestMultiSpeakerRoundtrip:
    """Test speaker preservation through read/write cycles."""

    @pytest.mark.parametrize(
        "input_format,output_format",
        [
            ("vtt", "srt"),
            ("srt", "vtt"),
            ("vtt", "vtt"),
            ("srt", "srt"),
            ("vtt", "json"),
            ("srt", "json"),
        ],
    )
    def test_speaker_roundtrip_include_speaker_true(self, input_format, output_format, tmp_path):
        """Test speaker labels preserved with include_speaker_in_text=True."""
        input_file = TEST_DATA_DIR / f"SA1_multi_speaker.{input_format}"
        output_file = tmp_path / f"output.{output_format}"

        # Read input
        caption = Caption.read(input_file)
        original_speakers = [sup.speaker for sup in caption.supervisions]

        # Write output with speaker in text
        caption.write(output_file, include_speaker_in_text=True)

        # Read back
        caption_read = Caption.read(output_file)

        # Verify speaker count
        assert len(caption_read.supervisions) == len(
            original_speakers
        ), f"Speaker count mismatch: {len(caption_read.supervisions)} vs {len(original_speakers)}"

        # Verify each speaker is preserved
        for i, (sup, expected) in enumerate(zip(caption_read.supervisions, original_speakers)):
            if expected is None or expected == "":
                assert (
                    sup.speaker is None or sup.speaker == ""
                ), f"Line {i + 1}: Expected no speaker, got '{sup.speaker}'"
            else:
                assert sup.speaker == expected, f"Line {i + 1}: Expected '{expected}', got '{sup.speaker}'"

    @pytest.mark.parametrize(
        "input_format,output_format",
        [
            ("vtt", "srt"),
            ("srt", "vtt"),
            ("vtt", "vtt"),
            ("srt", "srt"),
        ],
    )
    def test_speaker_roundtrip_include_speaker_false(self, input_format, output_format, tmp_path):
        """Test that speaker labels are NOT in text with include_speaker_in_text=False."""
        input_file = TEST_DATA_DIR / f"SA1_multi_speaker.{input_format}"
        output_file = tmp_path / f"output.{output_format}"

        # Read input
        caption = Caption.read(input_file)

        # Write output WITHOUT speaker in text
        caption.write(output_file, include_speaker_in_text=False)

        # Read back
        caption_read = Caption.read(output_file)

        # Verify speaker count
        assert len(caption_read.supervisions) == len(
            caption.supervisions
        ), f"Segment count mismatch: {len(caption_read.supervisions)} vs {len(caption.supervisions)}"

        # Verify NO speaker labels in output (they should all be None or empty)
        for i, sup in enumerate(caption_read.supervisions):
            assert (
                sup.speaker is None or sup.speaker == ""
            ), f"Line {i + 1}: Expected no speaker with include_speaker_in_text=False, got '{sup.speaker}'"

        # Verify text does NOT contain speaker prefix
        for i, sup in enumerate(caption_read.supervisions):
            assert not sup.text.startswith("ALICE:"), f"Line {i + 1}: Text should not start with 'ALICE:'"
            assert not sup.text.startswith("BOB:"), f"Line {i + 1}: Text should not start with 'BOB:'"
            assert not sup.text.startswith("[SPEAKER_"), f"Line {i + 1}: Text should not start with '[SPEAKER_'"
            assert not sup.text.startswith(">> "), f"Line {i + 1}: Text should not start with '>> '"

    def test_include_speaker_true_then_false_roundtrip(self, tmp_path):
        """Test reading file with speakers, writing without, then reading again."""
        # Step 1: Read original with speakers
        caption = Caption.read(MULTI_SPEAKER_VTT)
        original_texts = [sup.text for sup in caption.supervisions]
        original_speakers = [sup.speaker for sup in caption.supervisions]

        # Verify we have speakers
        non_empty_speakers = [s for s in original_speakers if s]
        assert len(non_empty_speakers) == 6, f"Expected 6 speakers, got {len(non_empty_speakers)}"

        # Step 2: Write without speakers
        no_speaker_file = tmp_path / "no_speaker.srt"
        caption.write(no_speaker_file, include_speaker_in_text=False)

        # Step 3: Read back - should have no speakers
        caption_no_speaker = Caption.read(no_speaker_file)
        for i, sup in enumerate(caption_no_speaker.supervisions):
            assert (
                sup.speaker is None or sup.speaker == ""
            ), f"Line {i + 1}: Should have no speaker, got '{sup.speaker}'"
            # Text should be clean (same as original without speaker prefix)
            assert sup.text == original_texts[i], f"Line {i + 1}: Text mismatch: '{sup.text}' vs '{original_texts[i]}'"

        # Step 4: Write WITH speakers again
        with_speaker_file = tmp_path / "with_speaker.srt"
        caption_no_speaker.write(with_speaker_file, include_speaker_in_text=True)

        # Step 5: Read back - should still have no speakers (they were lost)
        caption_final = Caption.read(with_speaker_file)
        for i, sup in enumerate(caption_final.supervisions):
            assert (
                sup.speaker is None or sup.speaker == ""
            ), f"Line {i + 1}: Speaker should remain empty after losing it, got '{sup.speaker}'"

    def test_include_speaker_false_preserves_text_content(self, tmp_path):
        """Test that text content is preserved correctly when include_speaker_in_text=False."""
        caption = Caption.read(MULTI_SPEAKER_VTT)

        # Write without speakers
        output_file = tmp_path / "output.srt"
        caption.write(output_file, include_speaker_in_text=False)

        # Read back and verify text is exactly the expected text
        caption_read = Caption.read(output_file)
        for i, sup in enumerate(caption_read.supervisions):
            assert sup.text == EXPECTED_TEXT, f"Line {i + 1}: Text should be '{EXPECTED_TEXT}', got '{sup.text}'"

    def test_speaker_pattern_coverage(self, tmp_path):
        """Test all speaker patterns are handled correctly."""
        caption = Caption.read(MULTI_SPEAKER_VTT)

        # Verify we have the expected pattern coverage
        speakers = [sup.speaker for sup in caption.supervisions]

        # Pattern: speaker followed by no speaker
        assert speakers[0] is not None and (
            speakers[1] is None or speakers[1] == ""
        ), "Pattern 'speaker -> no speaker' not covered"

        # Pattern: consecutive same speaker
        assert speakers[2] == speakers[3] and speakers[2] is not None, "Pattern 'consecutive same speaker' not covered"

        # Pattern: different speaker formats
        speaker_formats = [s for s in speakers if s]
        assert any(
            ":" in s and not s.startswith("[") and not s.startswith(">>") for s in speaker_formats
        ), "Pattern 'NAME:' format not covered"
        assert any(s.startswith("[SPEAKER_") for s in speaker_formats), "Pattern '[SPEAKER_XX]:' format not covered"
        assert any(s.startswith(">>") for s in speaker_formats), "Pattern '>> NAME:' format not covered"

        # Pattern: speaker reappears
        assert speakers[0] == speakers[7], "Pattern 'speaker reappears' not covered"


class TestMultiSpeakerCLI:
    """End-to-end CLI tests for multi-speaker processing."""

    @pytest.mark.skipif(not os.environ.get("LATTIFAI_API_KEY"), reason="Requires LATTIFAI_API_KEY environment variable")
    def test_cli_alignment_preserves_speakers(self, tmp_path):
        """Test CLI alignment command preserves speaker labels (include_speaker_in_text=True by default)."""
        output_file = tmp_path / "aligned.srt"

        # Run CLI alignment (include_speaker_in_text=True is default)
        cmd = [
            "lai",
            "alignment",
            "align",
            str(MULTI_SPEAKER_AUDIO),
            str(MULTI_SPEAKER_VTT),
            str(output_file),
            "-Y",
            "alignment.model_hub=modelscope",  # Skip confirmation prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for model download + alignment
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

        # Read output and verify speakers
        caption = Caption.read(output_file)

        assert len(caption.supervisions) == 8, f"Expected 8 supervisions, got {len(caption.supervisions)}"

        # Verify speaker preservation
        for i, (sup, expected) in enumerate(zip(caption.supervisions, EXPECTED_SPEAKERS)):
            if expected is None or expected == "":
                assert (
                    sup.speaker is None or sup.speaker == ""
                ), f"Line {i + 1}: Speaker should be empty after alignment, got '{sup.speaker}'"
            else:
                assert (
                    sup.speaker == expected
                ), f"Line {i + 1}: Expected '{expected}' after alignment, got '{sup.speaker}'"

    @pytest.mark.skipif(not os.environ.get("LATTIFAI_API_KEY"), reason="Requires LATTIFAI_API_KEY environment variable")
    def test_cli_alignment_include_speaker_false(self, tmp_path):
        """Test CLI alignment with caption.include_speaker_in_text=false."""
        output_file = tmp_path / "aligned_no_speaker.srt"

        # Run CLI alignment with include_speaker_in_text=false
        cmd = [
            "lai",
            "alignment",
            "align",
            str(MULTI_SPEAKER_AUDIO),
            str(MULTI_SPEAKER_VTT),
            str(output_file),
            "caption.include_speaker_in_text=false",
            "-Y",
            "alignment.model_hub=modelscope",  # Skip confirmation prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for model download + alignment
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

        # Read output and verify NO speakers
        caption = Caption.read(output_file)

        assert len(caption.supervisions) == 8, f"Expected 8 supervisions, got {len(caption.supervisions)}"

        # Verify NO speaker labels in output
        for i, sup in enumerate(caption.supervisions):
            assert (
                sup.speaker is None or sup.speaker == ""
            ), f"Line {i + 1}: Expected no speaker with include_speaker_in_text=false, got '{sup.speaker}'"
            # Verify text doesn't contain speaker prefix
            assert not any(
                sup.text.startswith(prefix) for prefix in ["ALICE:", "BOB:", "[SPEAKER_", ">> "]
            ), f"Line {i + 1}: Text should not contain speaker prefix: '{sup.text}'"

    @pytest.mark.skipif(not os.environ.get("LATTIFAI_API_KEY"), reason="Requires LATTIFAI_API_KEY environment variable")
    @pytest.mark.parametrize("output_format", ["srt", "vtt", "json"])
    def test_cli_alignment_different_output_formats(self, output_format, tmp_path):
        """Test CLI alignment preserves speakers across different output formats."""
        output_file = tmp_path / f"aligned.{output_format}"

        cmd = [
            "lai",
            "alignment",
            "align",
            str(MULTI_SPEAKER_AUDIO),
            str(MULTI_SPEAKER_VTT),
            str(output_file),
            "-Y",
            "alignment.model_hub=modelscope",  # Skip confirmation prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for model download + alignment
        )

        assert result.returncode == 0, f"CLI failed for {output_format}: {result.stderr}"
        assert output_file.exists(), f"Output file not created for {output_format}"

        # Read and verify
        caption = Caption.read(output_file)
        speakers = [sup.speaker for sup in caption.supervisions]

        # Count non-empty speakers (should be 6: ALICE, BOB, BOB, SPEAKER_02, DAVID, ALICE)
        non_empty = [s for s in speakers if s and s.strip()]
        assert len(non_empty) == 6, f"Expected 6 non-empty speakers, got {len(non_empty)} in {output_format}"

    @pytest.mark.skipif(not os.environ.get("LATTIFAI_API_KEY"), reason="Requires LATTIFAI_API_KEY environment variable")
    @pytest.mark.parametrize("include_speaker", [True, False])
    def test_cli_alignment_include_speaker_param(self, include_speaker, tmp_path):
        """Test CLI alignment with both include_speaker_in_text=true and false."""
        output_file = tmp_path / f"aligned_speaker_{include_speaker}.srt"

        cmd = [
            "lai",
            "alignment",
            "align",
            str(MULTI_SPEAKER_AUDIO),
            str(MULTI_SPEAKER_VTT),
            str(output_file),
            f"caption.include_speaker_in_text={str(include_speaker).lower()}",
            "-Y",
            "alignment.model_hub=modelscope",  # Skip confirmation prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for model download + alignment
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

        caption = Caption.read(output_file)

        if include_speaker:
            # Should have 6 non-empty speakers
            non_empty = [s for s in [sup.speaker for sup in caption.supervisions] if s and s.strip()]
            assert len(non_empty) == 6, f"Expected 6 speakers with include_speaker=True, got {len(non_empty)}"
        else:
            # Should have 0 speakers
            for i, sup in enumerate(caption.supervisions):
                assert (
                    sup.speaker is None or sup.speaker == ""
                ), f"Line {i + 1}: Expected no speaker with include_speaker=False, got '{sup.speaker}'"


class TestMultiSpeakerEdgeCases:
    """Test edge cases in multi-speaker handling."""

    def test_empty_speaker_not_overwritten(self, tmp_path):
        """Test that lines without speakers stay without speakers."""
        caption = Caption.read(MULTI_SPEAKER_VTT)

        # Lines 2 and 6 should have no speaker
        assert caption.supervisions[1].speaker is None or caption.supervisions[1].speaker == ""
        assert caption.supervisions[5].speaker is None or caption.supervisions[5].speaker == ""

        # Write and read back
        output_file = tmp_path / "output.srt"
        caption.write(output_file, include_speaker_in_text=True)
        caption_read = Caption.read(output_file)

        # Verify empty speakers are preserved
        assert caption_read.supervisions[1].speaker is None or caption_read.supervisions[1].speaker == ""
        assert caption_read.supervisions[5].speaker is None or caption_read.supervisions[5].speaker == ""

    def test_consecutive_same_speaker_preserved(self, tmp_path):
        """Test consecutive lines with same speaker are preserved."""
        caption = Caption.read(MULTI_SPEAKER_VTT)

        # Lines 3 and 4 both have BOB:
        assert caption.supervisions[2].speaker == caption.supervisions[3].speaker
        assert caption.supervisions[2].speaker == "BOB:"

        # Write and read back
        output_file = tmp_path / "output.vtt"
        caption.write(output_file, include_speaker_in_text=True)
        caption_read = Caption.read(output_file)

        # Verify both still have BOB:
        assert caption_read.supervisions[2].speaker == "BOB:"
        assert caption_read.supervisions[3].speaker == "BOB:"

    def test_speaker_reappearance_preserved(self, tmp_path):
        """Test that a speaker appearing multiple times is handled correctly."""
        caption = Caption.read(MULTI_SPEAKER_VTT)

        # ALICE appears at lines 1 and 8
        assert caption.supervisions[0].speaker == "ALICE:"
        assert caption.supervisions[7].speaker == "ALICE:"

        # Write and read back
        output_file = tmp_path / "output.srt"
        caption.write(output_file, include_speaker_in_text=True)
        caption_read = Caption.read(output_file)

        # Verify ALICE preserved at both positions
        assert caption_read.supervisions[0].speaker == "ALICE:"
        assert caption_read.supervisions[7].speaker == "ALICE:"


def run_tests():
    """Run all tests manually."""
    print("🧪 Running Multi-Speaker E2E Tests\n")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("\n📄 Testing multi-speaker parsing...")
        test_parsing = TestMultiSpeakerParsing()
        test_parsing.test_vtt_multi_speaker_parsing()
        print("  ✓ VTT parsing")
        test_parsing.test_srt_multi_speaker_parsing()
        print("  ✓ SRT parsing")

        print("\n🔄 Testing speaker roundtrip (include_speaker_in_text=True)...")
        test_roundtrip = TestMultiSpeakerRoundtrip()
        for in_fmt, out_fmt in [("vtt", "srt"), ("srt", "vtt"), ("vtt", "json")]:
            test_roundtrip.test_speaker_roundtrip_include_speaker_true(in_fmt, out_fmt, tmp_path)
            print(f"  ✓ {in_fmt} -> {out_fmt} (include_speaker=True)")

        print("\n🔄 Testing speaker roundtrip (include_speaker_in_text=False)...")
        for in_fmt, out_fmt in [("vtt", "srt"), ("srt", "vtt")]:
            test_roundtrip.test_speaker_roundtrip_include_speaker_false(in_fmt, out_fmt, tmp_path)
            print(f"  ✓ {in_fmt} -> {out_fmt} (include_speaker=False)")

        test_roundtrip.test_include_speaker_true_then_false_roundtrip(tmp_path)
        print("  ✓ True -> False roundtrip")

        test_roundtrip.test_include_speaker_false_preserves_text_content(tmp_path)
        print("  ✓ Text content preserved with include_speaker=False")

        test_roundtrip.test_speaker_pattern_coverage(tmp_path)
        print("  ✓ Pattern coverage verified")

        print("\n🔧 Testing edge cases...")
        test_edge = TestMultiSpeakerEdgeCases()
        test_edge.test_empty_speaker_not_overwritten(tmp_path)
        print("  ✓ Empty speaker preservation")
        test_edge.test_consecutive_same_speaker_preserved(tmp_path)
        print("  ✓ Consecutive speaker preservation")
        test_edge.test_speaker_reappearance_preserved(tmp_path)
        print("  ✓ Speaker reappearance preservation")

        # CLI tests (require API key)
        if os.environ.get("LATTIFAI_API_KEY"):
            print("\n🖥️  Testing CLI alignment (with API key)...")
            test_cli = TestMultiSpeakerCLI()

            test_cli.test_cli_alignment_preserves_speakers(tmp_path)
            print("  ✓ CLI alignment preserves speakers (include_speaker=True)")

            test_cli.test_cli_alignment_include_speaker_false(tmp_path)
            print("  ✓ CLI alignment removes speakers (include_speaker=False)")

            for fmt in ["srt", "vtt", "json"]:
                test_cli.test_cli_alignment_different_output_formats(fmt, tmp_path)
                print(f"  ✓ CLI alignment output format: {fmt}")

            for include in [True, False]:
                test_cli.test_cli_alignment_include_speaker_param(include, tmp_path)
                print(f"  ✓ CLI alignment include_speaker={include}")
        else:
            print("\n⏭️  Skipping CLI tests (LATTIFAI_API_KEY not set)")

    print("\n" + "=" * 60)
    print("✅ All multi-speaker tests passed!")
    print("\n📝 Test Coverage:")
    print("   • Line with speaker -> Line without speaker")
    print("   • Consecutive lines with same speaker")
    print("   • Different speaker formats (NAME:, [SPEAKER_XX]:, >> NAME:)")
    print("   • Speaker reappearing after others")
    print("   • Format conversion roundtrip (VTT/SRT/JSON)")
    print("   • include_speaker_in_text=True (preserve speakers)")
    print("   • include_speaker_in_text=False (remove speakers)")
    if os.environ.get("LATTIFAI_API_KEY"):
        print("   • CLI alignment end-to-end tests")


if __name__ == "__main__":
    import sys

    try:
        run_tests()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
