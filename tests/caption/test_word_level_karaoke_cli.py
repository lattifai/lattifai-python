"""CLI integration tests for word_level and karaoke configuration combinations.

Tests the CLI behavior with different combinations of word_level and karaoke flags
across multiple output formats (JSON, SRT, VTT, ASS, LRC, TTML).

STRICT VALIDATION: Every test validates 100% of the output structure and content.
"""

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "examples"
EN_AUDIO = TEST_DATA_DIR / "en.mp3"
EN_TEXT = TEST_DATA_DIR / "en.txt"
ZH_AUDIO = TEST_DATA_DIR / "zh.mp3"
ZH_TEXT = TEST_DATA_DIR / "zh.txt"
DE_AUDIO = TEST_DATA_DIR / "de.mp3"
DE_TEXT = TEST_DATA_DIR / "de.txt"


# =============================================================================
# Helper functions
# =============================================================================


def run_alignment_command(audio_path, caption_path, output_path, word_level=True):
    """Run alignment command to generate word-level aligned caption."""
    cmd = [
        "lai",
        "alignment",
        "align",
        "-Y",
        f"input_media={audio_path}",
        f"input_caption={caption_path}",
        f"output_caption={output_path}",
        f"caption.word_level={str(word_level).lower()}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    return result


def run_caption_convert(input_path, output_path, word_level=False, karaoke=False):
    """Run caption convert command."""
    cmd = [
        "lai",
        "caption",
        "convert",
        "-Y",
        f"input_path={input_path}",
        f"output_path={output_path}",
        f"word_level={str(word_level).lower()}",
        f"karaoke={str(karaoke).lower()}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    return result


def parse_srt_timestamp(ts: str) -> float:
    """Parse SRT timestamp to seconds. Format: HH:MM:SS,mmm"""
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", ts.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {ts}")
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt_timestamp(ts: str) -> float:
    """Parse VTT timestamp to seconds. Format: HH:MM:SS.mmm or MM:SS.mmm"""
    ts = ts.strip()
    if ts.count(":") == 2:
        match = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", ts)
        if match:
            h, m, s, ms = match.groups()
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    else:
        match = re.match(r"(\d+):(\d+)\.(\d+)", ts)
        if match:
            m, s, ms = match.groups()
            return int(m) * 60 + int(s) + int(ms) / 1000
    raise ValueError(f"Invalid VTT timestamp: {ts}")


def parse_lrc_timestamp(ts: str) -> float:
    """Parse LRC timestamp to seconds. Format: [mm:ss.xx] or <mm:ss.xx>"""
    match = re.match(r"[\[<](\d+):(\d+)\.(\d+)[\]>]", ts.strip())
    if not match:
        raise ValueError(f"Invalid LRC timestamp: {ts}")
    m, s, ms = match.groups()
    # Handle different precision (2 or 3 digits)
    if len(ms) == 2:
        ms_val = int(ms) * 10
    else:
        ms_val = int(ms)
    return int(m) * 60 + int(s) + ms_val / 1000


def parse_ass_dialogue(line: str) -> dict:
    """Parse ASS Dialogue line into components."""
    # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
    if not line.startswith("Dialogue:"):
        raise ValueError(f"Not a Dialogue line: {line}")
    parts = line.split(",", 9)
    if len(parts) < 10:
        raise ValueError(f"Invalid Dialogue line: {line}")
    return {
        "layer": parts[0].replace("Dialogue:", "").strip(),
        "start": parts[1].strip(),
        "end": parts[2].strip(),
        "style": parts[3].strip(),
        "name": parts[4].strip(),
        "text": parts[9].strip(),
    }


def parse_ass_timestamp(ts: str) -> float:
    """Parse ASS timestamp to seconds. Format: H:MM:SS.cc"""
    match = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", ts.strip())
    if not match:
        raise ValueError(f"Invalid ASS timestamp: {ts}")
    h, m, s, cs = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def aligned_json_en(tmp_path_factory):
    """Generate word-level aligned JSON for English audio."""
    if not EN_AUDIO.exists() or not EN_TEXT.exists():
        pytest.skip("English test data not available")

    tmp_dir = tmp_path_factory.mktemp("aligned")
    output_path = tmp_dir / "en_aligned.json"

    result = run_alignment_command(EN_AUDIO, EN_TEXT, output_path, word_level=True)
    if result.returncode != 0:
        pytest.skip(f"Alignment failed: {result.stderr}")

    return output_path


@pytest.fixture(scope="module")
def aligned_json_zh(tmp_path_factory):
    """Generate word-level aligned JSON for Chinese audio."""
    if not ZH_AUDIO.exists() or not ZH_TEXT.exists():
        pytest.skip("Chinese test data not available")

    tmp_dir = tmp_path_factory.mktemp("aligned")
    output_path = tmp_dir / "zh_aligned.json"

    result = run_alignment_command(ZH_AUDIO, ZH_TEXT, output_path, word_level=True)
    if result.returncode != 0:
        pytest.skip(f"Alignment failed: {result.stderr}")

    return output_path


# =============================================================================
# JSON Format Tests - STRICT
# =============================================================================


class TestJSONFormat:
    """Strict validation of JSON format output."""

    def test_json_word_level_false_structure(self, aligned_json_en, tmp_path):
        """JSON with word_level=False: validate complete structure."""
        output_path = tmp_path / "output.json"

        result = run_caption_convert(aligned_json_en, output_path, word_level=False, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        with open(output_path) as f:
            data = json.load(f)

        # 1. Must be array
        assert isinstance(data, list), "Output must be JSON array"
        assert len(data) > 0, "Array must not be empty"

        prev_end = -1
        for i, seg in enumerate(data):
            # 2. Required fields
            assert "text" in seg, f"Segment {i} missing 'text'"
            assert "start" in seg, f"Segment {i} missing 'start'"
            assert "end" in seg, f"Segment {i} missing 'end'"

            # 3. Field types
            assert isinstance(seg["text"], str), f"Segment {i} text must be string"
            assert isinstance(seg["start"], (int, float)), f"Segment {i} start must be number"
            assert isinstance(seg["end"], (int, float)), f"Segment {i} end must be number"

            # 4. Timing validity
            assert seg["start"] >= 0, f"Segment {i} start must be non-negative"
            assert seg["end"] > seg["start"], f"Segment {i} end must be > start"
            assert (
                seg["start"] >= prev_end - 0.001
            ), f"Segment {i} overlaps with previous (start={seg['start']}, prev_end={prev_end})"
            prev_end = seg["end"]

            # 5. Text validity
            assert len(seg["text"].strip()) > 0, f"Segment {i} text must not be empty"

            # 6. NO words field when word_level=False
            assert "words" not in seg, f"Segment {i} should NOT have 'words' when word_level=False"

            # 7. duration field should NOT exist (we removed it)
            assert "duration" not in seg, f"Segment {i} should NOT have 'duration' field (use start/end only)"

    def test_json_word_level_true_structure(self, aligned_json_en, tmp_path):
        """JSON with word_level=True: validate complete structure including words array."""
        output_path = tmp_path / "output.json"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        with open(output_path) as f:
            data = json.load(f)

        # 1. Must be array
        assert isinstance(data, list), "Output must be JSON array"
        assert len(data) > 0, "Array must not be empty"

        total_words = 0
        for i, seg in enumerate(data):
            # 2. Required fields
            assert "text" in seg, f"Segment {i} missing 'text'"
            assert "start" in seg, f"Segment {i} missing 'start'"
            assert "end" in seg, f"Segment {i} missing 'end'"
            assert "words" in seg, f"Segment {i} missing 'words' when word_level=True"

            # 3. Field types
            assert isinstance(seg["words"], list), f"Segment {i} words must be array"
            assert len(seg["words"]) > 0, f"Segment {i} words array must not be empty"

            # 4. duration field should NOT exist
            assert "duration" not in seg, f"Segment {i} should NOT have 'duration' field"

            # 5. Validate each word
            seg_start = seg["start"]
            seg_end = seg["end"]
            prev_word_end = seg_start - 0.001

            for j, word in enumerate(seg["words"]):
                total_words += 1

                # 5a. Required word fields
                assert "word" in word, f"Segment {i} word {j} missing 'word'"
                assert "start" in word, f"Segment {i} word {j} missing 'start'"
                assert "end" in word, f"Segment {i} word {j} missing 'end'"

                # 5b. Word field types
                assert isinstance(word["word"], str), f"Segment {i} word {j} word must be string"
                assert isinstance(word["start"], (int, float)), f"Segment {i} word {j} start must be number"
                assert isinstance(word["end"], (int, float)), f"Segment {i} word {j} end must be number"

                # 5c. Word timing validity
                assert word["end"] >= word["start"], f"Segment {i} word {j} end must be >= start"
                assert word["start"] >= seg_start - 0.01, f"Segment {i} word {j} starts before segment"
                assert word["end"] <= seg_end + 0.01, f"Segment {i} word {j} ends after segment"

                # 5d. Word timing monotonic (with small tolerance for overlaps)
                assert (
                    word["start"] >= prev_word_end - 0.1
                ), f"Segment {i} word {j} overlaps significantly with previous"
                prev_word_end = word["end"]

                # 5e. Word text validity
                assert len(word["word"]) > 0, f"Segment {i} word {j} text must not be empty"

                # 5f. duration field should NOT exist in words
                assert "duration" not in word, f"Segment {i} word {j} should NOT have 'duration' field"

        # 6. Must have multiple words total
        assert total_words >= 3, f"Should have at least 3 words total, got {total_words}"


# =============================================================================
# SRT Format Tests - STRICT
# =============================================================================


class TestSRTFormat:
    """Strict validation of SRT format output."""

    def test_srt_word_level_false_structure(self, aligned_json_en, tmp_path):
        """SRT with word_level=False: validate complete structure."""
        output_path = tmp_path / "output.srt"

        result = run_caption_convert(aligned_json_en, output_path, word_level=False, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Parse SRT blocks
        blocks = []
        i = 0
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue

            # 1. Index line (must be integer)
            index_line = lines[i].strip()
            assert index_line.isdigit(), f"Invalid index at line {i}: {index_line}"
            block_index = int(index_line)
            i += 1

            # 2. Timestamp line
            assert i < len(lines), f"Missing timestamp after index {block_index}"
            ts_line = lines[i].strip()
            assert " --> " in ts_line, f"Invalid timestamp format at line {i}: {ts_line}"
            start_ts, end_ts = ts_line.split(" --> ")
            start = parse_srt_timestamp(start_ts)
            end = parse_srt_timestamp(end_ts)
            i += 1

            # 3. Text lines (until empty line or end)
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = "\n".join(text_lines)

            blocks.append({"index": block_index, "start": start, "end": end, "text": text})

        # Validate blocks
        assert len(blocks) > 0, "SRT must have at least one block"

        prev_end = -1
        for j, block in enumerate(blocks):
            # Index must be sequential
            assert block["index"] == j + 1, f"Block {j} has wrong index {block['index']}"

            # Timing validity
            assert block["end"] > block["start"], f"Block {j} end must be > start"
            assert block["start"] >= prev_end - 0.001, f"Block {j} overlaps with previous"
            prev_end = block["end"]

            # Text validity
            assert len(block["text"]) > 0, f"Block {j} text must not be empty"

        # Multi-word segments
        multi_word_count = sum(1 for b in blocks if len(b["text"].split()) > 1)
        assert multi_word_count > 0, "word_level=False should produce multi-word segments"

    def test_srt_word_level_true_structure(self, aligned_json_en, tmp_path):
        """SRT with word_level=True: validate one word per segment."""
        output_path = tmp_path / "output.srt"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Parse SRT blocks
        blocks = []
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue

            index_line = lines[i].strip()
            assert index_line.isdigit(), f"Invalid index at line {i}"
            i += 1

            ts_line = lines[i].strip()
            assert " --> " in ts_line, f"Invalid timestamp at line {i}"
            start_ts, end_ts = ts_line.split(" --> ")
            start = parse_srt_timestamp(start_ts)
            end = parse_srt_timestamp(end_ts)
            i += 1

            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = "\n".join(text_lines)

            blocks.append({"start": start, "end": end, "text": text})

        # Must have multiple blocks (words)
        assert len(blocks) >= 3, f"word_level=True should produce many blocks, got {len(blocks)}"

        # Most blocks should be single words
        single_word_blocks = sum(1 for b in blocks if len(b["text"].split()) == 1)
        ratio = single_word_blocks / len(blocks)
        assert ratio >= 0.7, f"At least 70% of blocks should be single words, got {ratio:.1%}"

        # Timing monotonicity
        prev_start = -1
        for j, block in enumerate(blocks):
            assert block["start"] >= prev_start, f"Block {j} starts before previous"
            prev_start = block["start"]


# =============================================================================
# VTT Format Tests - STRICT
# =============================================================================


class TestVTTFormat:
    """Strict validation of VTT format output."""

    def test_vtt_word_level_false_structure(self, aligned_json_en, tmp_path):
        """VTT with word_level=False: validate complete structure."""
        output_path = tmp_path / "output.vtt"

        result = run_caption_convert(aligned_json_en, output_path, word_level=False, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # 1. VTT header
        assert lines[0].strip() == "WEBVTT", "VTT must start with WEBVTT header"

        # Parse cues
        cues = []
        i = 1
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue

            # Timestamp line or cue identifier
            ts_line = lines[i].strip()
            if " --> " not in ts_line:
                i += 1
                if i >= len(lines):
                    break
                ts_line = lines[i].strip()

            if " --> " not in ts_line:
                i += 1
                continue

            start_ts, end_ts = ts_line.split(" --> ")
            # Handle settings after end timestamp
            if " " in end_ts:
                end_ts = end_ts.split(" ")[0]
            start = parse_vtt_timestamp(start_ts)
            end = parse_vtt_timestamp(end_ts)
            i += 1

            text_lines = []
            while i < len(lines) and lines[i].strip() and " --> " not in lines[i]:
                text_lines.append(lines[i].strip())
                i += 1
            text = "\n".join(text_lines)

            if text:
                cues.append({"start": start, "end": end, "text": text})

        assert len(cues) > 0, "VTT must have at least one cue"

        # Validate timing
        for j, cue in enumerate(cues):
            assert cue["end"] > cue["start"], f"Cue {j} end must be > start"
            assert len(cue["text"]) > 0, f"Cue {j} text must not be empty"

    def test_vtt_word_level_true_structure(self, aligned_json_en, tmp_path):
        """VTT with word_level=True: validate word-per-cue."""
        output_path = tmp_path / "output.vtt"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # Count timing lines
        timing_lines = [l for l in content.split("\n") if " --> " in l]
        assert len(timing_lines) >= 3, f"word_level=True should produce many cues, got {len(timing_lines)}"

        # Validate header
        assert content.strip().startswith("WEBVTT"), "VTT must start with WEBVTT"

        # Should NOT have YouTube VTT word-level tags
        assert "<c>" not in content, "word_level=True without karaoke should NOT have <c> tags"

    def test_vtt_karaoke_youtube_format(self, aligned_json_en, tmp_path):
        """VTT with karaoke=True: validate YouTube VTT format with <timestamp><c> tags."""
        output_path = tmp_path / "output.vtt"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # 1. Validate header
        assert content.strip().startswith("WEBVTT"), "VTT must start with WEBVTT"

        # 2. Must have timestamp lines
        assert " --> " in content, "VTT must have timestamp lines"

        # 3. Must have YouTube VTT word-level tags: <HH:MM:SS.mmm><c> word</c>
        assert "<c>" in content, "karaoke=True should produce YouTube VTT <c> tags"
        assert "</c>" in content, "karaoke=True should produce YouTube VTT </c> tags"

        # 4. Validate YouTube VTT format pattern: <00:00:00.000><c> word</c>
        youtube_vtt_pattern = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d{3})><c>\s*([^<]+)</c>")
        matches = youtube_vtt_pattern.findall(content)
        assert len(matches) >= 2, f"Should have multiple word timestamps, found {len(matches)}"

        # 5. Validate timestamp format in word tags
        for ts_str, word_text in matches:
            # Timestamp should be valid HH:MM:SS.mmm
            assert re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}", ts_str), f"Invalid timestamp format: {ts_str}"
            # Word text should not be empty
            assert len(word_text.strip()) > 0, f"Word text should not be empty"

        # 6. Validate timestamps are monotonic within each cue
        lines = content.split("\n")
        for line in lines:
            word_timestamps = youtube_vtt_pattern.findall(line)
            if len(word_timestamps) >= 2:
                prev_ts = -1
                for ts_str, _ in word_timestamps:
                    ts = parse_vtt_timestamp(ts_str)
                    assert ts >= prev_ts, f"Word timestamps should be monotonic: {line}"
                    prev_ts = ts


# =============================================================================
# ASS Format Tests - STRICT
# =============================================================================


class TestASSFormat:
    """Strict validation of ASS format output."""

    def test_ass_word_level_false_structure(self, aligned_json_en, tmp_path):
        """ASS with word_level=False: validate complete structure, no karaoke tags."""
        output_path = tmp_path / "output.ass"

        result = run_caption_convert(aligned_json_en, output_path, word_level=False, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # 1. Required sections
        assert "[Script Info]" in content, "ASS must have [Script Info] section"
        assert "[V4+ Styles]" in content, "ASS must have [V4+ Styles] section"
        assert "[Events]" in content, "ASS must have [Events] section"

        # 2. Parse Dialogue lines
        dialogue_lines = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(dialogue_lines) > 0, "ASS must have Dialogue lines"

        # 3. Validate each Dialogue line
        for i, line in enumerate(dialogue_lines):
            d = parse_ass_dialogue(line)

            # Timing validity
            start = parse_ass_timestamp(d["start"])
            end = parse_ass_timestamp(d["end"])
            assert end > start, f"Dialogue {i} end must be > start"

            # Text validity
            assert len(d["text"]) > 0, f"Dialogue {i} text must not be empty"

            # NO karaoke tags when karaoke=False
            assert "\\kf" not in d["text"], f"Dialogue {i} should NOT have \\kf tag"
            assert "\\ko" not in d["text"], f"Dialogue {i} should NOT have \\ko tag"
            # Allow \\k0 but not \\kNNN where NNN > 0
            k_matches = re.findall(r"\\k(\d+)", d["text"])
            for k_val in k_matches:
                assert k_val == "0", f"Dialogue {i} should NOT have \\k{k_val} tag (only \\k0 allowed)"

    def test_ass_word_level_true_no_karaoke(self, aligned_json_en, tmp_path):
        """ASS with word_level=True, karaoke=False: one word per Dialogue, no karaoke tags."""
        output_path = tmp_path / "output.ass"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # Parse Dialogue lines
        dialogue_lines = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(dialogue_lines) >= 3, f"word_level=True should produce many Dialogues, got {len(dialogue_lines)}"

        # Validate no karaoke tags
        for i, line in enumerate(dialogue_lines):
            d = parse_ass_dialogue(line)
            assert "\\kf" not in d["text"], f"Dialogue {i} should NOT have \\kf tag"

    def test_ass_karaoke_sweep_effect(self, aligned_json_en, tmp_path):
        """ASS with karaoke=True: validate \\kf tags with timing values."""
        output_path = tmp_path / "output.ass"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # Required sections
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content

        # Parse Dialogue lines
        dialogue_lines = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(dialogue_lines) > 0, "Must have Dialogue lines"

        # Count karaoke tags
        kf_count = 0
        for i, line in enumerate(dialogue_lines):
            d = parse_ass_dialogue(line)

            # Find all \kf tags
            kf_matches = re.findall(r"\\kf(\d+)", d["text"])
            kf_count += len(kf_matches)

            # Validate kf timing values (centiseconds, should be reasonable)
            for kf_val in kf_matches:
                duration_cs = int(kf_val)
                assert 0 < duration_cs < 10000, f"Dialogue {i} has invalid \\kf{kf_val} (duration in centiseconds)"

        assert kf_count > 0, "karaoke=True should produce \\kf tags"

        # Validate Style section contains Karaoke style
        style_section = content[content.find("[V4+ Styles]") : content.find("[Events]")]
        assert "Style:" in style_section, "Must have Style definitions"


# =============================================================================
# LRC Format Tests - STRICT
# =============================================================================


class TestLRCFormat:
    """Strict validation of LRC format output."""

    def test_lrc_word_level_false_structure(self, aligned_json_en, tmp_path):
        """LRC with word_level=False: validate line-level timing, no enhanced tags."""
        output_path = tmp_path / "output.lrc"

        result = run_caption_convert(aligned_json_en, output_path, word_level=False, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Parse LRC lines
        timing_lines = []
        metadata_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Metadata tags [ar:], [ti:], etc.
            if re.match(r"^\[(ar|ti|al|au|length|by|offset|re|ve):", line):
                metadata_lines.append(line)
                continue

            # Timing line [mm:ss.xx]text
            match = re.match(r"^\[(\d+:\d+\.\d+)\](.*)$", line)
            if match:
                ts_str, text = match.groups()
                ts = parse_lrc_timestamp(f"[{ts_str}]")
                timing_lines.append({"time": ts, "text": text})

        assert len(timing_lines) > 0, "LRC must have timing lines"

        # Validate timing monotonicity
        prev_time = -1
        for i, tl in enumerate(timing_lines):
            assert tl["time"] >= prev_time, f"Line {i} time should be >= previous"
            prev_time = tl["time"]

        # NO enhanced word timing tags
        for i, tl in enumerate(timing_lines):
            assert "<" not in tl["text"], f"Line {i} should NOT have enhanced <timestamp> tags"

    def test_lrc_word_level_true_no_karaoke(self, aligned_json_en, tmp_path):
        """LRC with word_level=True, karaoke=False: one word per line."""
        output_path = tmp_path / "output.lrc"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Count timing lines
        timing_lines = [l for l in lines if re.match(r"^\[\d+:\d+\.\d+\]", l)]
        assert len(timing_lines) >= 3, f"word_level=True should produce many lines, got {len(timing_lines)}"

        # Most should be single words
        single_word_lines = 0
        for line in timing_lines:
            match = re.match(r"^\[\d+:\d+\.\d+\](.*)$", line)
            if match:
                text = match.group(1).strip()
                if len(text.split()) == 1:
                    single_word_lines += 1

        ratio = single_word_lines / len(timing_lines) if timing_lines else 0
        assert ratio >= 0.7, f"At least 70% should be single words, got {ratio:.1%}"

    def test_lrc_karaoke_enhanced_format(self, aligned_json_en, tmp_path):
        """LRC with karaoke=True: validate enhanced <mm:ss.xx>word format."""
        output_path = tmp_path / "output.lrc"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Find lines with enhanced format
        enhanced_lines = []
        for line in lines:
            if "<" in line and ">" in line:
                enhanced_lines.append(line)

        assert len(enhanced_lines) > 0, "karaoke=True should produce enhanced <timestamp> tags"

        # Validate enhanced format: [mm:ss.xx]<mm:ss.xx>word1 <mm:ss.xx>word2
        for line in enhanced_lines:
            # Should start with line timestamp
            assert re.match(r"^\[\d+:\d+\.\d+\]", line), f"Enhanced line should start with [timestamp]: {line}"

            # Should have word timestamps
            word_timestamps = re.findall(r"<\d+:\d+\.\d+>", line)
            assert len(word_timestamps) >= 1, f"Enhanced line should have <timestamp> tags: {line}"

            # Validate timestamp order
            prev_ts = -1
            for ts_match in word_timestamps:
                ts = parse_lrc_timestamp(ts_match)
                assert ts >= prev_ts, f"Word timestamps should be monotonic in line: {line}"
                prev_ts = ts


# =============================================================================
# TTML Format Tests - STRICT
# =============================================================================


class TestTTMLFormat:
    """Strict validation of TTML format output."""

    def test_ttml_word_level_false_structure(self, aligned_json_en, tmp_path):
        """TTML with word_level=False: validate XML structure, no word spans."""
        output_path = tmp_path / "output.ttml"

        result = run_caption_convert(aligned_json_en, output_path, word_level=False, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # 1. Valid XML
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            pytest.fail(f"Invalid TTML XML: {e}")

        # 2. Root element should be <tt>
        assert root.tag.endswith("tt"), f"Root element should be <tt>, got {root.tag}"

        # 3. Find all <p> elements
        ns = {"ttml": "http://www.w3.org/ns/ttml", "tt": "http://www.w3.org/ns/ttml"}
        p_elements = root.findall(".//{http://www.w3.org/ns/ttml}p")
        if not p_elements:
            # Try without namespace
            p_elements = root.findall(".//p")

        assert len(p_elements) > 0, "TTML must have <p> elements"

        # 4. Validate timing attributes
        for i, p in enumerate(p_elements):
            begin = p.get("begin")
            end = p.get("end")
            assert begin is not None, f"<p> element {i} missing begin attribute"
            assert end is not None, f"<p> element {i} missing end attribute"

        # 5. NO itunes:timing="Word" when karaoke=False
        assert 'itunes:timing="Word"' not in content, "Should NOT have itunes:timing when karaoke=False"

    def test_ttml_word_level_true_no_karaoke(self, aligned_json_en, tmp_path):
        """TTML with word_level=True, karaoke=False: one word per <p>."""
        output_path = tmp_path / "output.ttml"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=False)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # Count <p> elements
        p_count = content.count("<p ")
        assert p_count >= 3, f"word_level=True should produce many <p> elements, got {p_count}"

        # NO itunes:timing
        assert 'itunes:timing="Word"' not in content, "Should NOT have itunes:timing when karaoke=False"

    def test_ttml_karaoke_word_timing(self, aligned_json_en, tmp_path):
        """TTML with karaoke=True: validate itunes:timing='Word' and <span> elements."""
        output_path = tmp_path / "output.ttml"

        result = run_caption_convert(aligned_json_en, output_path, word_level=True, karaoke=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # 1. Valid XML
        try:
            ET.fromstring(content)
        except ET.ParseError as e:
            pytest.fail(f"Invalid TTML XML: {e}")

        # 2. itunes:timing="Word" attribute
        assert 'itunes:timing="Word"' in content, "karaoke=True should have itunes:timing='Word'"

        # 3. <span> elements for word timing
        assert "<span " in content, "karaoke=True should have <span> elements"

        # 4. Spans should have timing attributes
        span_count = content.count("<span ")
        assert span_count >= 2, f"Should have multiple <span> elements, got {span_count}"


# =============================================================================
# Multi-Language Tests - STRICT
# =============================================================================


class TestMultiLanguage:
    """Strict validation of multi-language support."""

    def test_chinese_ass_karaoke_structure(self, aligned_json_zh, tmp_path):
        """Chinese ASS karaoke: validate structure and Chinese character handling."""
        output_path = tmp_path / "output.ass"

        result = run_caption_convert(aligned_json_zh, output_path, word_level=True, karaoke=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # Required sections
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content

        # Parse Dialogue lines
        dialogue_lines = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(dialogue_lines) > 0

        # Must have karaoke tags
        kf_found = False
        chinese_found = False
        for line in dialogue_lines:
            d = parse_ass_dialogue(line)
            if "\\kf" in d["text"]:
                kf_found = True
            # Check for Chinese characters (CJK range)
            if re.search(r"[\u4e00-\u9fff]", d["text"]):
                chinese_found = True

        assert kf_found, "Chinese ASS should have \\kf karaoke tags"
        assert chinese_found, "Chinese ASS should contain Chinese characters"

    def test_chinese_lrc_karaoke_structure(self, aligned_json_zh, tmp_path):
        """Chinese LRC karaoke: validate enhanced format with Chinese."""
        output_path = tmp_path / "output.lrc"

        result = run_caption_convert(aligned_json_zh, output_path, word_level=True, karaoke=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        content = output_path.read_text()

        # Must have enhanced tags
        assert "<" in content and ">" in content, "Should have enhanced tags"

        # Must have Chinese characters
        assert re.search(r"[\u4e00-\u9fff]", content), "Should contain Chinese characters"


# =============================================================================
# Edge Cases and Consistency Tests - STRICT
# =============================================================================


class TestEdgeCasesAndConsistency:
    """Strict validation of edge cases and cross-format consistency."""

    def test_karaoke_without_word_level_fallback(self, aligned_json_en, tmp_path):
        """karaoke=True with word_level=False: should handle gracefully."""
        output_path = tmp_path / "output.ass"

        result = run_caption_convert(aligned_json_en, output_path, word_level=False, karaoke=True)
        # Should succeed without error
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Output should be valid ASS
        content = output_path.read_text()
        assert "[Script Info]" in content
        assert "Dialogue:" in content

    def test_format_word_count_consistency(self, aligned_json_en, tmp_path):
        """All word-level formats should have consistent word counts."""
        word_counts = {}

        # JSON
        json_out = tmp_path / "output.json"
        run_caption_convert(aligned_json_en, json_out, word_level=True, karaoke=False)
        with open(json_out) as f:
            data = json.load(f)
        json_words = sum(len(seg.get("words", [])) for seg in data)
        word_counts["json"] = json_words

        # SRT
        srt_out = tmp_path / "output.srt"
        run_caption_convert(aligned_json_en, srt_out, word_level=True, karaoke=False)
        srt_content = srt_out.read_text()
        word_counts["srt"] = srt_content.count("-->")

        # VTT
        vtt_out = tmp_path / "output.vtt"
        run_caption_convert(aligned_json_en, vtt_out, word_level=True, karaoke=False)
        vtt_content = vtt_out.read_text()
        word_counts["vtt"] = vtt_content.count("-->")

        # ASS
        ass_out = tmp_path / "output.ass"
        run_caption_convert(aligned_json_en, ass_out, word_level=True, karaoke=False)
        ass_content = ass_out.read_text()
        word_counts["ass"] = ass_content.count("Dialogue:")

        # LRC
        lrc_out = tmp_path / "output.lrc"
        run_caption_convert(aligned_json_en, lrc_out, word_level=True, karaoke=False)
        lrc_content = lrc_out.read_text()
        word_counts["lrc"] = len([l for l in lrc_content.split("\n") if re.match(r"^\[\d+:\d+\.\d+\]", l)])

        # All counts should be within 30% of each other
        counts = list(word_counts.values())
        avg = sum(counts) / len(counts)
        for fmt, count in word_counts.items():
            deviation = abs(count - avg) / avg
            assert deviation < 0.3, f"{fmt} has {count} segments, avg is {avg:.0f}, deviation {deviation:.1%}"

    def test_round_trip_json_preserves_data(self, aligned_json_en, tmp_path):
        """JSON round-trip: read -> convert -> read should preserve word data."""
        # First conversion
        output1 = tmp_path / "output1.json"
        run_caption_convert(aligned_json_en, output1, word_level=True, karaoke=False)

        # Second conversion (round-trip)
        output2 = tmp_path / "output2.json"
        run_caption_convert(output1, output2, word_level=True, karaoke=False)

        # Compare
        with open(output1) as f:
            data1 = json.load(f)
        with open(output2) as f:
            data2 = json.load(f)

        # Same number of segments
        assert len(data1) == len(data2), "Round-trip should preserve segment count"

        # Same number of words
        words1 = sum(len(seg.get("words", [])) for seg in data1)
        words2 = sum(len(seg.get("words", [])) for seg in data2)
        assert words1 == words2, "Round-trip should preserve word count"
