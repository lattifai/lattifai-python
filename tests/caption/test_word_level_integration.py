"""Integration tests for word-level export across all formats."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats import get_writer
from lattifai.caption.supervision import Supervision
from lattifai.config.caption import CaptionFonts, CaptionStyle, KaraokeConfig


@pytest.fixture
def supervision_with_alignment():
    """Create test supervision with word alignment."""
    return Supervision(
        text="Hello beautiful world",
        start=15.2,
        duration=3.3,
        alignment={
            "word": [
                AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                AlignmentItem(symbol="beautiful", start=15.65, duration=0.75),
                AlignmentItem(symbol="world", start=16.4, duration=2.1),
            ]
        },
    )


class TestWordLevelIntegration:
    """Test word-level export across all supported formats."""

    def test_lrc_writer_registered(self):
        """LRC format should be registered."""
        writer = get_writer("lrc")
        assert writer is not None

    def test_all_formats_support_word_level(self, supervision_with_alignment):
        """All karaoke formats should support word_level parameter."""
        formats = ["lrc", "ass", "ttml"]
        karaoke_config = KaraokeConfig(enabled=True)
        for fmt in formats:
            writer = get_writer(fmt)
            assert writer is not None, f"Format {fmt} not registered"

            # Should not raise TypeError
            result = writer.to_bytes([supervision_with_alignment], word_level=True, karaoke_config=karaoke_config)
            assert isinstance(result, bytes)
            assert len(result) > 0

    def test_custom_karaoke_config(self, supervision_with_alignment):
        """Custom karaoke config should work across formats."""
        style = CaptionStyle(
            primary_color="#FF00FF",
            font_name=CaptionFonts.NOTO_SANS_SC,
        )
        config = KaraokeConfig(
            enabled=True,
            effect="instant",
            style=style,
            lrc_metadata={"ar": "Test Artist"},
        )

        # LRC
        lrc_writer = get_writer("lrc")
        lrc_result = lrc_writer.to_bytes([supervision_with_alignment], word_level=True, karaoke_config=config)
        assert b"[ar:Test Artist]" in lrc_result

        # ASS
        ass_writer = get_writer("ass")
        ass_result = ass_writer.to_bytes([supervision_with_alignment], word_level=True, karaoke_config=config)
        assert b"{\\k" in ass_result  # instant effect uses \k

    def test_graceful_fallback(self):
        """Formats should gracefully handle missing alignment."""
        sup_no_alignment = Supervision(text="No alignment data", start=0.0, duration=1.0)
        karaoke_config = KaraokeConfig(enabled=True)

        for fmt in ["lrc", "ass", "ttml"]:
            writer = get_writer(fmt)
            # Should not raise, should output line-level
            result = writer.to_bytes([sup_no_alignment], word_level=True, karaoke_config=karaoke_config)
            assert b"No alignment data" in result or b"No alignment" in result


class TestKaraokeTimestampBoundary:
    """Test that karaoke word timestamps stay within segment boundaries."""

    @pytest.fixture
    def supervision_with_word_alignment(self):
        """Supervision where word timestamps differ from segment timestamps."""
        # Segment: 10.0 - 15.0 (5s duration)
        # Words: 10.5 - 14.2 (words don't span full segment)
        return Supervision(
            text="Hello beautiful world",
            start=10.0,
            duration=5.0,
            alignment={
                "word": [
                    AlignmentItem(symbol="Hello", start=10.5, duration=0.8),
                    AlignmentItem(symbol="beautiful", start=11.5, duration=1.2),
                    AlignmentItem(symbol="world", start=13.0, duration=1.2),
                ]
            },
        )

    def test_vtt_karaoke_uses_word_timestamps_for_cue(self, supervision_with_word_alignment):
        """VTT karaoke cue timing should use word timestamps, not segment timestamps."""
        import re

        writer = get_writer("vtt")
        karaoke_config = KaraokeConfig(enabled=True)
        result = writer.to_bytes(
            [supervision_with_word_alignment], word_level=True, karaoke_config=karaoke_config
        ).decode("utf-8")

        # Parse cue timestamp line
        ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")
        match = ts_pattern.search(result)
        assert match, f"No timestamp found in VTT output: {result}"

        def parse_ts(ts: str) -> float:
            h, m, rest = ts.split(":")
            s, ms = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

        cue_start = parse_ts(match.group(1))
        cue_end = parse_ts(match.group(2))

        # Cue should use word timestamps (10.5 - 14.2), not segment (10.0 - 15.0)
        assert abs(cue_start - 10.5) < 0.01, f"Cue start {cue_start} should be ~10.5 (first word start)"
        assert abs(cue_end - 14.2) < 0.01, f"Cue end {cue_end} should be ~14.2 (last word end)"

    def test_vtt_karaoke_word_timestamps_within_cue(self, supervision_with_word_alignment):
        """VTT karaoke word timestamps should be within cue boundaries."""
        import re

        writer = get_writer("vtt")
        karaoke_config = KaraokeConfig(enabled=True)
        result = writer.to_bytes(
            [supervision_with_word_alignment], word_level=True, karaoke_config=karaoke_config
        ).decode("utf-8")

        # Parse cue timestamp
        ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")
        cue_match = ts_pattern.search(result)
        assert cue_match

        def parse_ts(ts: str) -> float:
            h, m, rest = ts.split(":")
            s, ms = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

        cue_start = parse_ts(cue_match.group(1))
        cue_end = parse_ts(cue_match.group(2))

        # Parse word timestamps
        word_pattern = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d{3})><c>")
        word_timestamps = [parse_ts(m) for m in word_pattern.findall(result)]

        assert len(word_timestamps) == 3, f"Expected 3 words, got {len(word_timestamps)}"

        for i, word_ts in enumerate(word_timestamps):
            assert word_ts >= cue_start - 0.01, f"Word {i} timestamp {word_ts} < cue start {cue_start}"
            assert word_ts <= cue_end + 0.01, f"Word {i} timestamp {word_ts} > cue end {cue_end}"

    def test_ass_karaoke_uses_word_timestamps_for_event(self, supervision_with_word_alignment):
        """ASS karaoke event timing should use word timestamps, not segment timestamps."""
        import re

        writer = get_writer("ass")
        karaoke_config = KaraokeConfig(enabled=True)
        result = writer.to_bytes(
            [supervision_with_word_alignment], word_level=True, karaoke_config=karaoke_config
        ).decode("utf-8")

        # Parse Dialogue line
        dialogue_pattern = re.compile(r"Dialogue:\s*\d+,(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+)")
        match = dialogue_pattern.search(result)
        assert match, f"No Dialogue found in ASS output: {result}"

        def parse_ass_ts(ts: str) -> float:
            h, m, rest = ts.split(":")
            s, cs = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100

        event_start = parse_ass_ts(match.group(1))
        event_end = parse_ass_ts(match.group(2))

        # Event should use word timestamps (10.5 - 14.2), not segment (10.0 - 15.0)
        assert abs(event_start - 10.5) < 0.02, f"Event start {event_start} should be ~10.5"
        assert abs(event_end - 14.2) < 0.02, f"Event end {event_end} should be ~14.2"

    def test_ass_karaoke_tag_durations_match_word_durations(self, supervision_with_word_alignment):
        """ASS karaoke \\kf tag durations should match individual word durations."""
        import re

        writer = get_writer("ass")
        karaoke_config = KaraokeConfig(enabled=True, effect="sweep")
        result = writer.to_bytes(
            [supervision_with_word_alignment], word_level=True, karaoke_config=karaoke_config
        ).decode("utf-8")

        # Parse \kf durations (in centiseconds)
        kf_pattern = re.compile(r"\\kf(\d+)")
        kf_values = [int(m) for m in kf_pattern.findall(result)]

        assert len(kf_values) == 3, f"Expected 3 \\kf tags, got {len(kf_values)}"

        # Each kf value should match the corresponding word duration
        # Word durations: 0.8s (80cs), 1.2s (120cs), 1.2s (120cs)
        expected_durations = [80, 120, 120]
        for i, (actual, expected) in enumerate(zip(kf_values, expected_durations)):
            assert abs(actual - expected) <= 2, f"Word {i} \\kf{actual} differs from expected {expected}cs"

        # Total should be sum of word durations (not event span)
        total_kf_cs = sum(kf_values)
        expected_total_cs = sum(expected_durations)  # 320cs
        assert (
            abs(total_kf_cs - expected_total_cs) <= 5
        ), f"Total \\kf duration {total_kf_cs}cs differs from expected {expected_total_cs}cs"

    def test_lrc_karaoke_word_timestamps_monotonic(self, supervision_with_word_alignment):
        """LRC karaoke word timestamps should be monotonically increasing."""
        import re

        writer = get_writer("lrc")
        karaoke_config = KaraokeConfig(enabled=True)
        result = writer.to_bytes(
            [supervision_with_word_alignment], word_level=True, karaoke_config=karaoke_config
        ).decode("utf-8")

        # Parse enhanced LRC word timestamps <mm:ss.xx>
        word_pattern = re.compile(r"<(\d+):(\d+)\.(\d+)>")
        matches = word_pattern.findall(result)

        assert len(matches) >= 3, f"Expected at least 3 word timestamps, got {len(matches)}"

        def parse_lrc_ts(m, s, ms):
            ms_val = int(ms) * 10 if len(ms) == 2 else int(ms)
            return int(m) * 60 + int(s) + ms_val / 1000

        prev_ts = -1
        for m, s, ms in matches:
            ts = parse_lrc_ts(m, s, ms)
            assert ts >= prev_ts, f"Word timestamp {ts} not monotonically increasing from {prev_ts}"
            prev_ts = ts

    def test_ttml_karaoke_span_timestamps_within_paragraph(self, supervision_with_word_alignment):
        """TTML karaoke span timestamps should be within paragraph boundaries."""
        import re

        writer = get_writer("ttml")
        karaoke_config = KaraokeConfig(enabled=True)
        result = writer.to_bytes(
            [supervision_with_word_alignment], word_level=True, karaoke_config=karaoke_config
        ).decode("utf-8")

        # Parse paragraph begin/end
        p_pattern = re.compile(r'<p[^>]*begin="([^"]+)"[^>]*end="([^"]+)"')
        p_match = p_pattern.search(result)

        if p_match:

            def parse_ttml_ts(ts: str) -> float:
                # Format: HH:MM:SS.mmm or similar
                parts = ts.replace(",", ".").split(":")
                if len(parts) == 3:
                    h, m, rest = parts
                    s = float(rest)
                    return int(h) * 3600 + int(m) * 60 + s
                return 0

            p_start = parse_ttml_ts(p_match.group(1))
            p_end = parse_ttml_ts(p_match.group(2))

            # Parse span timestamps
            span_pattern = re.compile(r'<span[^>]*begin="([^"]+)"')
            span_starts = [parse_ttml_ts(m) for m in span_pattern.findall(result)]

            for i, span_start in enumerate(span_starts):
                assert span_start >= p_start - 0.01, f"Span {i} start {span_start} < p start {p_start}"
                assert span_start <= p_end + 0.01, f"Span {i} start {span_start} > p end {p_end}"
