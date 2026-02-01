"""Tests for extended Caption class with transcription/alignment/diarization support."""

import pytest

from lattifai.caption import AlignmentItem, Supervision
from lattifai.data import Caption


class TestExtendedCaption:
    """Test extended Caption fields and methods."""

    def test_caption_has_extended_fields(self):
        """Test that extended Caption has transcription, alignments, etc."""
        c = Caption()
        assert hasattr(c, "transcription")
        assert hasattr(c, "alignments")
        assert hasattr(c, "audio_events")
        assert hasattr(c, "speaker_diarization")

    def test_caption_with_transcription(self):
        """Test creating Caption with transcription data."""
        transcription = [
            Supervision(text="hello", start=0, duration=1),
            Supervision(text="world", start=1, duration=1),
        ]
        c = Caption(transcription=transcription)
        assert len(c.transcription) == 2
        assert c.transcription[0].text == "hello"

    def test_caption_with_alignments(self):
        """Test creating Caption with alignments data."""
        alignments = [
            Supervision(text="hello", start=0.1, duration=0.9),
            Supervision(text="world", start=1.0, duration=0.8),
        ]
        c = Caption(alignments=alignments)
        assert len(c.alignments) == 2
        assert c.alignments[0].start == 0.1

    def test_caption_len_prefers_supervisions(self):
        """Test __len__ prefers supervisions over transcription."""
        c = Caption(
            supervisions=[Supervision(text="a", start=0, duration=1)],
            transcription=[
                Supervision(text="b", start=0, duration=1),
                Supervision(text="c", start=1, duration=1),
            ],
        )
        assert len(c) == 1  # supervisions count

    def test_caption_len_falls_back_to_transcription(self):
        """Test __len__ falls back to transcription when supervisions empty."""
        c = Caption(
            transcription=[
                Supervision(text="a", start=0, duration=1),
                Supervision(text="b", start=1, duration=1),
            ],
        )
        assert len(c) == 2  # transcription count

    def test_from_transcription_results(self):
        """Test creating Caption from transcription results."""
        transcription = [Supervision(text="test", start=0, duration=1)]
        c = Caption.from_transcription_results(
            transcription=transcription,
            language="en",
        )
        assert len(c.transcription) == 1
        assert c.kind == "transcription"
        assert c.source_format == "asr"
        assert c.language == "en"

    def test_caption_with_margins_uses_alignments(self):
        """Test that with_margins prefers alignments over supervisions."""
        original_seg = Supervision(id="1", start=0.0, duration=5.0, text="original")
        aligned_seg = Supervision(
            id="1",
            start=1.0,
            duration=2.0,
            text="aligned",
            alignment={
                "word": [
                    AlignmentItem(symbol="aligned", start=1.1, duration=0.5, score=0.95),
                ]
            },
        )

        caption = Caption(supervisions=[original_seg], alignments=[aligned_seg])
        adjusted = caption.with_margins(start_margin=0.05, end_margin=0.10)

        # Should use aligned_seg, not original_seg
        assert adjusted.supervisions[0].text == "aligned"
        assert adjusted.supervisions[0].start == pytest.approx(1.05, abs=0.001)

    def test_write_prefers_alignments(self, tmp_path):
        """Test that write() uses alignments when available."""
        sup = Supervision(text="original", start=0, duration=1)
        aligned = Supervision(text="aligned version", start=0.1, duration=0.9)

        c = Caption(supervisions=[sup], alignments=[aligned])
        output = tmp_path / "test.srt"
        c.write(output)

        content = output.read_text()
        assert "aligned version" in content
        assert "original" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
