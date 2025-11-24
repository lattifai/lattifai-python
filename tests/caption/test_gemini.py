"""Tests for YouTube transcript reader and writer."""

import pytest

from lattifai.caption import GeminiReader, GeminiSegment, GeminiWriter, Supervision

# Sample transcript content for testing
SAMPLE_TRANSCRIPT = """## OpenAI Spring Update: GPT-4o

## Table of Contents
* [00:00:00] Introduction
* [00:53:00] Announcing GPT-4o

## [00:00:00] Introduction

[Music starts] [00:00:08]

[Applause] [00:00:13]

**Mira Murati:** Hi everyone. [00:00:13]

[Applause] [00:00:16]

**Mira Murati:** Hi everyone. Thank you. Thank you. It's great to have you here today. [00:00:19]

Today I'm going to talk about three things. That's it. [00:00:23]

## [00:53:00] Announcing GPT-4o

**Mira Murati:** But the big news today is that we are launching our new flagship model. [00:00:57]

And we are calling it GPT-4o. [00:01:01]

The special thing about GPT-4o is that it brings GPT-4 level intelligence to everyone. [00:01:11]
"""

# Sample YouTube format transcript content for testing
SAMPLE_YOUTUBE_TRANSCRIPT = """Introducing GPT-4o

## Table of Contents
* [[00:12](http://www.youtube.com/watch?v=DQacCB9tDaw&t=12)] Introduction
* [[00:54](http://www.youtube.com/watch?v=DQacCB9tDaw&t=54)] Introducing the New Flagship Model: GPT-4o

## [[00:12](http://www.youtube.com/watch?v=DQacCB9tDaw&t=12)] Introduction

**Mira Murati:** hi everyone Hi everyone thank you thank you it's great to have you here today today I'm going to talk [[00:21](http://www.youtube.com/watch?v=DQacCB9tDaw&t=21)]

about three things that's it we will start with why it's so important to us to have a product that [[00:29](http://www.youtube.com/watch?v=DQacCB9tDaw&t=29)]

we can make freely available and broadly available to everyone and we're always trying to find out ways to reduce [[00:37](http://www.youtube.com/watch?v=DQacCB9tDaw&t=37)]

## [[00:54](http://www.youtube.com/watch?v=DQacCB9tDaw&t=54)] Introducing the New Flagship Model: GPT-4o

**Mira Murati:** that we are launching our new flagship model and we are calling it gbt 40 the special thing about gbt [[01:03](http://www.youtube.com/watch?v=DQacCB9tDaw&t=63)]

40 is that it brings gb4 level intelligence to everyone including our free users we'll be showing some live demos [[01:13](http://www.youtube.com/watch?v=DQacCB9tDaw&t=73)]
"""


class TestGeminiReader:
    """Tests for GeminiReader class (formerly GeminiReader)."""

    def test_read_all_segments(self, tmp_path):
        """Test reading all segments including events and sections."""
        # Create temp file
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Read all segments
        segments = GeminiReader.read(transcript_file, include_events=True, include_sections=True)

        # Should have sections, events, and dialogue
        assert len(segments) > 0

        # Check segment types
        types = {seg.segment_type for seg in segments}
        assert "section_header" in types
        assert "event" in types
        assert "dialogue" in types

    def test_read_dialogue_only(self, tmp_path):
        """Test reading only dialogue segments."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Read dialogue only
        segments = GeminiReader.read(transcript_file, include_events=False, include_sections=False)

        # Should only have dialogue
        types = {seg.segment_type for seg in segments}
        assert types == {"dialogue"}

    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        timestamp = GeminiReader.parse_timestamp("00", "00", "13")
        assert timestamp == 13.0

        timestamp = GeminiReader.parse_timestamp("00", "01", "01")
        assert timestamp == 61.0

        timestamp = GeminiReader.parse_timestamp("01", "00", "00")
        assert timestamp == 3600.0

    def test_speaker_extrevent(self, tmp_path):
        """Test speaker name extrevent."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        segments = GeminiReader.read(transcript_file)

        # Find dialogue segments with speaker
        dialogue_with_speaker = [s for s in segments if s.speaker is not None]
        assert len(dialogue_with_speaker) > 0

        # Check speaker name
        speakers = {s.speaker for s in dialogue_with_speaker}
        assert "Mira Murati:" in speakers

    def test_section_tracking(self, tmp_path):
        """Test section title tracking."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        segments = GeminiReader.read(transcript_file, include_events=True, include_sections=True)

        # Segments should have section information
        sections = {s.section for s in segments if s.section is not None}
        assert "Introduction" in sections
        assert "Announcing GPT-4o" in sections

    def test_extract_for_alignment(self, tmp_path):
        """Test extracting supervisions for alignment."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract for alignment
        supervisions = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Should return Supervision objects
        assert len(supervisions) > 0
        assert all(isinstance(sup, Supervision) for sup in supervisions)

        # Should have text and timestamps
        for sup in supervisions:
            assert sup.text is not None
            assert sup.start >= 0
            assert sup.duration > 0

    def test_extract_with_merge(self, tmp_path):
        """Test extracting with consecutive segment merging."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract without merge
        sups_no_merge = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Extract with merge
        sups_with_merge = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=True)

        # Merged should have fewer or equal segments
        assert len(sups_with_merge) <= len(sups_no_merge)


class TestYouTubeGeminiReader:
    """Tests for GeminiReader with YouTube link format."""

    def test_read_youtube_format(self, tmp_path):
        """Test reading YouTube format transcript with link timestamps."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # Read all segments
        segments = GeminiReader.read(transcript_file, include_events=True, include_sections=True)

        # Should have sections and dialogue
        assert len(segments) > 0

        # Check segment types
        types = {seg.segment_type for seg in segments}
        assert "section_header" in types
        assert "dialogue" in types

    def test_youtube_timestamp_parsing(self):
        """Test YouTube timestamp parsing from URL format."""
        # Test seconds parsing
        timestamp = GeminiReader.parse_timestamp("12")
        assert timestamp == 12.0

        timestamp = GeminiReader.parse_timestamp("63")
        assert timestamp == 63.0

        timestamp = GeminiReader.parse_timestamp("3661")
        assert timestamp == 3661.0

    def test_youtube_section_headers(self, tmp_path):
        """Test YouTube format section headers."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        segments = GeminiReader.read(transcript_file, include_sections=True)

        # Find section headers
        section_headers = [s for s in segments if s.segment_type == "section_header"]
        assert len(section_headers) > 0

        # Check section information
        sections = {s.section for s in segments if s.section is not None}
        assert "Introduction" in sections
        assert "Introducing the New Flagship Model: GPT-4o" in sections

    def test_youtube_speaker_dialogue(self, tmp_path):
        """Test YouTube format speaker dialogue parsing."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        segments = GeminiReader.read(transcript_file)

        # Find dialogue segments with speaker
        dialogue_with_speaker = [s for s in segments if s.speaker is not None]
        assert len(dialogue_with_speaker) > 0

        # Check speaker name
        speakers = {s.speaker for s in dialogue_with_speaker}
        assert "Mira Murati:" in speakers

        # Check timestamps are correctly parsed
        for seg in dialogue_with_speaker:
            if seg.timestamp is not None:
                assert seg.timestamp > 0

    def test_youtube_extract_for_alignment(self, tmp_path):
        """Test extracting YouTube format for alignment."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # Extract for alignment
        supervisions = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Should return Supervision objects
        assert len(supervisions) > 0
        assert all(isinstance(sup, Supervision) for sup in supervisions)

        # Should have text and timestamps
        for sup in supervisions:
            assert sup.text is not None
            assert sup.start >= 0
            assert sup.duration > 0

    def test_youtube_with_merge(self, tmp_path):
        """Test YouTube format with consecutive segment merging."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # Extract without merge
        sups_no_merge = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Extract with merge
        sups_with_merge = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=True)

        # Merged should have fewer or equal segments
        assert len(sups_with_merge) <= len(sups_no_merge)


class TestGeminiWriter:
    """Tests for GeminiWriter class (formerly GeminiWriter)."""

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test various timestamps
        assert GeminiWriter.format_timestamp(13.0) == "[00:00:13]"
        assert GeminiWriter.format_timestamp(61.0) == "[00:01:01]"
        assert GeminiWriter.format_timestamp(3661.0) == "[01:01:01]"
        assert GeminiWriter.format_timestamp(0.0) == "[00:00:00]"

    def test_update_timestamps(self, tmp_path):
        """Test updating transcript with new timestamps."""
        # Create original transcript
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract supervisions
        supervisions = GeminiReader.extract_for_alignment(original_file)

        # Modify timestamps slightly (simulate alignment)
        aligned_supervisions = []
        for sup in supervisions:
            aligned_sup = Supervision(
                id=sup.id,
                text=sup.text,
                start=sup.start + 0.1,  # Add 0.1 second
                duration=sup.duration,
            )
            aligned_supervisions.append(aligned_sup)

        # Update timestamps
        output_file = tmp_path / "updated.txt"
        GeminiWriter.update_timestamps(original_file, aligned_supervisions, output_file)

        # Check output file exists
        assert output_file.exists()

        # Read updated content
        updated_content = output_file.read_text()
        assert len(updated_content) > 0

    def test_write_aligned_transcript(self, tmp_path):
        """Test writing simplified aligned transcript."""
        # Create original transcript
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract and create aligned supervisions
        supervisions = GeminiReader.extract_for_alignment(original_file)

        # Add word-level alignment
        for sup in supervisions:
            words = sup.text.split()
            word_duration = sup.duration / max(len(words), 1)
            word_alignments = []
            for i, word in enumerate(words):
                word_alignments.append(
                    {
                        "symbol": word,
                        "start": sup.start + i * word_duration,
                        "end": sup.start + (i + 1) * word_duration,
                    }
                )
            sup.alignment = {"word": word_alignments}

        # Write aligned transcript
        output_file = tmp_path / "aligned.txt"
        GeminiWriter.write_aligned_transcript(supervisions, output_file, include_word_timestamps=True)

        # Check output
        assert output_file.exists()
        content = output_file.read_text()
        assert "Aligned Transcript" in content
        assert "[00:00:" in content  # Should have timestamps

    def test_write_aligned_without_words(self, tmp_path):
        """Test writing aligned transcript without word timestamps."""
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)

        supervisions = GeminiReader.extract_for_alignment(original_file)

        output_file = tmp_path / "aligned_no_words.txt"
        GeminiWriter.write_aligned_transcript(supervisions, output_file, include_word_timestamps=False)

        assert output_file.exists()
        content = output_file.read_text()

        # Should not contain word-level details
        assert "Words:" not in content


class TestGeminiGeminiSegment:
    """Tests for GeminiSegment dataclass (shared)."""

    def test_segment_creation(self):
        """Test creating a GeminiSegment."""
        segment = GeminiSegment(
            text="Hello world",
            timestamp=13.0,
            speaker="Speaker",
            section="Section 1",
            segment_type="dialogue",
            line_number=10,
        )

        assert segment.text == "Hello world"
        assert segment.timestamp == 13.0
        assert segment.speaker == "Speaker"
        assert segment.section == "Section 1"
        assert segment.segment_type == "dialogue"
        assert segment.line_number == 10

    def test_start_property(self):
        """Test the start property."""
        segment = GeminiSegment(text="Test", timestamp=10.5)
        assert segment.start == 10.5

        segment_no_ts = GeminiSegment(text="Test", timestamp=None)
        assert segment_no_ts.start == 0.0


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete read -> align -> write workflow."""
        # 1. Create transcript
        transcript_file = tmp_path / "Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # 2. Extract for alignment
        supervisions = GeminiReader.extract_for_alignment(transcript_file)
        assert len(supervisions) > 0

        # 3. Simulate alignment (add small corrections)
        aligned_supervisions = []
        for sup in supervisions:
            # Simulate alignment correction
            aligned_sup = Supervision(
                id=sup.id,
                text=sup.text,
                start=sup.start + 0.05,
                duration=sup.duration * 0.95,
            )

            # Add word alignment
            words = sup.text.split()
            word_duration = aligned_sup.duration / max(len(words), 1)
            word_alignments = []
            for i, word in enumerate(words):
                word_alignments.append(
                    {
                        "symbol": word,
                        "start": aligned_sup.start + i * word_duration,
                        "end": aligned_sup.start + (i + 1) * word_duration,
                    }
                )
            aligned_sup.alignment = {"word": word_alignments}
            aligned_supervisions.append(aligned_sup)

        # 4. Write updated transcript
        updated_file = tmp_path / "updated_Gemini.md"
        GeminiWriter.update_timestamps(transcript_file, aligned_supervisions, updated_file)
        assert updated_file.exists()

        # 5. Write simplified aligned transcript
        simple_file = tmp_path / "simple_aligned.txt"
        GeminiWriter.write_aligned_transcript(aligned_supervisions, simple_file, include_word_timestamps=True)
        assert simple_file.exists()

        # Verify content
        simple_content = simple_file.read_text()
        assert "Aligned Transcript" in simple_content
        assert "Words:" in simple_content

    def test_youtube_workflow(self, tmp_path):
        """Test complete workflow with YouTube format transcript."""
        # 1. Create YouTube format transcript
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # 2. Extract for alignment
        supervisions = GeminiReader.extract_for_alignment(transcript_file)
        assert len(supervisions) > 0

        # 3. Simulate alignment (add small corrections)
        aligned_supervisions = []
        for sup in supervisions:
            # Simulate alignment correction
            aligned_sup = Supervision(
                id=sup.id,
                text=sup.text,
                start=sup.start + 0.05,
                duration=sup.duration * 0.95,
            )

            # Add word alignment
            words = sup.text.split()
            word_duration = aligned_sup.duration / max(len(words), 1)
            word_alignments = []
            for i, word in enumerate(words):
                word_alignments.append(
                    {
                        "symbol": word,
                        "start": aligned_sup.start + i * word_duration,
                        "end": aligned_sup.start + (i + 1) * word_duration,
                    }
                )
            aligned_sup.alignment = {"word": word_alignments}
            aligned_supervisions.append(aligned_sup)

        # 4. Write updated transcript
        updated_file = tmp_path / "updated_youtube_Gemini.md"
        GeminiWriter.update_timestamps(transcript_file, aligned_supervisions, updated_file)
        assert updated_file.exists()

        # 5. Write simplified aligned transcript
        simple_file = tmp_path / "simple_youtube_aligned.txt"
        GeminiWriter.write_aligned_transcript(aligned_supervisions, simple_file, include_word_timestamps=True)
        assert simple_file.exists()

        # Verify content
        simple_content = simple_file.read_text()
        assert "Aligned Transcript" in simple_content
        assert "Words:" in simple_content

    def test_mixed_format_compatibility(self, tmp_path):
        """Test that both formats can be processed together."""
        # Test original format
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)
        original_sups = GeminiReader.extract_for_alignment(original_file)

        # Test YouTube format
        youtube_file = tmp_path / "youtube.txt"
        youtube_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)
        youtube_sups = GeminiReader.extract_for_alignment(youtube_file)

        # Both should work and return supervisions
        assert len(original_sups) > 0
        assert len(youtube_sups) > 0

        # Both should have valid supervisions
        for sup in original_sups + youtube_sups:
            assert isinstance(sup, Supervision)
            assert sup.text is not None
            assert sup.start >= 0
            assert sup.duration > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
