"""Test _transcribe saves raw string for Gemini.

Regression test for PR #32: TypeError when using Gemini transcription with output_dir.
Bug: mixin.py:468-470 passes Caption object to write() which expects str.
"""

import os
import tempfile
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestTranscribeGeminiSave:
    """Real API test for Gemini transcription save."""

    def test_gemini_transcribe_with_output_dir(self):
        """Ensure Gemini transcription saves without TypeError.

        This test verifies PR #32 fix: when using Gemini transcription with
        output_dir parameter, the raw transcription string should be saved
        before it's converted to Caption object.
        """
        from lattifai.client import LattifAI
        from lattifai.config import TranscriptionConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            client = LattifAI(
                transcription_config=TranscriptionConfig(
                    model_name="gemini-2.5-pro",
                    gemini_api_key=os.environ["GEMINI_API_KEY"],
                )
            )
            # Use test audio
            audio = Path(__file__).parent.parent / "data" / "SA1.wav"
            output_dir = Path(tmpdir)

            # This would trigger TypeError: data must be str, not Caption
            # before the PR #32 fix
            result = client._transcribe(
                media_file=audio,
                source_lang="en",
                output_dir=output_dir,
            )

            # Verify result is returned
            assert result is not None

            # Verify transcript file was saved
            transcript_files = list(output_dir.glob("*.md"))
            assert len(transcript_files) == 1, f"Expected 1 .md file, found: {list(output_dir.iterdir())}"

            # Verify file content is valid
            content = transcript_files[0].read_text()
            assert len(content) > 0, "Transcript file should not be empty"
