"""Test optional dependency imports for each installation extra.

These tests verify that each installation option works correctly:
- pip install lattifai              # base (includes alignment)
- pip install lattifai[transcription] # + ASR models
- pip install lattifai[youtube]     # + yt-dlp
- pip install lattifai[diarization] # + speaker diarization
- pip install lattifai[event]       # + event detection
- pip install lattifai[all]         # transcription + youtube
"""

import importlib

import pytest


def _can_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# =============================================================================
# Base installation (includes alignment)
# =============================================================================


class TestBaseInstall:
    """Tests for base installation: pip install lattifai"""

    def test_dotenv(self):
        """python-dotenv should be available."""
        import dotenv

        assert dotenv is not None

    def test_colorful(self):
        """colorful should be available."""
        import colorful

        assert colorful is not None

    def test_lattifai_errors(self):
        """lattifai.errors should be importable."""
        from lattifai.errors import AlignmentError, ModelLoadError

        assert AlignmentError is not None
        assert ModelLoadError is not None

    def test_lattifai_utils(self):
        """lattifai.utils should be importable."""
        from lattifai.utils import safe_print

        assert safe_print is not None

    def test_captions_package(self):
        """lattifai-captions should be available."""
        from lattifai.caption import Caption, Supervision

        assert Caption is not None
        assert Supervision is not None

    def test_sentence_splitter(self):
        """SentenceSplitter from captions should work."""
        from lattifai.caption import SentenceSplitter

        assert SentenceSplitter is not None

    def test_lattifai_core(self):
        """lattifai-core-hq should be available."""
        from lattifai_core.client import SyncAPIClient

        assert SyncAPIClient is not None

    def test_lhotse(self):
        """lhotse should be available."""
        import lhotse

        assert lhotse is not None

    def test_k2py(self):
        """k2py should be available."""
        import k2

        assert k2 is not None

    def test_onnxruntime(self):
        """onnxruntime should be available."""
        import onnxruntime

        assert onnxruntime is not None

    def test_av(self):
        """av (PyAV) should be available."""
        import av

        assert av is not None

    def test_lattice1_aligner(self):
        """Lattice1Aligner should be importable."""
        from lattifai.alignment import Lattice1Aligner

        assert Lattice1Aligner is not None

    def test_lattice1_worker(self):
        """Lattice1Worker should be importable."""
        from lattifai.alignment.lattice1_worker import Lattice1Worker

        assert Lattice1Worker is not None

    def test_tokenizer(self):
        """tokenize_multilingual_text should be importable."""
        from lattifai.alignment import tokenize_multilingual_text

        assert tokenize_multilingual_text is not None

    def test_client(self):
        """LattifAI client should be importable."""
        from lattifai.client import LattifAI

        assert LattifAI is not None


# =============================================================================
# Transcription installation
# =============================================================================


@pytest.mark.skipif(not _can_import("google.genai"), reason="lattifai[transcription] not installed")
class TestTranscriptionInstall:
    """Tests for transcription installation: pip install lattifai[transcription]"""

    def test_google_genai(self):
        """google-genai should be available."""
        import google.genai

        assert google.genai is not None

    def test_nemo_asr(self):
        """nemo_toolkit_asr should be available."""
        import nemo.collections.asr

        assert nemo.collections.asr is not None

    def test_gemini_transcriber(self):
        """GeminiTranscriber should be importable."""
        from lattifai.transcription.gemini import GeminiTranscriber

        assert GeminiTranscriber is not None

    def test_transcription_config(self):
        """TranscriptionConfig should be importable."""
        from lattifai.config import TranscriptionConfig

        assert TranscriptionConfig is not None


@pytest.mark.skipif(not _can_import("OmniSenseVoice"), reason="OmniSenseVoice not installed")
class TestSenseVoiceInstall:
    """Tests for SenseVoice transcription."""

    def test_omnisensevoice(self):
        """OmniSenseVoice should be available."""
        import OmniSenseVoice

        assert OmniSenseVoice is not None


# =============================================================================
# YouTube installation
# =============================================================================


@pytest.mark.skipif(not _can_import("yt_dlp"), reason="lattifai[youtube] not installed")
class TestYouTubeInstall:
    """Tests for YouTube installation: pip install lattifai[youtube]"""

    def test_yt_dlp(self):
        """yt-dlp should be available."""
        import yt_dlp

        assert yt_dlp is not None

    def test_questionary(self):
        """questionary should be available."""
        import questionary

        assert questionary is not None

    def test_pycryptodome(self):
        """pycryptodome should be available."""
        from Crypto.Cipher import AES

        assert AES is not None

    def test_youtube_loader(self):
        """YoutubeLoader should be importable."""
        from lattifai.youtube.client import YoutubeLoader

        assert YoutubeLoader is not None


# =============================================================================
# Diarization installation
# =============================================================================


@pytest.mark.skipif(not _can_import("pyannote.audio"), reason="lattifai[diarization] not installed")
class TestDiarizationInstall:
    """Tests for diarization installation: pip install lattifai[diarization]"""

    def test_pyannote(self):
        """pyannote-audio should be available."""
        import pyannote.audio

        assert pyannote.audio is not None

    def test_nemo_asr(self):
        """nemo_toolkit_asr should be available."""
        import nemo.collections.asr

        assert nemo.collections.asr is not None

    def test_diarization_config(self):
        """DiarizationConfig should be importable."""
        from lattifai.config import DiarizationConfig

        assert DiarizationConfig is not None


# =============================================================================
# Event detection installation
# =============================================================================


@pytest.mark.skipif(not _can_import("pyannote.audio"), reason="lattifai[event] not installed")
class TestEventInstall:
    """Tests for event installation: pip install lattifai[event]"""

    def test_event_config(self):
        """EventConfig should be importable."""
        from lattifai.config import EventConfig

        assert EventConfig is not None


# =============================================================================
# Full installation summary
# =============================================================================


class TestInstallSummary:
    """Summary test that reports installation status."""

    def test_report_installed_extras(self):
        """Report which extras are installed."""
        extras = {
            "base": ["dotenv", "colorful", "lattifai.caption", "lattifai_core", "k2", "lhotse", "onnxruntime", "av"],
            "transcription": ["google.genai", "OmniSenseVoice", "nemo.collections.asr"],
            "youtube": ["yt_dlp", "questionary", "Crypto"],
            "diarization": ["pyannote.audio", "nemo.collections.asr"],
            "event": ["pyannote.audio"],
        }

        print("\n" + "=" * 60)
        print("LattifAI Installation Status")
        print("=" * 60)

        for extra, modules in extras.items():
            installed = all(_can_import(m) for m in modules)
            status = "✓ installed" if installed else "✗ not installed"
            print(f"  [{extra}]: {status}")

        print("=" * 60)

        # This test always passes - it's just for reporting
        assert True
