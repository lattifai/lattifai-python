"""Tests for source_lang parameter functionality."""

import inspect
from unittest.mock import AsyncMock, Mock, patch

import pytest

from lattifai import LattifAI
from lattifai.caption import Caption
from lattifai.config import CaptionConfig, TranscriptionConfig
from lattifai.mixin import LattifAIClientMixin
from lattifai.transcription import GeminiTranscriber
from lattifai.workflow.youtube import YouTubeDownloader


class TestSourceLangInCaptionConfig:
    """Test source_lang in CaptionConfig."""

    def test_source_lang_default_none(self):
        """Test that source_lang defaults to None."""
        config = CaptionConfig()
        assert config.source_lang is None

    def test_source_lang_initialization(self):
        """Test source_lang can be set during initialization."""
        config = CaptionConfig(source_lang="en")
        assert config.source_lang == "en"

    @pytest.mark.parametrize(
        "lang_code",
        ["en", "zh", "es", "fr", "de", "ja", "ko", "pt", "ru", "ar"],
    )
    def test_source_lang_various_languages(self, lang_code):
        """Test source_lang with various language codes."""
        config = CaptionConfig(source_lang=lang_code)
        assert config.source_lang == lang_code

    def test_source_lang_with_region_code(self):
        """Test source_lang with region-specific codes."""
        config = CaptionConfig(source_lang="en-US")
        assert config.source_lang == "en-US"

        config = CaptionConfig(source_lang="zh-CN")
        assert config.source_lang == "zh-CN"

    def test_source_lang_can_be_modified(self):
        """Test source_lang can be changed after initialization."""
        config = CaptionConfig()
        assert config.source_lang is None

        config.source_lang = "en"
        assert config.source_lang == "en"

        config.source_lang = "zh"
        assert config.source_lang == "zh"


class TestSourceLangInTranscription:
    """Test source_lang usage in transcription workflow."""

    def test_transcriber_language_parameter(self):
        """Test transcriber receives language parameter."""
        config = TranscriptionConfig(
            model_name="gemini-2.5-pro",
            gemini_api_key="test_key",
            language="en",
        )

        assert config.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_with_source_lang(self):
        """Test transcription with source_lang parameter."""
        config = TranscriptionConfig(
            model_name="gemini-2.5-pro",
            gemini_api_key="test_key",
        )

        transcriber = GeminiTranscriber(config)

        # Mock the transcribe method
        with patch.object(transcriber, "transcribe", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "Test transcription"

            result = await transcriber.transcribe("https://example.com/video", language="en")
            del result  # unused

            # Verify transcribe was called with language parameter
            mock_transcribe.assert_called_once()
            call_kwargs = mock_transcribe.call_args.kwargs
            assert call_kwargs.get("language") == "en"

    @pytest.mark.asyncio
    async def test_transcribe_file_with_source_lang(self):
        """Test file transcription with source_lang parameter."""
        config = TranscriptionConfig(
            model_name="gemini-2.5-pro",
            gemini_api_key="test_key",
        )

        transcriber = GeminiTranscriber(config)

        # Mock the transcribe_file method
        with patch.object(transcriber, "transcribe_file", new_callable=AsyncMock) as mock_transcribe_file:
            mock_transcribe_file.return_value = "Test transcription"

            result = await transcriber.transcribe_file("/path/to/audio.mp3", language="zh")
            del result  # unused

            # Verify transcribe_file was called with language parameter
            mock_transcribe_file.assert_called_once()
            call_kwargs = mock_transcribe_file.call_args.kwargs
            assert call_kwargs.get("language") == "zh"


class TestSourceLangInYouTubeDownload:
    """Test source_lang usage in YouTube download workflow."""

    def test_download_captions_signature_accepts_source_lang(self):
        """Test that download_captions method accepts source_lang parameter."""
        downloader = YouTubeDownloader()

        sig = inspect.signature(downloader.download_captions)
        assert "source_lang" in sig.parameters

    def test_download_captions_source_lang_default_none(self):
        """Test that source_lang defaults to None in download_captions method."""
        downloader = YouTubeDownloader()

        sig = inspect.signature(downloader.download_captions)
        param = sig.parameters["source_lang"]
        assert param.default is None


class TestSourceLangInClientMethods:
    """Test source_lang parameter in client methods."""

    def test_youtube_method_accepts_source_lang(self):
        """Test that youtube() method accepts source_lang parameter."""
        client = LattifAI()

        # Check method signature includes source_lang
        sig = inspect.signature(client.youtube)
        assert "source_lang" in sig.parameters

    def test_youtube_method_source_lang_default_none(self):
        """Test that source_lang defaults to None in youtube() method."""
        client = LattifAI()

        sig = inspect.signature(client.youtube)
        param = sig.parameters["source_lang"]
        assert param.default is None

    def test_transcribe_method_uses_source_lang(self):
        """Test that _transcribe method uses source_lang parameter."""
        mixin = LattifAIClientMixin()

        # Check method signature
        sig = inspect.signature(mixin._transcribe)
        assert "source_lang" in sig.parameters


class TestSourceLangIntegration:
    """Integration tests for source_lang parameter flow."""

    def test_source_lang_config_to_youtube_workflow(self):
        """Test source_lang flows from config to youtube workflow."""
        caption_config = CaptionConfig(source_lang="en")
        client = LattifAI(caption_config=caption_config)

        assert client.caption_config.source_lang == "en"

    def test_source_lang_parameter_overrides_config(self):
        """Test that source_lang parameter overrides config value."""
        # Set default in config
        caption_config = CaptionConfig(source_lang="en")
        client = LattifAI(caption_config=caption_config)

        # Mock all the necessary components
        with (
            patch.object(client, "_download_media_sync") as mock_download_media,
            patch.object(client, "audio_loader") as mock_audio_loader,
            patch.object(client, "_download_or_transcribe_caption") as mock_caption,
            patch.object(client, "alignment") as mock_alignment,
        ):

            mock_download_media.return_value = "/tmp/audio.mp3"
            mock_audio_loader.return_value = Mock()  # Mock AudioData
            mock_caption.return_value = "/tmp/caption.srt"
            mock_alignment.return_value = Caption()

            # Call youtube with different source_lang
            client.youtube(
                url="https://youtube.com/watch?v=test",
                source_lang="zh",  # Override config value
            )

            # Verify the overridden value was used
            mock_caption.assert_called_once()
            call_args = mock_caption.call_args
            # source_lang should be 'zh' (overridden), not 'en' (from config)
            assert call_args[0][4] == "zh"

    def test_source_lang_none_uses_config_default(self):
        """Test that None source_lang uses config default."""
        caption_config = CaptionConfig(source_lang="fr")
        client = LattifAI(caption_config=caption_config)

        with (
            patch.object(client, "_download_media_sync") as mock_download_media,
            patch.object(client, "audio_loader") as mock_audio_loader,
            patch.object(client, "_download_or_transcribe_caption") as mock_caption,
            patch.object(client, "alignment") as mock_alignment,
        ):

            mock_download_media.return_value = "/tmp/audio.mp3"
            mock_audio_loader.return_value = Mock()  # Mock AudioData
            mock_caption.return_value = "/tmp/caption.srt"
            mock_alignment.return_value = Caption()

            # Call youtube without source_lang (should use config)
            client.youtube(
                url="https://youtube.com/watch?v=test",
                # source_lang not provided
            )

            # Verify config value was used
            mock_caption.assert_called_once()
            call_args = mock_caption.call_args
            assert call_args[0][4] == "fr"


class TestSourceLangEdgeCases:
    """Test edge cases for source_lang parameter."""

    def test_empty_string_source_lang(self):
        """Test behavior with empty string source_lang."""
        config = CaptionConfig(source_lang="")
        assert config.source_lang == ""

    def test_source_lang_with_special_characters(self):
        """Test source_lang with special characters."""
        # Some YouTube videos have tracks like 'en-orig', 'en-GB', etc.
        config = CaptionConfig(source_lang="en-orig")
        assert config.source_lang == "en-orig"

    def test_source_lang_case_sensitivity(self):
        """Test that source_lang preserves case."""
        config = CaptionConfig(source_lang="EN")
        assert config.source_lang == "EN"

        config = CaptionConfig(source_lang="en")
        assert config.source_lang == "en"

    @pytest.mark.parametrize(
        "invalid_lang",
        [None, "", "invalid-lang-code-that-does-not-exist"],
    )
    def test_invalid_source_lang_handling(self, invalid_lang):
        """Test that invalid source_lang values don't crash initialization."""
        # Should not raise an error - validation happens at usage time
        config = CaptionConfig(source_lang=invalid_lang)
        assert config.source_lang == invalid_lang


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
