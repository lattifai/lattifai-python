"""Tests for podcast audio download fallback logic."""

from pathlib import Path

import pytest

import lattifai.podcast.client as podcast_client
from lattifai.podcast.types import EpisodeMetadata


def test_build_audio_download_candidates_blubrry_proxy() -> None:
    url = "https://media.blubrry.com/takeituneasy/ins.blubrry.com/takeituneasy/lex_ai_rick_beato.mp3"

    candidates = podcast_client._build_audio_download_candidates(url)

    assert candidates[0] == url
    assert "https://ins.blubrry.com/takeituneasy/lex_ai_rick_beato.mp3" in candidates


def test_build_audio_download_candidates_non_blubrry() -> None:
    url = "https://example.com/audio/episode.mp3"
    assert podcast_client._build_audio_download_candidates(url) == [url]


def test_build_ytdlp_download_candidates_dedupes_and_prioritizes_source() -> None:
    candidates = podcast_client._build_ytdlp_download_candidates(
        "https://podcasts.apple.com/us/podcast/foo/id123?i=1000",
        [
            "https://example.com/a.mp3",
            "https://example.com/a.mp3",
        ],
    )
    assert candidates == [
        "https://podcasts.apple.com/us/podcast/foo/id123?i=1000",
        "https://example.com/a.mp3",
    ]


def test_download_audio_uses_fallback_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loader = podcast_client.PodcastLoader(cache_dir=str(tmp_path / "cache"))
    episode = EpisodeMetadata(
        audio_url="https://media.blubrry.com/takeituneasy/ins.blubrry.com/takeituneasy/lex_ai_rick_beato.mp3"
    )

    attempts: list[str] = []

    def fake_download(url: str, output_path: Path, timeout: int = 300) -> int:
        attempts.append(url)
        if len(attempts) == 1:
            raise TimeoutError("ssl handshake timeout")
        output_path.write_bytes(b"audio-bytes")
        return len(b"audio-bytes")

    monkeypatch.setattr(podcast_client, "_http_download_to_file", fake_download)

    output = loader.download_audio(episode, output_dir=str(tmp_path / "out"))

    assert Path(output).exists()
    assert len(attempts) == 2
    assert attempts[0].startswith("https://media.blubrry.com/")
    assert attempts[1] == "https://ins.blubrry.com/takeituneasy/lex_ai_rick_beato.mp3"


def test_download_audio_falls_back_to_ytdlp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loader = podcast_client.PodcastLoader(cache_dir=str(tmp_path / "cache"))
    episode = EpisodeMetadata(audio_url="https://example.com/audio/episode.mp3")
    source_url = "https://podcasts.apple.com/us/podcast/foo/id123?i=1000"

    def always_fail_http(url: str, output_path: Path, timeout: int = 300) -> int:
        raise TimeoutError("network timeout")

    ytdlp_attempts: list[str] = []

    def fake_ytdlp(url: str, output_path: Path, timeout: int = 900) -> Path:
        ytdlp_attempts.append(url)
        downloaded = output_path.with_suffix(".m4a")
        downloaded.write_bytes(b"audio-bytes")
        return downloaded

    monkeypatch.setattr(podcast_client, "_http_download_to_file", always_fail_http)
    monkeypatch.setattr(podcast_client, "_download_with_ytdlp", fake_ytdlp)

    output = loader.download_audio(episode, output_dir=str(tmp_path / "out"), source_url=source_url)

    assert Path(output).exists()
    assert output.endswith(".m4a")
    assert ytdlp_attempts[0] == source_url
