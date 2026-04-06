"""Tests for local serve mode helpers, HTTP endpoints, and UI assets."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The cli package re-exports `serve` as a function, so `lattifai.cli.serve`
# resolves to the function rather than the module.  We grab the real module
# from sys.modules so that `patch.object` works correctly.
from lattifai.cli.serve import (
    HTML_FILE,
    ServeHTTPServer,
    _build_browser_url,
    normalize_device,
    normalize_output_suffix,
    parse_bool,
)

_serve_mod = sys.modules["lattifai.cli.serve"]


# ---------------------------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------------------------


def test_parse_bool_variants() -> None:
    assert parse_bool("true") is True
    assert parse_bool("YES") is True
    assert parse_bool("0") is False
    assert parse_bool("off") is False
    assert parse_bool("unknown", default=True) is True


def test_normalize_output_suffix() -> None:
    assert normalize_output_suffix("srt") == ".srt"
    assert normalize_output_suffix("textgrid") == ".TextGrid"
    assert normalize_output_suffix("invalid", default=".vtt") == ".vtt"
    assert normalize_output_suffix(None, default="json") == ".json"


def test_serve_html_has_four_tabs_and_routes() -> None:
    html = HTML_FILE.read_text(encoding="utf-8")

    for tab in ("align", "transcribe", "translate", "convert"):
        assert f'data-tab="{tab}"' in html
        assert f'id="tab-{tab}"' in html

    for endpoint in ("/api/align", "/api/transcribe", "/api/translate", "/api/convert"):
        assert endpoint in html


class TestNormalizeDevice:
    def test_none_returns_auto(self) -> None:
        assert normalize_device(None) == "auto"

    def test_empty_returns_auto(self) -> None:
        assert normalize_device("") == "auto"

    def test_known_devices(self) -> None:
        assert normalize_device("cpu") == "cpu"
        assert normalize_device("cuda") == "cuda"
        assert normalize_device("mps") == "mps"
        assert normalize_device("auto") == "auto"

    def test_cuda_index(self) -> None:
        assert normalize_device("cuda:0") == "cuda:0"
        assert normalize_device("cuda:1") == "cuda:1"

    def test_case_insensitive(self) -> None:
        assert normalize_device("CPU") == "cpu"
        assert normalize_device("CUDA") == "cuda"

    def test_invalid_returns_auto(self) -> None:
        assert normalize_device("tpu") == "auto"
        assert normalize_device("gpu") == "auto"


class TestBuildBrowserUrl:
    def test_wildcard_address(self) -> None:
        assert _build_browser_url("0.0.0.0", 8765) == "http://127.0.0.1:8765/"

    def test_localhost(self) -> None:
        assert _build_browser_url("localhost", 9000) == "http://localhost:9000/"

    def test_custom_host(self) -> None:
        assert _build_browser_url("192.168.1.10", 3000) == "http://192.168.1.10:3000/"


# ---------------------------------------------------------------------------
# 2. Server fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def serve_server(tmp_path: Path):
    """Start a ServeHTTPServer on a random port and tear it down after the test."""
    server = ServeHTTPServer(("127.0.0.1", 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _url(server: ServeHTTPServer, path: str = "/") -> str:
    host, port = server.server_address
    return f"http://{host}:{port}{path}"


def _get(server: ServeHTTPServer, path: str = "/") -> urllib.request.Request:
    return urllib.request.Request(_url(server, path))


def _multipart_body(fields: dict[str, str | tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """Build a multipart/form-data body.

    Values are either plain strings or (filename, data, content_type) tuples.
    Returns (body_bytes, content_type_header).
    """
    boundary = "----TestBoundary7MA4YWxkTrZu0gW"
    parts: list[bytes] = []
    for name, value in fields.items():
        if isinstance(value, tuple):
            filename, data, ct = value
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {ct}\r\n\r\n".encode() + data + b"\r\n"
            )
        else:
            parts.append(
                f"--{boundary}\r\n" f'Content-Disposition: form-data; name="{name}"\r\n\r\n' f"{value}\r\n".encode()
            )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _post_multipart(
    server: ServeHTTPServer,
    path: str,
    fields: dict[str, str | tuple[str, bytes, str]],
) -> tuple[int, dict]:
    """Send a multipart POST and return (status_code, json_body)."""
    body, content_type = _multipart_body(fields)
    req = urllib.request.Request(
        _url(server, path),
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# 3. Server lifecycle
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    def test_server_starts_and_responds(self, serve_server: ServeHTTPServer) -> None:
        with urllib.request.urlopen(_url(serve_server)) as resp:
            assert resp.status == 200


# ---------------------------------------------------------------------------
# 4. GET endpoints
# ---------------------------------------------------------------------------


class TestGetEndpoints:
    def test_index_returns_html(self, serve_server: ServeHTTPServer) -> None:
        with urllib.request.urlopen(_url(serve_server)) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers["Content-Type"]
            body = resp.read().decode()
            assert "LattifAI" in body or "lattifai" in body.lower()

    def test_unknown_path_returns_404(self, serve_server: ServeHTTPServer) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(_url(serve_server, "/unknown"))
        assert exc_info.value.code == 404
        data = json.loads(exc_info.value.read())
        assert data["ok"] is False

    def test_download_existing_file(self, serve_server: ServeHTTPServer) -> None:
        run_dir = serve_server.workdir / "run_xxx"
        run_dir.mkdir()
        test_file = run_dir / "file.srt"
        test_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

        with urllib.request.urlopen(_url(serve_server, "/downloads/run_xxx/file.srt")) as resp:
            assert resp.status == 200
            assert b"Hello" in resp.read()

    def test_download_path_traversal_blocked(self, serve_server: ServeHTTPServer) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(_url(serve_server, "/downloads/../etc/passwd"))
        assert exc_info.value.code == 403

    def test_download_nonexistent_returns_404(self, serve_server: ServeHTTPServer) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(_url(serve_server, "/downloads/no_such_file.txt"))
        assert exc_info.value.code == 404


# ---------------------------------------------------------------------------
# 5. POST endpoints (mocked external dependencies)
# ---------------------------------------------------------------------------


class TestPostAlign:
    @patch.object(_serve_mod, "LattifAI")
    def test_align_success(self, mock_cls: MagicMock, serve_server: ServeHTTPServer) -> None:
        mock_client = mock_cls.return_value

        # alignment() writes output; simulate by creating the file in the side effect
        def fake_alignment(**kwargs):
            Path(kwargs["output_caption_path"]).write_text("aligned output")

        mock_client.alignment.side_effect = fake_alignment

        status, data = _post_multipart(
            serve_server,
            "/api/align",
            {
                "media_file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav"),
                "caption_file": ("test.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHi\n", "application/x-subrip"),
                "output_format": "srt",
            },
        )
        assert status == 200
        assert data["ok"] is True
        assert "download_url" in data
        assert data["output_file"].endswith(".srt")


class TestPostTranscribe:
    @patch.object(_serve_mod, "transcribe_run")
    def test_transcribe_success(self, mock_transcribe: MagicMock, serve_server: ServeHTTPServer) -> None:
        def fake_transcribe(**kwargs):
            Path(kwargs["output_caption"]).write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

        mock_transcribe.side_effect = fake_transcribe

        status, data = _post_multipart(
            serve_server,
            "/api/transcribe",
            {
                "media_file": ("test.mp3", b"\xff\xfb\x90" + b"\x00" * 100, "audio/mpeg"),
                "output_format": "srt",
            },
        )
        assert status == 200
        assert data["ok"] is True
        assert "download_url" in data


class TestPostConvert:
    @patch.object(_serve_mod, "caption_convert")
    def test_convert_success(self, mock_convert: MagicMock, serve_server: ServeHTTPServer) -> None:
        def fake_convert(**kwargs):
            Path(kwargs["output_path"]).write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n")

        mock_convert.side_effect = fake_convert

        status, data = _post_multipart(
            serve_server,
            "/api/convert",
            {
                "caption_file": ("test.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHi\n", "application/x-subrip"),
                "output_format": "vtt",
            },
        )
        assert status == 200
        assert data["ok"] is True
        assert data["output_file"].endswith(".vtt")


class TestPostTranslate:
    @patch.object(_serve_mod, "translate_run")
    def test_translate_success(self, mock_translate: MagicMock, serve_server: ServeHTTPServer) -> None:
        def fake_translate(**kwargs):
            Path(kwargs["output"]).write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n")

        mock_translate.side_effect = fake_translate

        status, data = _post_multipart(
            serve_server,
            "/api/translate",
            {
                "caption_file": ("test.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHello\n", "application/x-subrip"),
                "target_lang": "zh",
                "model_name": "gemini-3-flash-preview",
                "mode": "normal",
            },
        )
        assert status == 200
        assert data["ok"] is True
        assert "download_url" in data


# ---------------------------------------------------------------------------
# 6. POST 404 for unknown endpoints
# ---------------------------------------------------------------------------


class TestPostUnknownEndpoints:
    def test_post_unknown_api(self, serve_server: ServeHTTPServer) -> None:
        status, data = _post_multipart(
            serve_server,
            "/api/unknown",
            {"caption_file": ("test.srt", b"data", "application/x-subrip")},
        )
        assert status == 404
        assert data["ok"] is False

    def test_post_non_api_path(self, serve_server: ServeHTTPServer) -> None:
        status, data = _post_multipart(
            serve_server,
            "/random",
            {"caption_file": ("test.srt", b"data", "application/x-subrip")},
        )
        assert status == 404
        assert data["ok"] is False


# ---------------------------------------------------------------------------
# 7. Parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    @patch.object(_serve_mod, "translate_run")
    def test_translate_unknown_provider_falls_back_to_gemini(
        self, mock_translate: MagicMock, serve_server: ServeHTTPServer
    ) -> None:
        """Unknown provider hint gracefully falls back to gemini default model."""

        def fake_translate(**kwargs):
            Path(kwargs["output"]).write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n")

        mock_translate.side_effect = fake_translate

        status, data = _post_multipart(
            serve_server,
            "/api/translate",
            {
                "caption_file": ("test.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHi\n", "application/x-subrip"),
                "provider": "anthropic",
                "target_lang": "zh",
            },
        )
        assert status == 200
        assert data["ok"] is True

    @patch.object(_serve_mod, "translate_run")
    def test_translate_invalid_mode(self, mock_translate: MagicMock, serve_server: ServeHTTPServer) -> None:
        status, data = _post_multipart(
            serve_server,
            "/api/translate",
            {
                "caption_file": ("test.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHi\n", "application/x-subrip"),
                "mode": "ultra",
                "target_lang": "zh",
            },
        )
        assert status == 400
        assert data["ok"] is False
        assert "mode" in data["error"].lower() or "Unsupported" in data["error"]

    def test_missing_required_file_field(self, serve_server: ServeHTTPServer) -> None:
        status, data = _post_multipart(
            serve_server,
            "/api/align",
            {
                "media_file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav"),
                # missing caption_file
            },
        )
        assert status == 400
        assert data["ok"] is False
        assert "caption_file" in data["error"]

    def test_non_multipart_request(self, serve_server: ServeHTTPServer) -> None:
        req = urllib.request.Request(
            _url(serve_server, "/api/align"),
            data=b'{"media_file": "test.wav"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            status = exc.code
            data = json.loads(exc.read())

        assert status == 400
        assert data["ok"] is False
        assert "multipart" in data["error"].lower()
