"""Tests for local serve mode helpers and UI assets."""

from lattifai.cli.serve import HTML_FILE, normalize_output_suffix, parse_bool


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
