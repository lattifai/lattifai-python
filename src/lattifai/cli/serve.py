"""Local web UI server for trying LattifAI features."""

import cgi
import json
import mimetypes
import shutil
import traceback
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse
from uuid import uuid4

import nemo_run as run

from lattifai.cli.caption import convert as caption_convert
from lattifai.cli.transcribe import transcribe as transcribe_run
from lattifai.cli.translate import translate as translate_run
from lattifai.client import LattifAI
from lattifai.config import AlignmentConfig, CaptionConfig, EventConfig, TranscriptionConfig
from lattifai.config.translation import TranslationConfig

HTML_FILE = Path(__file__).with_name("serve.html")
DEFAULT_WORKDIR = Path.cwd() / ".lattifai-serve"

OUTPUT_SUFFIX = {
    "srt": ".srt",
    "vtt": ".vtt",
    "ass": ".ass",
    "json": ".json",
    "textgrid": ".TextGrid",
    "txt": ".txt",
}

ALLOWED_DEVICES = {"auto", "cpu", "cuda", "mps"}
ALLOWED_PROVIDERS = {"gemini", "openai"}
ALLOWED_TRANSLATION_MODES = {"quick", "normal", "refined"}


def parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse a common boolean string value."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_output_suffix(value: str | None, default: str = ".srt") -> str:
    """Normalize output format input to a safe extension string."""
    if value:
        normalized = value.strip().lower()
        if normalized in OUTPUT_SUFFIX:
            return OUTPUT_SUFFIX[normalized]
    if default.startswith("."):
        return default
    return f".{default}"


def normalize_device(value: str | None) -> str:
    """Normalize runtime device string."""
    if not value:
        return "auto"
    normalized = value.strip().lower()
    if normalized in ALLOWED_DEVICES or normalized.startswith("cuda:"):
        return normalized
    return "auto"


class ServeHTTPServer(ThreadingHTTPServer):
    """HTTP server with shared workspace path."""

    def __init__(self, server_address: tuple[str, int], workdir: Path):
        super().__init__(server_address, ServeHandler)
        self.workdir = workdir.resolve()


class ServeHandler(BaseHTTPRequestHandler):
    """Request handler for the local serve mode."""

    server_version = "LattifAIServe/1.0"

    def _server(self) -> ServeHTTPServer:
        return cast(ServeHTTPServer, self.server)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path.startswith("/downloads/"):
            self._serve_download(parsed.path)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        try:
            form = self._load_form_data()
            run_dir = self._new_run_dir()

            if parsed.path == "/api/align":
                output_path = self._run_align(form, run_dir)
            elif parsed.path == "/api/transcribe":
                output_path = self._run_transcribe(form, run_dir)
            elif parsed.path == "/api/convert":
                output_path = self._run_convert(form, run_dir)
            elif parsed.path == "/api/translate":
                output_path = self._run_translate(form, run_dir)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown API endpoint"})
                return

            download_path = f"/downloads/{run_dir.name}/{output_path.name}"
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "output_file": output_path.name,
                    "download_url": download_path,
                },
            )
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "ok": False,
                    "error": str(exc),
                    "details": traceback.format_exc(limit=4),
                },
            )

    def _serve_index(self) -> None:
        if not HTML_FILE.exists():
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"Missing {HTML_FILE.name}"})
            return

        content = HTML_FILE.read_text(encoding="utf-8")
        payload = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_download(self, request_path: str) -> None:
        relative = request_path.removeprefix("/downloads/")
        relative_path = Path(unquote(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "Invalid download path"})
            return

        workdir = self._server().workdir
        target = (workdir / relative_path).resolve()
        if not str(target).startswith(str(workdir)):
            self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "Invalid download path"})
            return
        if not target.exists() or not target.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Output file not found"})
            return

        content_type, _ = mimetypes.guess_type(target.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
        self.end_headers()
        with target.open("rb") as file_obj:
            shutil.copyfileobj(file_obj, self.wfile)

    def _load_form_data(self) -> cgi.FieldStorage:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Request content type must be multipart/form-data")

        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environ,
            keep_blank_values=True,
        )

    def _field_value(self, form: cgi.FieldStorage, name: str, default: str | None = None) -> str | None:
        value = form.getfirst(name)
        if value is None:
            return default
        return str(value)

    def _save_uploaded_file(self, form: cgi.FieldStorage, field_name: str, run_dir: Path, fallback: str) -> Path:
        if field_name not in form:
            raise ValueError(f"Missing required file field: {field_name}")

        field = form[field_name]
        if isinstance(field, list):
            field = field[0]

        filename = getattr(field, "filename", "") or ""
        if not filename:
            raise ValueError(f"Missing uploaded file for field: {field_name}")

        suffix = Path(filename).suffix
        safe_name = Path(filename).name
        target = run_dir / f"{field_name}_{uuid4().hex[:8]}_{safe_name}"
        if not suffix and not safe_name:
            target = run_dir / f"{field_name}_{uuid4().hex[:8]}_{fallback}"

        with target.open("wb") as output:
            shutil.copyfileobj(field.file, output)
        return target

    def _new_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._server().workdir / f"run_{stamp}_{uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _run_align(self, form: cgi.FieldStorage, run_dir: Path) -> Path:
        media_file = self._save_uploaded_file(form, "media_file", run_dir, "media.bin")
        caption_file = self._save_uploaded_file(form, "caption_file", run_dir, "caption.srt")

        output_suffix = normalize_output_suffix(self._field_value(form, "output_format"), default=".srt")
        output_path = run_dir / f"{media_file.stem}_aligned{output_suffix}"

        split_sentence = parse_bool(self._field_value(form, "split_sentence"), default=True)
        word_level = parse_bool(self._field_value(form, "word_level"), default=False)
        device = normalize_device(self._field_value(form, "device"))

        client = LattifAI(
            alignment_config=AlignmentConfig(device=device),
            caption_config=CaptionConfig(split_sentence=split_sentence, word_level=word_level),
        )
        client.alignment(
            input_media=str(media_file),
            input_caption=str(caption_file),
            output_caption_path=str(output_path),
            split_sentence=split_sentence,
        )
        return output_path

    def _run_transcribe(self, form: cgi.FieldStorage, run_dir: Path) -> Path:
        media_file = self._save_uploaded_file(form, "media_file", run_dir, "media.bin")

        output_suffix = normalize_output_suffix(self._field_value(form, "output_format"), default=".srt")
        output_path = run_dir / f"{media_file.stem}_transcript{output_suffix}"

        model_name = (self._field_value(form, "model_name", "gemini-2.5-flash") or "gemini-2.5-flash").strip()
        device = normalize_device(self._field_value(form, "device"))
        gemini_api_key = (self._field_value(form, "gemini_api_key") or "").strip() or None
        api_base_url = (self._field_value(form, "api_base_url") or "").strip() or None
        language = (self._field_value(form, "language") or "").strip() or None

        transcription_config = TranscriptionConfig(
            model_name=model_name,
            device=device,
            gemini_api_key=gemini_api_key,
            api_base_url=api_base_url,
            language=language,
        )
        transcribe_run(
            input=str(media_file),
            output_caption=str(output_path),
            transcription=transcription_config,
            event=EventConfig(enabled=False),
        )
        return output_path

    def _run_convert(self, form: cgi.FieldStorage, run_dir: Path) -> Path:
        caption_file = self._save_uploaded_file(form, "caption_file", run_dir, "caption.srt")

        output_suffix = normalize_output_suffix(self._field_value(form, "output_format"), default=".vtt")
        output_path = run_dir / f"{caption_file.stem}_converted{output_suffix}"

        include_speaker = parse_bool(self._field_value(form, "include_speaker"), default=False)
        normalize_text = parse_bool(self._field_value(form, "normalize_text"), default=False)
        word_level = parse_bool(self._field_value(form, "word_level"), default=False)
        karaoke = parse_bool(self._field_value(form, "karaoke"), default=False)
        if karaoke:
            word_level = True

        caption_convert(
            input_path=str(caption_file),
            output_path=str(output_path),
            include_speaker_in_text=include_speaker,
            normalize_text=normalize_text,
            word_level=word_level,
            karaoke=karaoke,
        )
        return output_path

    def _run_translate(self, form: cgi.FieldStorage, run_dir: Path) -> Path:
        caption_file = self._save_uploaded_file(form, "caption_file", run_dir, "caption.srt")

        provider = (self._field_value(form, "provider", "gemini") or "gemini").strip().lower()
        if provider not in ALLOWED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")

        mode = (self._field_value(form, "mode", "normal") or "normal").strip().lower()
        if mode not in ALLOWED_TRANSLATION_MODES:
            raise ValueError(f"Unsupported translation mode: {mode}")

        target_lang = (self._field_value(form, "target_lang", "zh") or "zh").strip()
        output_suffix = normalize_output_suffix(
            self._field_value(form, "output_format"),
            default=caption_file.suffix or ".srt",
        )
        output_path = run_dir / f"{caption_file.stem}_{target_lang}{output_suffix}"

        default_model = "gemini-3-flash-preview" if provider == "gemini" else "gpt-4.1-mini"
        model_name = (self._field_value(form, "model_name", default_model) or default_model).strip()
        api_key = (self._field_value(form, "api_key") or "").strip() or None
        api_base_url = (self._field_value(form, "api_base_url") or "").strip() or None
        bilingual = parse_bool(self._field_value(form, "bilingual"), default=False)

        translation_config = TranslationConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            api_base_url=api_base_url,
            target_lang=target_lang,
            mode=mode,
            bilingual=bilingual,
            ask_refine_after_normal=False,
        )
        translate_run(
            input=str(caption_file),
            output=str(output_path),
            translation=translation_config,
        )
        return output_path

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _build_browser_url(host: str, port: int) -> str:
    if host == "0.0.0.0":
        return f"http://127.0.0.1:{port}/"
    return f"http://{host}:{port}/"


@run.cli.entrypoint(name="run", namespace="serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    workdir: str = str(DEFAULT_WORKDIR),
    open_browser: bool = True,
):
    """Start the local LattifAI web demo server."""
    if not HTML_FILE.exists():
        raise FileNotFoundError(f"Missing UI file: {HTML_FILE}")

    workspace = Path(workdir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    server = ServeHTTPServer((host, port), workspace)
    url = _build_browser_url(host, port)

    print(f"LattifAI serve is running at {url}")
    print(f"Workspace: {workspace}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open(url, new=2)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("LattifAI serve stopped.")


def main() -> None:
    """Entry point for lai-serve command."""
    run.cli.main(serve)


if __name__ == "__main__":
    main()
