"""Authentication commands for the LattifAI CLI."""

from __future__ import annotations

import functools
import secrets
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import typer
from lattifai_auth import get_device_info
from rich.console import Console
from rich.table import Table

from lattifai.auth import (
    deobfuscate,
    load_dotenv_value,
    obfuscate,
    request_whoami,
    resolve_api_key,
    resolve_api_url,
    resolve_site_url,
    revoke_session,
)
from lattifai.cli.config import clear_auth, get_auth_value, set_auth_value
from lattifai.theme import _Theme as T

console = Console()


# ---------------------------------------------------------------------------
# CLI-specific helpers
# ---------------------------------------------------------------------------

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PATH = "/callback"
CALLBACK_TIMEOUT_SECS = 120.0
PORT_RANGE_START = 49152
PORT_RANGE_END = 65535
SUCCESS_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LattifAI Login</title>
    <style>
      body { font-family: sans-serif; margin: 40px; color: #111827; }
      .box { max-width: 560px; padding: 24px; border: 1px solid #d1d5db; border-radius: 12px; }
      h1 { margin-top: 0; font-size: 24px; }
      p { line-height: 1.6; }
    </style>
  </head>
  <body>
    <div class="box">
      <h1>Authorization complete</h1>
      <p>You can close this page and return to the terminal.</p>
    </div>
  </body>
</html>
"""


def _now_iso() -> str:
    """Return an ISO-8601 timestamp in local timezone with UTC offset."""
    return datetime.now().astimezone().isoformat()


@functools.lru_cache(maxsize=1)
def _migrate_dotenv_to_config() -> None:
    """One-time migration: copy LATTIFAI_API_KEY from .env into config.toml."""
    dotenv_key = load_dotenv_value("LATTIFAI_API_KEY")
    if not dotenv_key:
        return
    if get_auth_value("LATTIFAI_API_KEY"):
        console.print(
            f"[{T.RICH_WARN}]Ignoring LATTIFAI_API_KEY in .env — "
            f"using session from config.toml. "
            f"Remove the key from .env to silence this warning.[/{T.RICH_WARN}]"
        )
        return
    set_auth_value("LATTIFAI_API_KEY", obfuscate(dotenv_key))
    console.print(
        f"[{T.RICH_WARN}]Migrated LATTIFAI_API_KEY from .env to config.toml. "
        f"You can now remove it from .env.[/{T.RICH_WARN}]"
    )


def _resolve_api_key() -> Optional[str]:
    """Resolve API key with .env migration for CLI context."""
    _migrate_dotenv_to_config()
    return resolve_api_key()


def _exchange_code(
    code: str,
    state: str,
    device_name: str,
    site_url: str,
    *,
    device_id: Optional[str] = None,
    device_id_source: Optional[str] = None,
) -> dict[str, Any]:
    """Exchange an authorization code for an API key via the web site."""
    payload: dict[str, Any] = {"code": code, "state": state, "device_name": device_name}
    if device_id:
        payload["device_id"] = device_id
    if device_id_source:
        payload["device_id_source"] = device_id_source
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{site_url}/api/cli-auth/exchange", json=payload)
        response.raise_for_status()
        return response.json()


def _persist_auth(api_key: str, whoami_data: dict[str, Any]) -> None:
    """Persist auth metadata into config.toml."""
    clear_auth()
    set_auth_value("LATTIFAI_API_KEY", obfuscate(api_key))
    if whoami_data.get("user_email"):
        set_auth_value("USER_EMAIL", whoami_data["user_email"])
    if whoami_data.get("key_name"):
        set_auth_value("KEY_NAME", whoami_data["key_name"])
    set_auth_value("LOGGED_IN_AT", _now_iso())


def _persist_trial_auth(data: dict[str, Any]) -> None:
    """Persist trial auth metadata into config.toml."""
    clear_auth()
    set_auth_value("LATTIFAI_API_KEY", obfuscate(data["api_key"]))
    set_auth_value("IS_TRIAL", True)
    set_auth_value("EXPIRES_AT", data["expires_at"])
    set_auth_value("CREDITS", data.get("credits", 120))
    set_auth_value("LOGGED_IN_AT", _now_iso())


def _format_time(iso_str: Optional[str], *, future: bool = False) -> str:
    """Format an ISO-8601 timestamp into local time with relative offset."""
    if not iso_str:
        return ""
    try:
        clean = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        now = datetime.now(timezone.utc)
        local_dt = dt.astimezone()

        delta = dt - now if future else now - dt
        secs = int(delta.total_seconds())

        if secs < 0:
            age = "expired" if future else "in the future"
        elif secs < 60:
            age = "just now" if not future else "< 1m"
        elif secs < 3600:
            age = f"{secs // 60}m {'left' if future else 'ago'}"
        elif secs < 86400:
            age = f"{secs // 3600}h {(secs % 3600) // 60}m {'left' if future else 'ago'}"
        else:
            age = f"{secs // 86400}d {'left' if future else 'ago'}"

        return f"{local_dt.strftime('%Y-%m-%d %H:%M')} ({age})"
    except (ValueError, TypeError):
        return iso_str


def _print_session(whoami_data: dict[str, Any], api_key: str, *, is_trial: bool = False) -> None:
    """Render session info using Rich."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style=T.RICH_LABEL, min_width=10)
    table.add_column("Value")

    table.add_row("Account", whoami_data.get("user_email") or f"[{T.RICH_DIM}]anonymous[/{T.RICH_DIM}]")
    table.add_row("Device", whoami_data.get("key_name") or f"...{api_key[-4:]}")
    table.add_row("Since", _format_time(whoami_data.get("created_at")))

    if is_trial:
        expires = whoami_data.get("expires_at")
        credits = whoami_data.get("credits", whoami_data.get("alignment_limit"))
        if credits is not None:
            h, m = divmod(int(credits), 60)
            label = f"{h} hours {m} minutes" if h and m else f"{h} hours" if h else f"{m} minutes"
            table.add_row("Credits", label)
        if expires:
            table.add_row("Expires", _format_time(expires, future=True))

    session_type = "Trial" if is_trial else "Session"
    console.print()
    console.print(f"[{T.RICH_HEADER}]LattifAI {session_type}[/{T.RICH_HEADER}]")
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Local callback server for browser OAuth
# ---------------------------------------------------------------------------


class LocalCallbackServer:
    """Single-use local HTTP server for browser OAuth callbacks."""

    def __init__(self, state: str, timeout: float = CALLBACK_TIMEOUT_SECS):
        self.state = state
        self.timeout = timeout
        self.port: Optional[int] = None
        self.code: Optional[str] = None
        self.error: Optional[str] = None
        self._event = threading.Event()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Bind a random loopback port and start serving in a background thread."""
        self._server = self._create_server()
        self._thread = threading.Thread(target=self._serve, name="lattifai-cli-auth-callback", daemon=True)
        self._thread.start()

    def wait_for_code(self) -> str:
        """Wait for the authorization callback and return the code."""
        if not self._thread:
            raise RuntimeError("Callback server was not started.")

        self._thread.join(self.timeout + 2.0)
        if self._thread.is_alive():
            self.close()
            raise RuntimeError("Timed out waiting for authorization callback.")

        if self.error:
            raise RuntimeError(self.error)
        if not self.code:
            raise RuntimeError("Authorization callback did not return a code.")
        return self.code

    def close(self) -> None:
        """Close the underlying HTTP server."""
        if self._server is not None:
            try:
                self._server.server_close()
            except OSError:
                pass

    def _create_server(self) -> HTTPServer:
        """Create an HTTP server bound to a port in the dynamic range."""

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(handler) -> None:  # noqa: N802
                parsed = urlparse(handler.path)
                if parsed.path != CALLBACK_PATH:
                    handler.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                    return

                params = parse_qs(parsed.query)
                state = params.get("state", [None])[0]
                code = params.get("code", [None])[0]

                if state != self.state:
                    handler.send_error(HTTPStatus.BAD_REQUEST, "State mismatch")
                    return

                if not code:
                    self.error = "Authorization failed: callback code missing."
                    handler.send_error(HTTPStatus.BAD_REQUEST, "Missing code")
                    self._event.set()
                    return

                self.code = code
                payload = SUCCESS_HTML.encode("utf-8")
                handler.send_response(HTTPStatus.OK)
                handler.send_header("Content-Type", "text/html; charset=utf-8")
                handler.send_header("Content-Length", str(len(payload)))
                handler.end_headers()
                handler.wfile.write(payload)
                self._event.set()

            def log_message(handler, format: str, *args: Any) -> None:
                return

        for port in secrets.SystemRandom().sample(range(PORT_RANGE_START, PORT_RANGE_END + 1), 64):
            try:
                server = HTTPServer((CALLBACK_HOST, port), CallbackHandler)
                server.timeout = 0.5
                self.port = port
                return server
            except OSError:
                continue

        raise RuntimeError("Failed to bind a local callback port.")

    def _serve(self) -> None:
        """Serve requests until one callback succeeds or timeout expires."""
        assert self._server is not None

        deadline = time.time() + self.timeout
        while not self._event.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                self.error = "Timed out waiting for authorization callback."
                break
            self._server.timeout = min(0.5, remaining)
            self._server.handle_request()

        self.close()


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def login(
    api_key: bool = typer.Option(
        False,
        "--api-key",
        help="Enter an API key manually instead of opening the browser.",
    ),
    site_url: Optional[str] = typer.Option(
        None,
        "--site-url",
        help="Web site URL (auth page + code exchange). Env: LATTIFAI_SITE_URL.",
    ),
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        help="Backend API URL (whoami + revoke). Env: LATTIFAI_BASE_URL.",
    ),
) -> None:
    """Log in to LattifAI via browser callback or manual API key input."""
    resolved_site_url = resolve_site_url(site_url)
    resolved_api_url = resolve_api_url(api_url)

    if api_key:
        manual_api_key = typer.prompt("LattifAI API key", hide_input=True).strip()
        if not manual_api_key:
            console.print(f"[{T.RICH_ERR}]API key cannot be empty.[/{T.RICH_ERR}]")
            raise typer.Exit(1)

        try:
            whoami_data = request_whoami(manual_api_key, resolved_api_url)
        except httpx.HTTPError as exc:
            console.print(f"[{T.RICH_ERR}]Failed to verify API key: {exc}[/{T.RICH_ERR}]")
            raise typer.Exit(1) from exc

        _persist_auth(manual_api_key, whoami_data)
        console.print(f"[{T.RICH_OK}]Login successful.[/{T.RICH_OK}]")
        _print_session(whoami_data, manual_api_key)

        return

    state = secrets.token_urlsafe(32)
    device = get_device_info()
    callback_server = LocalCallbackServer(state=state)

    try:
        callback_server.start()
    except RuntimeError as exc:
        console.print(f"[{T.RICH_ERR}]{exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc

    auth_params: dict[str, Any] = {
        "state": state,
        "port": callback_server.port,
        "device_name": device["device_name"],
        "device_id": device["device_id"],
    }
    auth_url = f"{resolved_site_url}/cli-auth?" + urlencode(auth_params)

    console.print(f"[{T.RICH_STEP}]Opening browser for LattifAI login...[/{T.RICH_STEP}]")
    console.print(f"[underline cyan]{auth_url}[/underline cyan]")
    try:
        opened = webbrowser.open(auth_url)
    except Exception:
        opened = False

    if not opened:
        console.print(f"[{T.RICH_WARN}]Could not open browser. Open the URL above manually.[/{T.RICH_WARN}]")

    try:
        with console.status(
            f"[{T.RICH_STEP}]Waiting for authorization callback...[/{T.RICH_STEP}]",
            spinner="dots",
        ):
            code = callback_server.wait_for_code()
        exchange_data = _exchange_code(
            code,
            state,
            device["device_name"],
            resolved_site_url,
            device_id=device["device_id"],
            device_id_source=device["device_id_source"],
        )
        issued_api_key = exchange_data.get("api_key")
        if not issued_api_key:
            raise RuntimeError("Code exchange response did not include an API key.")

        whoami_data = {
            "user_email": exchange_data.get("user_email"),
            "key_name": f"CLI: {device['device_name']}",
            "permissions": exchange_data.get("permissions", []),
            "created_at": _now_iso(),
        }
        try:
            whoami_data = request_whoami(issued_api_key, resolved_api_url)
        except httpx.HTTPError:
            pass
    except (RuntimeError, httpx.HTTPError) as exc:
        console.print(f"[{T.RICH_ERR}]Login failed: {exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc
    finally:
        callback_server.close()

    _persist_auth(issued_api_key, whoami_data)
    console.print(f"[{T.RICH_OK}]Login successful.[/{T.RICH_OK}]")
    _print_session(whoami_data, issued_api_key)


def logout(
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        help="Backend API URL. Env: LATTIFAI_BASE_URL.",
    ),
) -> None:
    """Revoke the current session and clear local auth config."""
    resolved_api_url = resolve_api_url(api_url)
    raw_key = get_auth_value("LATTIFAI_API_KEY")
    if not raw_key:
        console.print(f"[{T.RICH_WARN}]No saved CLI session found.[/{T.RICH_WARN}]")
        return

    try:
        api_key = deobfuscate(raw_key)
    except RuntimeError as exc:
        clear_auth()
        console.print(f"[{T.RICH_WARN}]Local auth cleared, but remote revoke not attempted: {exc}[/{T.RICH_WARN}]")
        raise typer.Exit(1)

    if not api_key:
        console.print(f"[{T.RICH_WARN}]No saved CLI session found.[/{T.RICH_WARN}]")
        return

    revoke_error: Optional[Exception] = None
    try:
        response = revoke_session(api_key, resolved_api_url)
    except httpx.HTTPError as exc:
        revoke_error = exc
        response = None
    finally:
        clear_auth()

    if revoke_error:
        console.print(f"[{T.RICH_WARN}]Local auth cleared, but remote revoke failed: {revoke_error}[/{T.RICH_WARN}]")
        raise typer.Exit(1)

    message = (response or {}).get("message", "Session revoked")
    console.print(f"[{T.RICH_OK}]{message}[/{T.RICH_OK}]")


def whoami(
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        help="Backend API URL. Env: LATTIFAI_BASE_URL.",
    ),
) -> None:
    """Display current auth identity."""
    resolved_api_url = resolve_api_url(api_url)
    try:
        api_key = _resolve_api_key()
    except RuntimeError as exc:
        console.print(f"[{T.RICH_ERR}]{exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc
    if not api_key:
        console.print(f"[{T.RICH_WARN}]Not logged in. Run 'lai auth login' or set LATTIFAI_API_KEY.[/{T.RICH_WARN}]")
        raise typer.Exit(1)

    try:
        whoami_data = request_whoami(api_key, resolved_api_url)
    except httpx.HTTPError as exc:
        console.print(f"[{T.RICH_ERR}]Failed to fetch session info: {exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc

    _print_session(whoami_data, api_key)


def trial(
    site_url: Optional[str] = typer.Option(
        None,
        "--site-url",
        help="Web site URL. Env: LATTIFAI_SITE_URL.",
    ),
) -> None:
    """Get a free trial API key — no sign-up required."""
    resolved_site_url = resolve_site_url(site_url)

    # Check for existing valid trial — reuse if not expired (L3)
    try:
        existing_key = deobfuscate(get_auth_value("LATTIFAI_API_KEY"))
    except RuntimeError:
        existing_key = None
    is_trial = get_auth_value("IS_TRIAL")
    expires_at = get_auth_value("EXPIRES_AT")
    if existing_key and is_trial and expires_at:
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp > datetime.now(timezone.utc):
                console.print(f"[{T.RICH_OK}]You already have an active trial (expires {expires_at}).[/{T.RICH_OK}]")
                console.print(f"[{T.RICH_WARN}]Run 'lai auth logout' first to get a new trial.[/{T.RICH_WARN}]")
                return
        except (ValueError, TypeError):
            pass

    # Warn if logged in with a real account
    if existing_key and not is_trial:
        overwrite = typer.confirm("You have an active login session. Replace with trial?", default=False)
        if not overwrite:
            raise typer.Exit(0)

    device = get_device_info()

    try:
        with console.status(
            f"[{T.RICH_STEP}]Requesting trial key...[/{T.RICH_STEP}]",
            spinner="dots",
        ):
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{resolved_site_url}/api/cli-auth/trial",
                    json={
                        "device_name": device["device_name"],
                        "device_id": device["device_id"],
                        "device_id_source": device["device_id_source"],
                    },
                )
    except httpx.HTTPError as exc:
        console.print(f"[{T.RICH_ERR}]Failed to request trial key: {exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc

    if response.status_code == 429:
        detail = response.json().get("detail", "Trial limit exceeded.")
        console.print(f"[{T.RICH_WARN}]{detail}[/{T.RICH_WARN}]")
        raise typer.Exit(1)

    if not response.is_success:
        error = response.json().get("error", response.text)
        console.print(f"[{T.RICH_ERR}]Trial request failed: {error}[/{T.RICH_ERR}]")
        raise typer.Exit(1)

    data = response.json()
    _persist_trial_auth(data)

    console.print(f"[{T.RICH_OK}]Trial activated![/{T.RICH_OK}]")
    _print_session(data, data["api_key"], is_trial=True)

    console.print(f"[{T.RICH_STEP}]Quick start:[/{T.RICH_STEP}]")
    console.print(f"  [{T.RICH_DIM}]# Forced alignment (supports .json .srt .vtt .ass .tsv .TextGrid)[/{T.RICH_DIM}]")
    console.print("  lai alignment align audio.wav caption.srt output.srt")
    console.print()
    console.print(f"  [{T.RICH_DIM}]# YouTube → aligned captions[/{T.RICH_DIM}]")
    console.print("  lai youtube align https://youtu.be/VIDEO_ID -o output.vtt")
    console.print()
    console.print(f"[{T.RICH_DIM}]Upgrade to full access: lai auth login[/{T.RICH_DIM}]")
