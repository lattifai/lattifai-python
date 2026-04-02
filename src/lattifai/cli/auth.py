"""Authentication commands for the LattifAI CLI."""

from __future__ import annotations

import os
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
from lattifai_auth import deobfuscate_key, generate_auth_payload, get_device_info, obfuscate_key
from rich.console import Console
from rich.table import Table

from lattifai.cli.config import clear_auth, get_auth_value, set_auth_value
from lattifai.theme import _Theme as T

console = Console()

DEFAULT_SITE_URL = "https://lattifai.com"
DEFAULT_API_URL = "https://api.lattifai.com/v1"


# ---------------------------------------------------------------------------
# lattifai-auth helpers
# ---------------------------------------------------------------------------


def _obfuscate(key: str) -> str:
    """Obfuscate an API key for device-bound local storage."""
    if not key:
        return key
    return obfuscate_key(key)


def _deobfuscate(raw: Optional[str]) -> Optional[str]:
    """Deobfuscate a stored API key.

    Raises RuntimeError with actionable messages when:
    - v1: ciphertext cannot be decrypted on this device
    - v1: ciphertext is malformed or corrupted
    """
    if not raw:
        return raw
    if not raw.startswith("v1:"):
        return raw  # plaintext — pass through
    try:
        return deobfuscate_key(raw)
    except RuntimeError:
        raise RuntimeError("Stored API key is bound to a different device.\n" "Run:  lai auth login")
    except ValueError:
        raise RuntimeError("Stored API key is malformed or corrupted.\n" "Run:  lai auth login")


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
    """Return an ISO-8601 timestamp in local timezone with UTC offset.

    Stored as local time (e.g., 2026-04-01T23:36:12+08:00) so the raw
    config.toml is human-readable without mental timezone conversion.
    The _format_time() parser handles both Z-suffix and offset formats.
    """
    return datetime.now().astimezone().isoformat()


def _resolve_site_url(site_url: Optional[str]) -> str:
    """Resolve the web site URL (authorization page + code exchange)."""
    return (site_url or os.environ.get("LATTIFAI_SITE_URL") or DEFAULT_SITE_URL).rstrip("/")


def _resolve_api_url(api_url: Optional[str]) -> str:
    """Resolve the backend API URL (whoami + session revoke).

    Strips trailing /v1 if present — endpoint paths already include it.
    This avoids double-prefixing when LATTIFAI_BASE_URL=http://host:8000/v1.
    """
    url = (api_url or os.environ.get("LATTIFAI_BASE_URL") or DEFAULT_API_URL).rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _load_dotenv_value(key: str) -> Optional[str]:
    """Read a value from the nearest .env file without mutating the environment."""
    try:
        from dotenv import dotenv_values, find_dotenv
    except ImportError:
        return None

    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        return None

    value = dotenv_values(dotenv_path).get(key)
    return str(value) if value else None


_dotenv_checked = False


def _migrate_dotenv_to_config() -> None:
    """One-time migration: copy LATTIFAI_API_KEY from .env into config.toml.

    Only runs once per process; subsequent calls are no-ops.
    """
    global _dotenv_checked  # noqa: PLW0603
    if _dotenv_checked:
        return
    _dotenv_checked = True

    dotenv_key = _load_dotenv_value("LATTIFAI_API_KEY")
    if not dotenv_key:
        return
    if get_auth_value("LATTIFAI_API_KEY"):
        console.print(
            f"[{T.RICH_WARN}]Ignoring LATTIFAI_API_KEY in .env — "
            f"using session from config.toml. "
            f"Remove the key from .env to silence this warning.[/{T.RICH_WARN}]"
        )
        return
    set_auth_value("LATTIFAI_API_KEY", _obfuscate(dotenv_key))
    console.print(
        f"[{T.RICH_WARN}]Migrated LATTIFAI_API_KEY from .env to config.toml. "
        f"You can now remove it from .env.[/{T.RICH_WARN}]"
    )


def _resolve_api_key() -> Optional[str]:
    """Resolve API key: explicit env var > config.toml session.

    On first call, migrates .env key to config.toml if no session exists.
    Deobfuscates stored keys when lattifai-auth is available.
    """
    if key := os.environ.get("LATTIFAI_API_KEY"):
        return key
    _migrate_dotenv_to_config()
    return _deobfuscate(get_auth_value("LATTIFAI_API_KEY"))


def _auth_headers(api_key: str) -> dict[str, str]:
    """Build authorization headers with X-Device-Auth HMAC signature."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        headers["X-Device-Auth"] = generate_auth_payload(api_key)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[{T.RICH_DIM}]Device auth header unavailable: {exc}[/{T.RICH_DIM}]")
    return headers


def _request_whoami(api_key: str, api_url: str) -> dict[str, Any]:
    """Fetch current auth metadata from the backend API."""
    with httpx.Client(timeout=15.0) as client:
        response = client.get(f"{api_url}/v1/auth/whoami", headers=_auth_headers(api_key))
        response.raise_for_status()
        return response.json()


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


def _revoke_session(api_key: str, api_url: str) -> dict[str, Any]:
    """Revoke the current API key session via the backend API."""
    with httpx.Client(timeout=15.0) as client:
        response = client.delete(f"{api_url}/v1/auth/session", headers=_auth_headers(api_key))
        response.raise_for_status()
        return response.json()


def _persist_auth(api_key: str, whoami_data: dict[str, Any]) -> None:
    """Persist auth metadata into config.toml."""
    clear_auth()
    set_auth_value("LATTIFAI_API_KEY", _obfuscate(api_key))
    if whoami_data.get("user_email"):
        set_auth_value("USER_EMAIL", whoami_data["user_email"])
    if whoami_data.get("key_name"):
        set_auth_value("KEY_NAME", whoami_data["key_name"])
    set_auth_value("LOGGED_IN_AT", _now_iso())


def _format_time(iso_str: Optional[str], *, future: bool = False) -> str:
    """Format an ISO-8601 timestamp into local time with relative offset.

    Args:
        iso_str: ISO-8601 timestamp (UTC). None or empty returns "".
        future: If True, show "in Xh" instead of "Xh ago" (for expiry times).
    """
    if not iso_str:
        return ""
    try:
        clean = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        now = datetime.now(timezone.utc)
        local_dt = dt.astimezone()  # convert to local timezone

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
                    # Do NOT close the server on state mismatch — keep waiting
                    # for the legitimate browser callback. A local DoS could
                    # otherwise kill the login flow with a single bad request.
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
    resolved_site_url = _resolve_site_url(site_url)
    resolved_api_url = _resolve_api_url(api_url)

    if api_key:
        manual_api_key = typer.prompt("LattifAI API key", hide_input=True).strip()
        if not manual_api_key:
            console.print(f"[{T.RICH_ERR}]API key cannot be empty.[/{T.RICH_ERR}]")
            raise typer.Exit(1)

        try:
            whoami_data = _request_whoami(manual_api_key, resolved_api_url)
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

        # Use exchange response as primary auth data; whoami is optional verification
        whoami_data = {
            "user_email": exchange_data.get("user_email"),
            "key_name": f"CLI: {device['device_name']}",
            "permissions": exchange_data.get("permissions", []),
            "created_at": _now_iso(),
        }
        # Best-effort whoami — may fail if site and backend use different databases
        try:
            whoami_data = _request_whoami(issued_api_key, resolved_api_url)
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
    resolved_api_url = _resolve_api_url(api_url)
    raw_key = get_auth_value("LATTIFAI_API_KEY")
    if not raw_key:
        console.print(f"[{T.RICH_WARN}]No saved CLI session found.[/{T.RICH_WARN}]")
        return

    try:
        api_key = _deobfuscate(raw_key)
    except RuntimeError as exc:
        # Cannot recover key — still clear local auth so user can re-login
        clear_auth()
        console.print(f"[{T.RICH_WARN}]Local auth cleared, but remote revoke not attempted: {exc}[/{T.RICH_WARN}]")
        raise typer.Exit(1)

    if not api_key:
        console.print(f"[{T.RICH_WARN}]No saved CLI session found.[/{T.RICH_WARN}]")
        return

    revoke_error: Optional[Exception] = None
    try:
        response = _revoke_session(api_key, resolved_api_url)
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
    resolved_api_url = _resolve_api_url(api_url)
    try:
        api_key = _resolve_api_key()
    except RuntimeError as exc:
        console.print(f"[{T.RICH_ERR}]{exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc
    if not api_key:
        console.print(f"[{T.RICH_WARN}]Not logged in. Run 'lai auth login' or set LATTIFAI_API_KEY.[/{T.RICH_WARN}]")
        raise typer.Exit(1)

    try:
        whoami_data = _request_whoami(api_key, resolved_api_url)
    except httpx.HTTPError as exc:
        console.print(f"[{T.RICH_ERR}]Failed to fetch session info: {exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc

    _print_session(whoami_data, api_key)


def _persist_trial_auth(data: dict[str, Any]) -> None:
    """Persist trial auth metadata into config.toml."""
    clear_auth()
    set_auth_value("LATTIFAI_API_KEY", _obfuscate(data["api_key"]))
    set_auth_value("IS_TRIAL", True)
    set_auth_value("EXPIRES_AT", data["expires_at"])
    set_auth_value("CREDITS", data.get("credits", 120))
    set_auth_value("LOGGED_IN_AT", _now_iso())


def trial(
    site_url: Optional[str] = typer.Option(
        None,
        "--site-url",
        help="Web site URL. Env: LATTIFAI_SITE_URL.",
    ),
) -> None:
    """Get a free trial API key — no sign-up required."""
    resolved_site_url = _resolve_site_url(site_url)

    # Check for existing valid trial — reuse if not expired (L3)
    try:
        existing_key = _deobfuscate(get_auth_value("LATTIFAI_API_KEY"))
    except RuntimeError:
        existing_key = None  # cannot recover — treat as no existing key
    is_trial = get_auth_value("IS_TRIAL")
    expires_at = get_auth_value("EXPIRES_AT")
    if existing_key and is_trial and expires_at:
        try:
            from datetime import datetime as _dt

            exp = _dt.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp > _dt.now(timezone.utc):
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
