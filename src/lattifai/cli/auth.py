"""Authentication commands for the LattifAI CLI."""

from __future__ import annotations

import os
import secrets
import socket
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
from rich.console import Console
from rich.table import Table

from lattifai.cli.config import clear_auth, get_auth_value, set_auth_value
from lattifai.theme import _Theme as T

console = Console()

DEFAULT_SITE_URL = "https://lattifai.com"
DEFAULT_API_URL = "https://api.lattifai.com/v1"

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PATH = "/callback"
CALLBACK_TIMEOUT_SECS = 120.0
PORT_RANGE_START = 49152
PORT_RANGE_END = 65535
SUCCESS_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
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


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _resolve_api_key() -> Optional[str]:
    """Resolve API key with the same precedence as the client config."""
    return (
        os.environ.get("LATTIFAI_API_KEY")
        or _load_dotenv_value("LATTIFAI_API_KEY")
        or get_auth_value("lattifai_api_key")
    )


def _auth_headers(api_key: str) -> dict[str, str]:
    """Build authorization headers."""
    return {"Authorization": f"Bearer {api_key}"}


def _request_whoami(api_key: str, api_url: str) -> dict[str, Any]:
    """Fetch current auth metadata from the backend API."""
    with httpx.Client(timeout=15.0) as client:
        response = client.get(f"{api_url}/v1/auth/whoami", headers=_auth_headers(api_key))
        response.raise_for_status()
        return response.json()


def _exchange_code(code: str, state: str, device_name: str, site_url: str) -> dict[str, Any]:
    """Exchange an authorization code for an API key via the web site."""
    payload = {"code": code, "state": state, "device_name": device_name}
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
    set_auth_value("lattifai_api_key", api_key)
    if whoami_data.get("user_email"):
        set_auth_value("user_email", whoami_data["user_email"])
    if whoami_data.get("key_name"):
        set_auth_value("key_name", whoami_data["key_name"])
    set_auth_value("logged_in_at", _utc_now_iso())


def _print_whoami_table(whoami_data: dict[str, Any], api_key: str) -> None:
    """Render whoami output using Rich."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style=T.RICH_LABEL, min_width=14)
    table.add_column("Value")
    table.add_row("User", whoami_data.get("user_email") or f"[{T.RICH_DIM}]unknown[/{T.RICH_DIM}]")
    table.add_row("Key", whoami_data.get("key_name") or f"...{api_key[-4:]}")
    table.add_row("Created", str(whoami_data.get("created_at", "")))

    console.print()
    console.print(f"[{T.RICH_HEADER}]LattifAI Session[/{T.RICH_HEADER}]")
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
        _print_whoami_table(whoami_data, manual_api_key)
        return

    state = secrets.token_urlsafe(32)
    device_name = socket.gethostname() or "unknown-device"
    callback_server = LocalCallbackServer(state=state)

    try:
        callback_server.start()
    except RuntimeError as exc:
        console.print(f"[{T.RICH_ERR}]{exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc

    auth_url = f"{resolved_site_url}/en/cli-auth?" + urlencode(
        {"state": state, "port": callback_server.port, "device_name": device_name}
    )

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
        exchange_data = _exchange_code(code, state, device_name, resolved_site_url)
        issued_api_key = exchange_data.get("api_key")
        if not issued_api_key:
            raise RuntimeError("Code exchange response did not include an API key.")

        # Use exchange response as primary auth data; whoami is optional verification
        whoami_data = {
            "user_email": exchange_data.get("user_email"),
            "key_name": f"CLI: {device_name}",
            "permissions": exchange_data.get("permissions", []),
            "created_at": _utc_now_iso(),
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
    _print_whoami_table(whoami_data, issued_api_key)


def logout(
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        help="Backend API URL. Env: LATTIFAI_BASE_URL.",
    ),
) -> None:
    """Revoke the current session and clear local auth config."""
    resolved_api_url = _resolve_api_url(api_url)
    api_key = get_auth_value("lattifai_api_key")
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
    api_key = _resolve_api_key()
    if not api_key:
        console.print(f"[{T.RICH_WARN}]No API key found in env, .env, or config.toml [auth].[/{T.RICH_WARN}]")
        raise typer.Exit(1)

    try:
        whoami_data = _request_whoami(api_key, resolved_api_url)
    except httpx.HTTPError as exc:
        console.print(f"[{T.RICH_ERR}]Failed to fetch session info: {exc}[/{T.RICH_ERR}]")
        raise typer.Exit(1) from exc

    _print_whoami_table(whoami_data, api_key)
