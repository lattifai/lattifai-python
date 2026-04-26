"""Custom lai CLI entry point that extends nemo_run with direct commands.

This wraps nemo_run's Typer app and registers doctor/update as top-level
commands (no namespace subcommand required).
"""

import importlib.metadata

import typer
from nemo_run.cli.api import create_cli

# Subcommands that hit the LattifAI backend and therefore benefit from a
# pre-flight trial-expiry warning. Local-only commands (caption format
# conversion, doctor, auth, config, update) are excluded so trivial
# invocations stay fast and side-effect free.
_SUBCOMMANDS_NEEDING_AUTH = frozenset(
    {"alignment", "youtube", "transcribe", "summarize", "translate", "diarize", "serve"}
)


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(importlib.metadata.version("lattifai"))
        raise typer.Exit()


def _register_direct_commands(app: typer.Typer) -> None:
    """Add doctor, update, auth, config, and --version as top-level commands."""
    from lattifai.cli.auth import login, logout, trial, whoami
    from lattifai.cli.config import app as config_app

    auth_app = typer.Typer(help="Manage LattifAI CLI authentication.")
    auth_app.command("login", help="Log in to LattifAI from the CLI.")(login)
    auth_app.command("logout", help="Log out and revoke the current CLI session.")(logout)
    auth_app.command("whoami", help="Show the current authenticated LattifAI identity.")(whoami)
    auth_app.command("trial", help="Get a free trial API key (no sign-up required).")(trial)
    app.add_typer(auth_app, name="auth")

    app.add_typer(config_app, name="config")

    @app.command("doctor", help="Run environment diagnostics for LattifAI.")
    def _doctor():
        from lattifai.cli.doctor import doctor

        raise SystemExit(doctor())

    @app.command("update", help="Update LattifAI CLI to the latest version.")
    def _update(
        force: bool = typer.Option(False, "--force", help="Force reinstall even if at latest version."),
    ):
        from lattifai.cli.update import update

        raise SystemExit(update(force=force))


def main():
    app = create_cli()
    _register_direct_commands(app)

    @app.callback()
    def _callback(
        ctx: typer.Context,
        version: bool = typer.Option(
            False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ):
        """LattifAI CLI — precision alignment, transcription, and subtitle tools."""
        # Pre-flight trial-expiry warning for commands that hit the backend.
        # Surfaces a friendly message before users get a 401 from the server.
        if ctx.invoked_subcommand in _SUBCOMMANDS_NEEDING_AUTH:
            from lattifai.cli.auth import warn_if_trial_expiring

            warn_if_trial_expiring()

    app()
