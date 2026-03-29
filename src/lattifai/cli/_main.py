"""Custom lai CLI entry point that extends nemo_run with direct commands.

This wraps nemo_run's Typer app and registers doctor/update as top-level
commands (no namespace subcommand required).
"""

import typer
from nemo_run.cli.api import create_cli


def _register_direct_commands(app: typer.Typer) -> None:
    """Add doctor, update, and config as top-level commands (no namespace group)."""
    from lattifai.cli.config import app as config_app

    app.add_typer(config_app, name="config")

    @app.command("doctor", help="Run environment diagnostics for LattifAI.")
    def _doctor():
        from lattifai.cli.doctor import doctor

        doctor()

    @app.command("update", help="Update LattifAI CLI to the latest version.")
    def _update(
        force: bool = typer.Option(False, "--force", help="Force reinstall even if at latest version."),
    ):
        from lattifai.cli.update import update

        update(force=force)


def main():
    app = create_cli()
    _register_direct_commands(app)
    app()
