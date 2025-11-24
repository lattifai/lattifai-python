import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # noqa: F401

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "lattifai" / "workflow" / "file_manager.py"
MODULE_SPEC = importlib.util.spec_from_file_location("file_manager_for_tests", MODULE_PATH)
file_manager = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC and MODULE_SPEC.loader
MODULE_SPEC.loader.exec_module(file_manager)
FileExistenceManager = file_manager.FileExistenceManager


def test_prompt_choice_uses_questionary_when_available():
    options = [("Use files", "use"), ("Overwrite files", "overwrite"), ("Cancel", "cancel")]

    with (
        patch.object(file_manager.FileExistenceManager, "is_interactive_mode", return_value=True) as mock_interactive,
        patch.object(file_manager.FileExistenceManager, "_supports_native_selector", return_value=False),
        patch.object(file_manager, "questionary") as mock_questionary,
    ):
        select_instance = MagicMock()
        select_instance.ask.return_value = "overwrite"
        mock_questionary.select.return_value = select_instance

        choice = FileExistenceManager._prompt_user_choice("Select", options, default="use")

        assert choice == "overwrite"
        mock_questionary.select.assert_called_once()
        select_instance.ask.assert_called_once()
        mock_interactive.assert_called_once()


def test_prompt_choice_falls_back_to_numeric_input():
    options = [("Use files", "use"), ("Overwrite files", "overwrite"), ("Cancel", "cancel")]

    with (
        patch.object(file_manager.FileExistenceManager, "is_interactive_mode", return_value=False) as mock_interactive,
        patch.object(file_manager.FileExistenceManager, "_supports_native_selector", return_value=False),
        patch("builtins.input", side_effect=["2"]),
    ):
        choice = FileExistenceManager._prompt_user_choice("Select", options, default="use")

    assert choice == "overwrite"
    mock_interactive.assert_called_once()


def test_prompt_per_file_type_combined_questionary_overwrite():
    existing_files = {"media": ["/tmp/video.mp4"], "caption": ["/tmp/video.srt"]}

    with (
        patch.object(file_manager.FileExistenceManager, "is_interactive_mode", return_value=True),
        patch.object(file_manager.FileExistenceManager, "_supports_native_selector", return_value=False),
        patch.object(file_manager.FileExistenceManager, "_is_asyncio_loop_running", return_value=False),
        patch.object(file_manager, "questionary") as mock_questionary,
    ):
        checkbox_instance = MagicMock()
        checkbox_instance.ask.return_value = ["media"]
        mock_questionary.checkbox.return_value = checkbox_instance

        choices = FileExistenceManager.prompt_per_file_type_confirmation(existing_files, "download")

    assert choices["media"] == "overwrite"
    assert choices["caption"] == "use"
    assert choices["audio"] == "proceed"


def test_prompt_per_file_type_combined_questionary_cancel():
    existing_files = {"media": ["/tmp/video.mp4"], "caption": ["/tmp/video.srt"]}

    with (
        patch.object(file_manager.FileExistenceManager, "is_interactive_mode", return_value=True),
        patch.object(file_manager.FileExistenceManager, "_supports_native_selector", return_value=False),
        patch.object(file_manager.FileExistenceManager, "_is_asyncio_loop_running", return_value=False),
        patch.object(file_manager, "questionary") as mock_questionary,
    ):
        checkbox_instance = MagicMock()
        checkbox_instance.ask.return_value = ["__cancel__"]
        mock_questionary.checkbox.return_value = checkbox_instance

        choices = FileExistenceManager.prompt_per_file_type_confirmation(existing_files, "download")

    assert choices["media"] == "cancel"
    assert choices["caption"] == "cancel"


def test_prompt_per_file_type_sequential_fallback():
    existing_files = {"media": ["/tmp/video.mp4"]}

    with (
        patch.object(file_manager.FileExistenceManager, "is_interactive_mode", return_value=False),
        patch.object(
            file_manager.FileExistenceManager,
            "prompt_file_type_confirmation",
            return_value="use",
        ) as mock_prompt,
    ):
        choices = FileExistenceManager.prompt_per_file_type_confirmation(existing_files, "download")

    mock_prompt.assert_called_once()
    assert choices["media"] == "use"
    assert choices["caption"] == "proceed"
