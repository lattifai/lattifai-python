"""
File existence management utilities for video processing workflows
"""

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import colorful

try:
    import questionary
except ImportError:  # pragma: no cover - optional dependency
    questionary = None


class FileExistenceManager:
    """Utility class for handling file existence checks and user confirmations"""

    FILE_TYPE_INFO = {
        "media": ("🎬", "Media"),
        # 'audio': ('📱', 'Audio'),
        # 'video': ('🎬', 'Video'),
        "subtitle": ("📝", "Subtitle"),
    }

    @staticmethod
    def check_existing_files(
        video_id: str,
        output_dir: str,
        media_formats: List[str] = None,
        subtitle_formats: List[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Enhanced version to check for existing media files with customizable formats

        Args:
            video_id: Video ID from any platform
            output_dir: Output directory to check
            media_formats: List of media formats to check (audio and video combined)
            subtitle_formats: List of subtitle formats to check

        Returns:
            Dictionary with 'media', 'subtitle' keys containing lists of existing files
        """
        output_path = Path(output_dir).expanduser()
        existing_files = {"media": [], "subtitle": []}

        if not output_path.exists():
            return existing_files

        # Default formats - combine audio and video formats
        media_formats = media_formats or ["mp3", "wav", "m4a", "aac", "opus", "mp4", "webm", "mkv", "avi"]
        subtitle_formats = subtitle_formats or ["md", "srt", "vtt", "ass", "ssa", "sub", "sbv", "txt"]

        # Check for media files (audio and video)
        for ext in set(media_formats):  # Remove duplicates
            # Pattern 1: Simple pattern like {video_id}.mp3
            media_file = output_path / f"{video_id}.{ext}"
            if media_file.exists():
                existing_files["media"].append(str(media_file))

            # Pattern 2: With suffix like {video_id}_Edit.mp3 or {video_id}.something.mp3
            for media_file in output_path.glob(f"{video_id}*.{ext}"):
                file_path = str(media_file)
                if file_path not in existing_files["media"]:
                    existing_files["media"].append(file_path)

        # Check for subtitle files
        for ext in set(subtitle_formats):  # Remove duplicates
            # Check multiple naming patterns for subtitle files
            # Pattern 1: Simple pattern like {video_id}.vtt
            subtitle_file = output_path / f"{video_id}.{ext}"
            if subtitle_file.exists():
                existing_files["subtitle"].append(str(subtitle_file))

            # Pattern 2: With language/track suffix like {video_id}.en-trackid.vtt
            for sub_file in output_path.glob(f"{video_id}.*.{ext}"):
                file_path = str(sub_file)
                if file_path not in existing_files["subtitle"]:
                    existing_files["subtitle"].append(file_path)

        if "md" in subtitle_formats:
            # Gemini-specific pattern: {video_id}_Gemini.md
            gemini_subtitle_file = output_path / f"{video_id}_Gemini.md"
            if gemini_subtitle_file.exists():
                existing_files["subtitle"].append(str(gemini_subtitle_file))

        return existing_files

    @staticmethod
    def prompt_user_confirmation(existing_files: Dict[str, List[str]], operation: str = "download") -> str:
        """
        Prompt user for confirmation when files already exist (legacy, confirms all files together)

        Args:
            existing_files: Dictionary of existing files
            operation: Type of operation (e.g., "download", "generate")

        Returns:
            User choice: 'use' (use existing), 'overwrite' (regenerate), or 'cancel'
        """
        has_media = bool(existing_files.get("media", []))
        has_subtitle = bool(existing_files.get("subtitle", []))

        if not has_media and not has_subtitle:
            return "proceed"  # No existing files, proceed normally

        # Header with warning color
        print(f'\n{colorful.bold_yellow("⚠️  Existing files found:")}')

        # Collect file paths for options
        file_paths = []
        if has_media:
            file_paths.extend(existing_files["media"])
        if has_subtitle:
            file_paths.extend(existing_files["subtitle"])

        # Create display options with emojis
        options = []
        for file_path in file_paths:
            # Determine emoji based on file type
            if has_media and file_path in existing_files["media"]:
                display_text = f'{colorful.green("•")} 🎬 Media file: {file_path}'
            else:
                display_text = f'{colorful.green("•")} 📝 Subtitle file: {file_path}'
            options.append((display_text, file_path))

        # Add overwrite and cancel options
        options.extend(
            [
                ("                  Overwrite existing files (re-generate or download)", "overwrite"),
                ("                  Cancel operation", "cancel"),
            ]
        )

        prompt_message = "What would you like to do?"
        default_value = file_paths[0] if file_paths else "use"
        choice = FileExistenceManager._prompt_user_choice(prompt_message, options, default=default_value)

        if choice == "overwrite":
            print(f'{colorful.yellow("🔄 Overwriting existing files")}')
        elif choice == "cancel":
            print(f'{colorful.red("❌ Operation cancelled")}')
        elif choice in file_paths:
            print(f'{colorful.green(f"✅ Using selected file: {choice}")}')
        else:
            print(f'{colorful.green("✅ Using existing files")}')

        return choice

    @staticmethod
    def prompt_file_type_confirmation(file_type: str, files: List[str], operation: str = "download") -> str:
        """
        Prompt user for confirmation for a specific file type

        Args:
            file_type: Type of file ('audio', 'video', 'subtitle', 'gemini')
            files: List of existing files of this type
            operation: Type of operation (e.g., "download", "generate")

        Returns:
            User choice: 'use' (use existing), 'overwrite' (regenerate), or 'cancel'
        """
        if not files:
            return "proceed"

        emoji, label = FileExistenceManager.FILE_TYPE_INFO.get(file_type, ("📄", file_type.capitalize()))
        del emoji  # Unused variable

        # Header with warning color
        print(f'\n{colorful.bold_yellow(f"⚠️  Existing {label} files found:")}')

        for file_path in files:
            print(f'   {colorful.green("•")} {file_path}')

        prompt_message = f"What would you like to do with {label.lower()} files?"
        options = [
            (f"Use existing {label.lower()} files (skip {operation})", "use"),
            (f"Overwrite {label.lower()} files (re-{operation})", "overwrite"),
            ("Cancel operation", "cancel"),
        ]
        choice = FileExistenceManager._prompt_user_choice(prompt_message, options, default="use")

        if choice == "use":
            print(f'{colorful.green(f"✅ Using existing {label.lower()} files")}')
        elif choice == "overwrite":
            print(f'{colorful.yellow(f"🔄 Overwriting {label.lower()} files")}')
        elif choice == "cancel":
            print(f'{colorful.red("❌ Operation cancelled")}')

        return choice

    @staticmethod
    def prompt_file_selection(
        file_type: str, files: List[str], operation: str = "use", enable_gemini: bool = False
    ) -> str:
        """
        Prompt user to select a specific file from a list, or choose to overwrite/cancel

        Args:
            file_type: Type of file (e.g., 'gemini transcript', 'subtitle')
            files: List of existing files to choose from
            operation: Type of operation (e.g., "transcribe", "download")
            enable_gemini: If True, adds "Transcribe with Gemini" option

        Returns:
            Selected file path, 'overwrite' to regenerate, 'gemini' to transcribe with Gemini, or 'cancel' to abort
        """
        if not files:
            return "proceed"

        # If only one file, simplify the choice
        if len(files) == 1:
            return (
                FileExistenceManager.prompt_file_type_confirmation(
                    file_type=file_type, files=files, operation=operation
                )
                if files
                else "proceed"
            )

        # Multiple files: let user choose which one
        print(f'\n{colorful.bold_yellow(f"⚠️  Multiple {file_type} files found:")}')

        # Create options with full file paths
        options = []
        for i, file_path in enumerate(files, 1):
            # Display full path for clarity
            options.append((f"{colorful.cyan(file_path)}", file_path))

        # Add Gemini transcription option if enabled
        if enable_gemini:
            options.append((colorful.magenta("✨ Transcribe with Gemini 2.5 Pro"), "gemini"))

        # Add overwrite and cancel options
        options.append((colorful.yellow(f"Overwrite (re-{operation} or download)"), "overwrite"))
        options.append((colorful.red("Cancel operation"), "cancel"))

        prompt_message = colorful.bold_black_on_cyan(f"Select which {file_type} to use:")
        choice = FileExistenceManager._prompt_user_choice(prompt_message, options, default=files[0])

        if choice == "cancel":
            print(f'{colorful.red("❌ Operation cancelled")}')
        elif choice == "overwrite":
            print(f'{colorful.yellow(f"🔄 Overwriting all {file_type} files")}')
        elif choice == "gemini":
            print(f'{colorful.magenta("✨ Will transcribe with Gemini 2.5 Pro")}')
        else:
            print(f'{colorful.green(f"✅ Using: {choice}")}')

        return choice

    @staticmethod
    def prompt_per_file_type_confirmation(
        existing_files: Dict[str, List[str]], operation: str = "download"
    ) -> Dict[str, str]:
        """
        Prompt user for confirmation for each file type, combining interactive selections when possible.

        Args:
            existing_files: Dictionary of existing files by type
            operation: Type of operation (e.g., "download", "generate")

        Returns:
            Dictionary mapping file type to user choice ('use', 'overwrite', 'proceed', or 'cancel')
        """
        ordered_types = []
        for preferred in ["media", "audio", "video", "subtitle"]:
            if preferred not in ordered_types:
                ordered_types.append(preferred)
        for file_type in existing_files.keys():
            if file_type not in ordered_types:
                ordered_types.append(file_type)

        file_types_with_files = [ft for ft in ordered_types if existing_files.get(ft)]
        choices = {ft: "proceed" for ft in ordered_types}

        if not file_types_with_files:
            return choices

        combined_result = FileExistenceManager._combined_file_type_prompt(
            existing_files, operation, file_types_with_files
        )
        if combined_result is not None:
            choices.update(combined_result)
            return choices

        for file_type in file_types_with_files:
            choice = FileExistenceManager.prompt_file_type_confirmation(file_type, existing_files[file_type], operation)
            choices[file_type] = choice
            if choice == "cancel":
                for remaining in file_types_with_files[file_types_with_files.index(file_type) + 1 :]:
                    choices[remaining] = "cancel"
                break

        return choices

    @staticmethod
    def is_interactive_mode() -> bool:
        """Check if we're running in interactive mode (TTY available)"""
        return sys.stdin.isatty() and sys.stdout.isatty()

    @staticmethod
    def _prompt_user_choice(
        prompt_message: str,
        options: Sequence[Tuple[str, str]],
        default: str = None,
    ) -> str:
        """
        Prompt the user to select from the provided options using an interactive selector when available.

        Args:
            prompt_message: Message displayed above the options.
            options: Sequence of (label, value) option tuples.
            default: Value to use when the user submits without a selection.

        Returns:
            The selected option value.
        """
        interactive_mode = FileExistenceManager.is_interactive_mode()

        if interactive_mode and FileExistenceManager._supports_native_selector():
            try:
                return FileExistenceManager._prompt_with_arrow_keys(prompt_message, options, default)
            except Exception:
                # Fall back to other mechanisms if native selector fails
                pass

        if interactive_mode and questionary is not None and not FileExistenceManager._is_asyncio_loop_running():
            try:
                questionary_choices = [questionary.Choice(title=str(label), value=value) for label, value in options]
                selection = questionary.select(
                    message=str(prompt_message),
                    choices=questionary_choices,
                    default=default,
                ).ask()
            except (KeyboardInterrupt, EOFError):
                return "cancel"
            except Exception:
                selection = None

            if selection:
                return selection
            if default:
                return default
            return "cancel"

        return FileExistenceManager._prompt_with_numeric_input(prompt_message, options, default)

    @staticmethod
    def _prompt_with_numeric_input(
        prompt_message: str,
        options: Sequence[Tuple[str, str]],
        default: str = None,
    ) -> str:
        numbered_choices = {str(index + 1): value for index, (_, value) in enumerate(options)}
        label_lines = [f"{index + 1}. {label}" for index, (label, _) in enumerate(options)]
        prompt_header = f"{prompt_message}"
        print(prompt_header)
        for line in label_lines:
            print(line)

        while True:
            try:
                raw_choice = input(f"\nEnter your choice (1-{len(options)}): ").strip()
            except (EOFError, KeyboardInterrupt):
                return "cancel"

            if not raw_choice and default:
                return default

            if raw_choice in numbered_choices:
                return numbered_choices[raw_choice]

            print("Invalid choice. Please enter one of the displayed numbers.")

    @staticmethod
    def _combined_file_type_prompt(
        existing_files: Dict[str, List[str]],
        operation: str,
        file_types: Sequence[str],
    ) -> Optional[Dict[str, str]]:
        interactive_mode = FileExistenceManager.is_interactive_mode()
        if not interactive_mode:
            return None

        if FileExistenceManager._supports_native_selector():
            result = FileExistenceManager._prompt_combined_file_actions_native(existing_files, file_types, operation)
            if result is not None:
                return result

        if questionary is not None and not FileExistenceManager._is_asyncio_loop_running():
            return FileExistenceManager._prompt_combined_file_actions_questionary(existing_files, file_types, operation)

        return None

    @staticmethod
    def _prompt_combined_file_actions_native(
        existing_files: Dict[str, List[str]],
        file_types: Sequence[str],
        operation: str,
    ) -> Optional[Dict[str, str]]:
        states = {file_type: "use" for file_type in file_types}
        selected_index = 0
        total_items = len(file_types) + 2  # file types + confirm + cancel
        total_lines = len(file_types) + 4  # prompt + items + confirm/cancel + instructions
        print()
        FileExistenceManager._render_combined_file_menu(existing_files, file_types, states, selected_index, operation)

        num_mapping = {str(index + 1): index for index in range(len(file_types))}

        try:
            if os.name == "nt":
                read_key = FileExistenceManager._read_key_windows
                raw_mode = _NullContext()
            else:
                read_key = FileExistenceManager._read_key_posix
                raw_mode = FileExistenceManager._stdin_raw_mode()

            with raw_mode:
                while True:
                    key = read_key()
                    if key == "up":
                        selected_index = (selected_index - 1) % total_items
                        FileExistenceManager._refresh_combined_file_menu(
                            total_lines, existing_files, file_types, states, selected_index, operation
                        )
                    elif key == "down":
                        selected_index = (selected_index + 1) % total_items
                        FileExistenceManager._refresh_combined_file_menu(
                            total_lines, existing_files, file_types, states, selected_index, operation
                        )
                    elif key in ("enter", "space"):
                        if selected_index < len(file_types):
                            current_type = file_types[selected_index]
                            states[current_type] = "overwrite" if states[current_type] == "use" else "use"
                            FileExistenceManager._refresh_combined_file_menu(
                                total_lines, existing_files, file_types, states, selected_index, operation
                            )
                        elif selected_index == len(file_types):
                            return FileExistenceManager._finalize_combined_states(file_types, states)
                        else:
                            return FileExistenceManager._cancel_combined_states(file_types)
                    elif key == "cancel":
                        return FileExistenceManager._cancel_combined_states(file_types)
                    elif key in num_mapping:
                        selected_index = num_mapping[key]
                        FileExistenceManager._refresh_combined_file_menu(
                            total_lines, existing_files, file_types, states, selected_index, operation
                        )
        except KeyboardInterrupt:
            return FileExistenceManager._cancel_combined_states(file_types)
        except Exception:
            return None
        finally:
            print()

        return None

    @staticmethod
    def _render_combined_file_menu(
        existing_files: Dict[str, List[str]],
        file_types: Sequence[str],
        states: Dict[str, str],
        selected_index: int,
        operation: str,
    ) -> None:
        prompt = colorful.bold_black_on_cyan("Select how to handle existing files")
        print(prompt)

        for idx, file_type in enumerate(file_types):
            label = FileExistenceManager.FILE_TYPE_INFO.get(file_type, ("📄", file_type.capitalize()))[1]
            count = len(existing_files.get(file_type, []))
            count_suffix = f' ({count} file{"s" if count != 1 else ""})'
            state_plain = "Use existing" if states[file_type] == "use" else f"Overwrite ({operation})"
            if idx == selected_index:
                prefix = colorful.bold_white(">")
                line = colorful.bold_black_on_cyan(f"{label}: {state_plain}{count_suffix}")
            else:
                prefix = " "
                if states[file_type] == "use":
                    state_text = colorful.green("Use existing")
                else:
                    state_text = colorful.yellow(f"Overwrite ({operation})")
                line = f"{label}: {state_text}{count_suffix}"
            print(f"{prefix} {line}")

        confirm_index = len(file_types)
        cancel_index = len(file_types) + 1

        if selected_index == confirm_index:
            confirm_line = colorful.bold_black_on_cyan("Confirm selections")
            confirm_prefix = colorful.bold_white(">")
        else:
            confirm_line = colorful.bold_green("Confirm selections")
            confirm_prefix = " "
        print(f"{confirm_prefix} {confirm_line}")

        if selected_index == cancel_index:
            cancel_line = colorful.bold_black_on_cyan("Cancel operation")
            cancel_prefix = colorful.bold_white(">")
        else:
            cancel_line = colorful.bold_red("Cancel operation")
            cancel_prefix = " "
        print(f"{cancel_prefix} {cancel_line}")

        print(
            "Use "
            + colorful.bold_black_on_cyan("↑/↓")
            + " to navigate. Enter/Space toggles an item. Confirm to proceed or cancel to abort."
        )

    @staticmethod
    def _refresh_combined_file_menu(
        total_lines: int,
        existing_files: Dict[str, List[str]],
        file_types: Sequence[str],
        states: Dict[str, str],
        selected_index: int,
        operation: str,
    ) -> None:
        move_up = "\033[F" * total_lines
        clear_line = "\033[K"
        sys.stdout.write(move_up)
        sys.stdout.write(clear_line)
        sys.stdout.flush()

        prompt = colorful.bold_black_on_cyan("Select how to handle existing files")
        print(prompt)

        for idx, file_type in enumerate(file_types):
            sys.stdout.write(clear_line)
            label = FileExistenceManager.FILE_TYPE_INFO.get(file_type, ("📄", file_type.capitalize()))[1]
            count = len(existing_files.get(file_type, []))
            count_suffix = f' ({count} file{"s" if count != 1 else ""})'
            state_plain = "Use existing" if states[file_type] == "use" else f"Overwrite ({operation})"
            if idx == selected_index:
                prefix = colorful.bold_white(">")
                line = colorful.bold_black_on_cyan(f"{label}: {state_plain}{count_suffix}")
            else:
                prefix = " "
                if states[file_type] == "use":
                    state_text = colorful.green("Use existing")
                else:
                    state_text = colorful.yellow(f"Overwrite ({operation})")
                line = f"{label}: {state_text}{count_suffix}"
            print(f"{prefix} {line}")

        sys.stdout.write(clear_line)
        confirm_index = len(file_types)
        cancel_index = len(file_types) + 1
        if selected_index == confirm_index:
            confirm_line = colorful.bold_black_on_cyan("Confirm selections")
            confirm_prefix = colorful.bold_white(">")
        else:
            confirm_line = colorful.bold_green("Confirm selections")
            confirm_prefix = " "
        print(f"{confirm_prefix} {confirm_line}")

        sys.stdout.write(clear_line)
        if selected_index == cancel_index:
            cancel_line = colorful.bold_black_on_cyan("Cancel operation")
            cancel_prefix = colorful.bold_white(">")
        else:
            cancel_line = colorful.bold_red("Cancel operation")
            cancel_prefix = " "
        print(f"{cancel_prefix} {cancel_line}")

        sys.stdout.write(clear_line)
        print(
            "Use "
            + colorful.bold_black_on_cyan("↑/↓")
            + " to navigate. Enter/Space toggles an item. Confirm to proceed or cancel to abort."
        )
        sys.stdout.flush()

    @staticmethod
    def _prompt_combined_file_actions_questionary(
        existing_files: Dict[str, List[str]],
        file_types: Sequence[str],
        operation: str,
    ) -> Dict[str, str]:
        label_choices = []
        for file_type in file_types:
            label = FileExistenceManager.FILE_TYPE_INFO.get(file_type, ("📄", file_type.capitalize()))[1]
            count = len(existing_files.get(file_type, []))
            count_suffix = f' ({count} file{"s" if count != 1 else ""})'
            label_choices.append(questionary.Choice(title=f"{label}{count_suffix}", value=file_type))

        label_choices.append(questionary.Choice(title="Cancel operation", value="__cancel__"))

        try:
            selection = questionary.checkbox(
                message="Select file types to overwrite (others will use existing files)",
                choices=label_choices,
                instruction="Press Space to toggle overwrite. Press Enter to confirm.",
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return FileExistenceManager._cancel_combined_states(file_types)

        if selection is None or "__cancel__" in selection:
            return FileExistenceManager._cancel_combined_states(file_types)

        states = {file_type: ("overwrite" if file_type in selection else "use") for file_type in file_types}
        return FileExistenceManager._finalize_combined_states(file_types, states)

    @staticmethod
    def _finalize_combined_states(file_types: Sequence[str], states: Dict[str, str]) -> Dict[str, str]:
        return {file_type: ("overwrite" if states.get(file_type) == "overwrite" else "use") for file_type in file_types}

    @staticmethod
    def _cancel_combined_states(file_types: Sequence[str]) -> Dict[str, str]:
        return {file_type: "cancel" for file_type in file_types}

    @staticmethod
    def _supports_native_selector() -> bool:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return False
        if os.name == "nt":
            try:
                import msvcrt  # noqa: F401
            except ImportError:
                return False
            return True
        try:
            import termios  # noqa: F401
            import tty  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _prompt_with_arrow_keys(
        prompt_message: str,
        options: Sequence[Tuple[str, str]],
        default: str = None,
    ) -> str:
        value_by_number = {str(index + 1): value for index, (_, value) in enumerate(options)}
        selected_index = 0
        if default is not None:
            for idx, (_, value) in enumerate(options):
                if value == default:
                    selected_index = idx
                    break

        helper_lines = 2  # prompt line + hint line
        total_lines = len(options) + helper_lines
        print()  # spacer before menu
        FileExistenceManager._render_menu(prompt_message, options, selected_index)

        try:
            if os.name == "nt":
                read_key = FileExistenceManager._read_key_windows
                raw_mode = _NullContext()
            else:
                read_key = FileExistenceManager._read_key_posix
                raw_mode = FileExistenceManager._stdin_raw_mode()

            with raw_mode:
                while True:
                    key = read_key()
                    if key == "up":
                        selected_index = (selected_index - 1) % len(options)
                        FileExistenceManager._refresh_menu(total_lines, prompt_message, options, selected_index)
                    elif key == "down":
                        selected_index = (selected_index + 1) % len(options)
                        FileExistenceManager._refresh_menu(total_lines, prompt_message, options, selected_index)
                    elif key == "enter":
                        return options[selected_index][1]
                    elif key in value_by_number:
                        return value_by_number[key]
                    elif key == "cancel":
                        return "cancel"
        except KeyboardInterrupt:
            return "cancel"
        except Exception:
            return FileExistenceManager._prompt_with_numeric_input(prompt_message, options, default)
        finally:
            print()

    @staticmethod
    def _render_menu(prompt_message: str, options: Sequence[Tuple[str, str]], selected_index: int) -> None:
        prompt = f"{prompt_message}"
        print(prompt)
        for idx, (label, _) in enumerate(options):
            if idx == selected_index:
                prefix = colorful.bold_white(">")
                suffix = colorful.bold_black_on_cyan(str(label))
            else:
                prefix = " "
                suffix = label
            print(f"{prefix} {suffix}")
        print("Use " + colorful.bold_black_on_cyan("↑/↓") + " to move, Enter to confirm, or press a number to choose.")

    @staticmethod
    def _refresh_menu(
        total_lines: int,
        prompt_message: str,
        options: Sequence[Tuple[str, str]],
        selected_index: int,
    ) -> None:
        move_up = "\033[F" * total_lines
        clear_line = "\033[K"
        sys.stdout.write(move_up)
        sys.stdout.write(clear_line)
        sys.stdout.flush()
        prompt = f"{prompt_message}"
        print(prompt)
        for idx, (label, _) in enumerate(options):
            sys.stdout.write(clear_line)
            if idx == selected_index:
                prefix = colorful.bold_white(">")
                suffix = colorful.bold_black_on_cyan(str(label))
            else:
                prefix = " "
                suffix = label
            print(f"{prefix} {suffix}")
        sys.stdout.write(clear_line)
        print("Use " + colorful.bold_black_on_cyan("↑/↓") + " to move, Enter to confirm, or press a number to choose.")
        sys.stdout.flush()

    @staticmethod
    def _read_key_windows() -> str:
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                extended = msvcrt.getwch()
                if extended == "H":
                    return "up"
                if extended == "P":
                    return "down"
                continue
            if ch in ("\r", "\n"):
                return "enter"
            if ch == " ":
                return "space"
            if ch.isdigit():
                return ch
            if ch.lower() in ("j", "k"):
                return "down" if ch.lower() == "j" else "up"
            if ch == "\x1b":
                return "cancel"
            if ch == "\x03":
                raise KeyboardInterrupt

    @staticmethod
    def _read_key_posix() -> str:
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "cancel"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch.isdigit():
            return ch
        if ch.lower() == "j":
            return "down"
        if ch.lower() == "k":
            return "up"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ""

    @staticmethod
    def _stdin_raw_mode():
        if os.name == "nt":
            return _NullContext()

        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        @contextmanager
        def _raw():
            try:
                tty.setcbreak(fd)
                yield
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return _raw()

    @staticmethod
    def _is_asyncio_loop_running() -> bool:
        """Detect whether an asyncio event loop is already running in this thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
