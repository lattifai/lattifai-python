"""Centralized CLI color theme for LattifAI.

All colors use bright ANSI variants (codes 90-97) for readability
on both light and dark terminal backgrounds. Zero cyan.
"""

import colorful


class _Theme:
    """Semantic color roles mapped to colorful style callables."""

    # ── Core semantic roles ──────────────────────────────────────
    step = colorful.bold_blue  # workflow steps, progress info
    ok = colorful.bold_green  # success, completion
    warn = colorful.bold_yellow  # warnings, caution
    err = colorful.bold_red  # errors, failures
    dim = colorful.bold_black  # muted text, dividers (bright black = dark gray)
    accent = colorful.bold_magenta  # highlights, special items
    label = colorful.bold  # table headers, emphasis
    value = colorful.bold_yellow  # numeric values, metrics

    # ── Interactive menu ─────────────────────────────────────────
    menu_active = colorful.bold_white_on_blue  # selected / focused item
    menu_cursor = colorful.bold_blue  # cursor indicator ">"
    menu_confirm = colorful.bold_green  # confirm action
    menu_cancel = colorful.bold_red  # cancel action
    menu_hint = colorful.bold_white_on_blue  # keyboard shortcut badges

    # ── Rich markup equivalents (for doctor.py / update.py) ──────
    RICH_STEP = "bold blue"
    RICH_OK = "bold green"
    RICH_WARN = "dark_orange"
    RICH_ERR = "bold red"
    RICH_DIM = "dim"
    RICH_ACCENT = "bold magenta"
    RICH_LABEL = "bold"
    RICH_HEADER = "bold blue"


theme = _Theme()
