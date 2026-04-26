"""Centralized CLI color theme for LattifAI.

Uses 256-color ANSI for readability on both light and dark terminal
backgrounds. Zero cyan. Bright yellow is avoided for warnings because
it washes out on white backgrounds — we use a dark orange instead,
matching the ``RICH_WARN = "dark_orange"`` Rich counterpart below.
"""

import colorful

# Enable 256-color mode and register custom palette colors. 256-color is
# universally supported by modern terminals (iTerm, Terminal.app, vscode,
# Windows Terminal, tmux) without requiring truecolor.
colorful.use_256_ansi_colors()
colorful.update_palette(
    {
        "laiWarn": "#D97706",  # dark orange (Tailwind amber-600); maps to ANSI 172
    }
)


class _Theme:
    """Semantic color roles mapped to colorful style callables."""

    # ── Core semantic roles ──────────────────────────────────────
    step = colorful.bold_blue  # workflow steps, progress info
    ok = colorful.bold_green  # success, completion
    warn = colorful.bold & colorful.laiWarn  # warnings, caution (dark orange)
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
