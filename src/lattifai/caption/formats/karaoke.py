"""Karaoke configuration classes for word-level subtitle export.

This module provides configuration dataclasses for karaoke-style exports
including ASS \\k tags, Enhanced LRC, and TTML Word timing.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal


class KaraokeFonts:
    """Common karaoke font constants.

    These are reference constants for popular fonts. You can use any
    system font name as the font_name parameter in KaraokeStyle.
    """

    # Western fonts
    ARIAL = "Arial"
    IMPACT = "Impact"
    VERDANA = "Verdana"
    HELVETICA = "Helvetica"

    # Chinese fonts
    NOTO_SANS_SC = "Noto Sans SC"
    MICROSOFT_YAHEI = "Microsoft YaHei"
    PINGFANG_SC = "PingFang SC"
    SIMHEI = "SimHei"

    # Japanese fonts
    NOTO_SANS_JP = "Noto Sans JP"
    MEIRYO = "Meiryo"
    HIRAGINO_SANS = "Hiragino Sans"

    # Korean fonts
    NOTO_SANS_KR = "Noto Sans KR"
    MALGUN_GOTHIC = "Malgun Gothic"


@dataclass
class KaraokeStyle:
    """ASS karaoke style configuration.

    Attributes:
        effect: Karaoke effect type
            - "sweep": Gradual fill from left to right (\\kf tag)
            - "instant": Instant highlight (\\k tag)
            - "outline": Outline then fill (\\ko tag)
        primary_color: Highlighted text color (#RRGGBB)
        secondary_color: Pre-highlight text color (#RRGGBB)
        outline_color: Text outline color (#RRGGBB)
        back_color: Shadow color (#RRGGBB)
        font_name: Font family name (use KaraokeFonts constants or any system font)
        font_size: Font size in points
        bold: Enable bold text
        italic: Enable italic text
        outline_width: Outline thickness
        shadow_depth: Shadow distance
        alignment: ASS alignment (1-9, numpad style), 2=bottom-center
        margin_l: Left margin in pixels
        margin_r: Right margin in pixels
        margin_v: Vertical margin in pixels
    """

    effect: Literal["sweep", "instant", "outline"] = "sweep"

    # Colors (#RRGGBB format)
    primary_color: str = "#00FFFF"
    secondary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    back_color: str = "#000000"

    # Font
    font_name: str = KaraokeFonts.ARIAL
    font_size: int = 48
    bold: bool = True
    italic: bool = False

    # Border and shadow
    outline_width: float = 2.0
    shadow_depth: float = 1.0

    # Position
    alignment: int = 2
    margin_l: int = 20
    margin_r: int = 20
    margin_v: int = 20


@dataclass
class KaraokeConfig:
    """Karaoke export configuration.

    Attributes:
        enabled: Whether karaoke mode is enabled
        style: ASS style configuration (also affects visual appearance)
        lrc_precision: LRC time precision ("centisecond" or "millisecond")
        lrc_metadata: LRC metadata dict (ar, ti, al, etc.)
        ttml_timing_mode: TTML timing attribute ("Word" or "Line")
    """

    enabled: bool = True
    style: KaraokeStyle = field(default_factory=KaraokeStyle)

    # LRC specific
    lrc_precision: Literal["centisecond", "millisecond"] = "millisecond"
    lrc_metadata: Dict[str, str] = field(default_factory=dict)

    # TTML specific
    ttml_timing_mode: Literal["Word", "Line"] = "Word"


__all__ = ["KaraokeFonts", "KaraokeStyle", "KaraokeConfig"]
