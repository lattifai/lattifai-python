# Word-Level 字幕导出实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 Enhanced LRC、ASS `\k` 卡拉OK标签、TTML Word timing 三种格式的字级时间戳导出功能

**Architecture:** 在现有 FormatWriter 架构上扩展，添加统一的 `word_level` 参数。新增 `karaoke.py` 配置模块和 `lrc.py` 格式处理器，修改 `pysubs2.py` 和 `ttml.py` 支持卡拉OK模式。

**Tech Stack:** Python 3.10+, pysubs2, lhotse, dataclasses

**设计文档:** `lattifai-site/.claude/2026-01-15-word-level-export-design.md`

---

## Task 1: 创建 KaraokeConfig 配置模块

**Files:**
- Create: `src/lattifai/caption/formats/karaoke.py`
- Test: `tests/caption/test_karaoke_config.py`

**Step 1: 写失败的测试**

```python
# tests/caption/test_karaoke_config.py
"""Tests for karaoke configuration classes."""

import pytest
from lattifai.caption.formats.karaoke import KaraokeFonts, KaraokeStyle, KaraokeConfig


class TestKaraokeFonts:
    """Test KaraokeFonts constants."""

    def test_western_fonts_exist(self):
        """Western font constants should be defined."""
        assert KaraokeFonts.ARIAL == "Arial"
        assert KaraokeFonts.IMPACT == "Impact"
        assert KaraokeFonts.VERDANA == "Verdana"

    def test_chinese_fonts_exist(self):
        """Chinese font constants should be defined."""
        assert KaraokeFonts.NOTO_SANS_SC == "Noto Sans SC"
        assert KaraokeFonts.MICROSOFT_YAHEI == "Microsoft YaHei"
        assert KaraokeFonts.PINGFANG_SC == "PingFang SC"

    def test_japanese_fonts_exist(self):
        """Japanese font constants should be defined."""
        assert KaraokeFonts.NOTO_SANS_JP == "Noto Sans JP"
        assert KaraokeFonts.MEIRYO == "Meiryo"


class TestKaraokeStyle:
    """Test KaraokeStyle dataclass."""

    def test_default_values(self):
        """Default style should have sensible defaults."""
        style = KaraokeStyle()
        assert style.effect == "sweep"
        assert style.primary_color == "#00FFFF"
        assert style.secondary_color == "#FFFFFF"
        assert style.font_name == KaraokeFonts.ARIAL
        assert style.font_size == 48
        assert style.bold is True

    def test_custom_values(self):
        """Custom values should override defaults."""
        style = KaraokeStyle(
            effect="instant",
            primary_color="#FF00FF",
            font_name=KaraokeFonts.NOTO_SANS_SC,
            font_size=56,
        )
        assert style.effect == "instant"
        assert style.primary_color == "#FF00FF"
        assert style.font_name == "Noto Sans SC"
        assert style.font_size == 56


class TestKaraokeConfig:
    """Test KaraokeConfig dataclass."""

    def test_default_config(self):
        """Default config should work."""
        config = KaraokeConfig()
        assert config.enabled is True
        assert isinstance(config.style, KaraokeStyle)
        assert config.lrc_precision == "millisecond"
        assert config.ttml_timing_mode == "Word"

    def test_lrc_metadata(self):
        """LRC metadata should be configurable."""
        config = KaraokeConfig(
            lrc_metadata={"ar": "Artist", "ti": "Title"}
        )
        assert config.lrc_metadata["ar"] == "Artist"
        assert config.lrc_metadata["ti"] == "Title"
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/caption/test_karaoke_config.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'lattifai.caption.formats.karaoke'"

**Step 3: 实现 karaoke.py**

```python
# src/lattifai/caption/formats/karaoke.py
"""Karaoke configuration classes for word-level subtitle export.

This module provides configuration dataclasses for karaoke-style exports
including ASS \k tags, Enhanced LRC, and TTML Word timing.
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
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/caption/test_karaoke_config.py -v
```

Expected: All tests PASS

**Step 5: 提交**

```bash
git add src/lattifai/caption/formats/karaoke.py tests/caption/test_karaoke_config.py
git commit -m "feat(caption): add KaraokeConfig for word-level export"
```

---

## Task 2: 实现 Enhanced LRC 格式写入

**Files:**
- Create: `src/lattifai/caption/formats/lrc.py`
- Modify: `src/lattifai/caption/formats/__init__.py:167-175` (添加 import)
- Test: `tests/caption/test_lrc_format.py`

**Step 1: 写失败的测试**

```python
# tests/caption/test_lrc_format.py
"""Tests for Enhanced LRC format."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats.lrc import LRCFormat
from lattifai.caption.formats.karaoke import KaraokeConfig
from lattifai.caption.supervision import Supervision


class TestLRCFormatWrite:
    """Test LRC format writing."""

    def test_standard_lrc_output(self):
        """Standard LRC should output line timestamps only."""
        sups = [
            Supervision(text="Hello world", start=15.2, duration=3.3),
            Supervision(text="This is karaoke", start=18.5, duration=2.0),
        ]
        result = LRCFormat.to_bytes(sups, word_level=False)
        content = result.decode("utf-8")

        assert "[00:15.200]Hello world" in content
        assert "[00:18.500]This is karaoke" in content
        # No word-level timestamps
        assert "<" not in content

    def test_enhanced_lrc_word_level(self):
        """Enhanced LRC should include word timestamps."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = LRCFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        assert "[00:15.200]" in content
        assert "<00:15.200>Hello" in content
        assert "<00:15.650>world" in content

    def test_lrc_with_metadata(self):
        """LRC should include metadata when provided."""
        sups = [Supervision(text="Hello", start=0.0, duration=1.0)]
        config = KaraokeConfig(
            lrc_metadata={"ar": "Artist", "ti": "Title", "al": "Album"}
        )
        result = LRCFormat.to_bytes(sups, word_level=False, karaoke_config=config)
        content = result.decode("utf-8")

        assert "[ar:Artist]" in content
        assert "[ti:Title]" in content
        assert "[al:Album]" in content

    def test_lrc_centisecond_precision(self):
        """LRC should support centisecond precision."""
        sups = [Supervision(text="Hello", start=15.234, duration=1.0)]
        config = KaraokeConfig(lrc_precision="centisecond")
        result = LRCFormat.to_bytes(sups, word_level=False, karaoke_config=config)
        content = result.decode("utf-8")

        # Centisecond: [00:15.23] not [00:15.234]
        assert "[00:15.23]" in content

    def test_lrc_fallback_without_alignment(self):
        """Word-level should fallback to line-level without alignment data."""
        sups = [Supervision(text="No alignment", start=10.0, duration=2.0)]
        result = LRCFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        assert "[00:10.000]No alignment" in content
        assert "<" not in content  # No word timestamps
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/caption/test_lrc_format.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 lrc.py**

```python
# src/lattifai/caption/formats/lrc.py
"""Enhanced LRC format handler.

LRC (LyRiCs) is a file format for synchronized song lyrics. Enhanced LRC
adds word-level timestamps for karaoke applications.

Standard LRC:
    [00:15.20]Hello beautiful world

Enhanced LRC (word-level):
    [00:15.20]<00:15.20>Hello <00:15.65>beautiful <00:16.40>world

Metadata tags:
    [ar:Artist Name]
    [ti:Song Title]
    [al:Album Name]
    [offset:±milliseconds]
"""

import re
from pathlib import Path
from typing import List, Optional

from lhotse.supervision import AlignmentItem

from ..supervision import Supervision
from . import register_format
from .base import FormatHandler
from .karaoke import KaraokeConfig


@register_format("lrc")
class LRCFormat(FormatHandler):
    """Enhanced LRC format with word-level timing support."""

    extensions = [".lrc"]
    description = "Enhanced LRC - karaoke lyrics format"

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read LRC file and return list of Supervision objects.

        Parses both standard LRC and enhanced LRC with word-level timestamps.
        """
        if cls.is_content(source):
            content = source
        else:
            content = Path(source).read_text(encoding="utf-8")

        supervisions = []
        # Match line timestamp: [mm:ss.xx] or [mm:ss.xxx]
        line_pattern = re.compile(r"\[(\d+):(\d+)\.(\d+)\](.+)")
        # Match word timestamp: <mm:ss.xx> or <mm:ss.xxx>
        word_pattern = re.compile(r"<(\d+):(\d+)\.(\d+)>([^<]+)")

        for line in content.split("\n"):
            line = line.strip()
            # Skip empty lines and metadata
            if not line or line.startswith("[ar:") or line.startswith("[ti:"):
                continue
            if line.startswith("[al:") or line.startswith("[offset:"):
                continue

            match = line_pattern.match(line)
            if match:
                mins, secs, frac, text = match.groups()
                # Handle centisecond vs millisecond
                if len(frac) == 2:
                    start = int(mins) * 60 + int(secs) + int(frac) / 100
                else:
                    start = int(mins) * 60 + int(secs) + int(frac) / 1000

                # Extract word-level alignment
                words = word_pattern.findall(text)
                alignment = None
                if words:
                    alignment = {"word": []}
                    for w_mins, w_secs, w_frac, w_text in words:
                        if len(w_frac) == 2:
                            w_start = int(w_mins) * 60 + int(w_secs) + int(w_frac) / 100
                        else:
                            w_start = int(w_mins) * 60 + int(w_secs) + int(w_frac) / 1000
                        alignment["word"].append(
                            AlignmentItem(
                                symbol=w_text.strip(),
                                start=w_start,
                                duration=0,  # LRC doesn't store duration
                            )
                        )
                    # Clean text (remove timestamp tags)
                    text = re.sub(r"<\d+:\d+\.\d+>", "", text)

                supervisions.append(
                    Supervision(
                        text=text.strip(),
                        start=start,
                        duration=0,  # Will calculate below
                        alignment=alignment,
                    )
                )

        # Calculate duration from next segment
        for i, sup in enumerate(supervisions):
            if i + 1 < len(supervisions):
                sup.duration = supervisions[i + 1].start - sup.start
            else:
                sup.duration = 5.0  # Default 5 seconds for last line

        return supervisions

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        **kwargs,
    ) -> Path:
        """Write supervisions to LRC file."""
        output_path = Path(output_path)
        content = cls.to_bytes(
            supervisions,
            include_speaker=include_speaker,
            word_level=word_level,
            karaoke_config=karaoke_config,
            **kwargs,
        )
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to LRC format bytes."""
        config = karaoke_config or KaraokeConfig()
        lines = []

        # Metadata header
        for key, value in config.lrc_metadata.items():
            lines.append(f"[{key}:{value}]")
        if config.lrc_metadata:
            lines.append("")

        for sup in supervisions:
            line_time = cls._format_time(sup.start, config.lrc_precision)

            if word_level and sup.alignment and "word" in sup.alignment:
                # Enhanced LRC mode: each word has timestamp
                word_parts = []
                for word in sup.alignment["word"]:
                    word_time = cls._format_time(word.start, config.lrc_precision)
                    word_parts.append(f"<{word_time}>{word.symbol}")
                lines.append(f"[{line_time}]{''.join(word_parts)}")
            else:
                # Standard LRC mode: only line timestamp
                text = sup.text or ""
                if cls._should_include_speaker(sup, include_speaker):
                    text = f"{sup.speaker}: {text}"
                lines.append(f"[{line_time}]{text}")

        return "\n".join(lines).encode("utf-8")

    @staticmethod
    def _format_time(seconds: float, precision: str) -> str:
        """Format time for LRC timestamp.

        Args:
            seconds: Time in seconds
            precision: "centisecond" for [mm:ss.xx] or "millisecond" for [mm:ss.xxx]

        Returns:
            Formatted time string
        """
        if seconds < 0:
            seconds = 0
        minutes = int(seconds // 60)
        secs = seconds % 60
        if precision == "millisecond":
            return f"{minutes:02d}:{secs:06.3f}"  # 00:15.200
        return f"{minutes:02d}:{secs:05.2f}"  # 00:15.20


__all__ = ["LRCFormat"]
```

**Step 4: 更新 `__init__.py` 添加 import**

```python
# 在 src/lattifai/caption/formats/__init__.py 的 import 部分添加:
from . import lrc  # Enhanced LRC
```

**Step 5: 运行测试验证通过**

```bash
pytest tests/caption/test_lrc_format.py -v
```

Expected: All tests PASS

**Step 6: 提交**

```bash
git add src/lattifai/caption/formats/lrc.py src/lattifai/caption/formats/__init__.py tests/caption/test_lrc_format.py
git commit -m "feat(caption): add Enhanced LRC format with word-level support"
```

---

## Task 3: 实现 ASS 卡拉OK标签

**Files:**
- Modify: `src/lattifai/caption/formats/pysubs2.py:202-244` (ASSFormat 类)
- Test: `tests/caption/test_ass_karaoke.py`

**Step 1: 写失败的测试**

```python
# tests/caption/test_ass_karaoke.py
"""Tests for ASS karaoke tag generation."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.formats.karaoke import KaraokeConfig, KaraokeStyle
from lattifai.caption.supervision import Supervision


class TestASSKaraoke:
    """Test ASS karaoke tag generation."""

    def test_karaoke_sweep_effect(self):
        """Sweep effect should use \\kf tag."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = ASSFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        # \kf45 means 45 centiseconds (0.45s)
        assert "{\\kf45}Hello" in content
        assert "{\\kf285}world" in content

    def test_karaoke_instant_effect(self):
        """Instant effect should use \\k tag."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        style = KaraokeStyle(effect="instant")
        config = KaraokeConfig(style=style)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=config)
        content = result.decode("utf-8")

        assert "{\\k45}Hello" in content
        assert "{\\k285}world" in content

    def test_karaoke_outline_effect(self):
        """Outline effect should use \\ko tag."""
        sups = [
            Supervision(
                text="Hello",
                start=0.0,
                duration=1.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    ]
                },
            )
        ]
        style = KaraokeStyle(effect="outline")
        config = KaraokeConfig(style=style)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=config)
        content = result.decode("utf-8")

        assert "{\\ko50}Hello" in content

    def test_karaoke_style_in_output(self):
        """Karaoke style should be defined in ASS output."""
        sups = [
            Supervision(
                text="Hello",
                start=0.0,
                duration=1.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    ]
                },
            )
        ]
        result = ASSFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        # Should have Karaoke style defined
        assert "Style: Karaoke" in content or "Karaoke," in content

    def test_fallback_without_alignment(self):
        """Without alignment, should output normal ASS."""
        sups = [Supervision(text="No alignment", start=10.0, duration=2.0)]
        result = ASSFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        assert "No alignment" in content
        assert "{\\k" not in content  # No karaoke tags

    def test_word_level_false_uses_original(self):
        """word_level=False should use original behavior."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = ASSFormat.to_bytes(sups, word_level=False)
        content = result.decode("utf-8")

        # Should NOT have karaoke tags
        assert "{\\k" not in content
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/caption/test_ass_karaoke.py -v
```

Expected: FAIL with "unexpected keyword argument 'word_level'"

**Step 3: 修改 pysubs2.py 的 ASSFormat**

在 `src/lattifai/caption/formats/pysubs2.py` 中修改 `ASSFormat` 类：

```python
# 在文件顶部添加 import
from typing import Dict, List, Optional
from .karaoke import KaraokeConfig, KaraokeStyle


@register_format("ass")
class ASSFormat(Pysubs2Format):
    """Advanced SubStation Alpha format with karaoke support."""

    extensions = [".ass"]
    pysubs2_format = "ass"
    description = "Advanced SubStation Alpha - rich styling and karaoke support"

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        fps: float = 25.0,
        **kwargs,
    ) -> bytes:
        """Convert to ASS format bytes.

        Args:
            supervisions: List of Supervision objects
            include_speaker: Whether to include speaker labels
            word_level: Enable karaoke mode with \\k tags
            karaoke_config: Karaoke style configuration
            fps: Framerate (not used for ASS)

        Returns:
            ASS content as bytes
        """
        # Non word-level mode: use original behavior
        if not word_level:
            return super().to_bytes(
                supervisions, include_speaker=include_speaker, fps=fps, **kwargs
            )

        config = karaoke_config or KaraokeConfig()
        subs = pysubs2.SSAFile()

        # Create karaoke style
        karaoke_style = cls._create_karaoke_style(config.style)
        subs.styles["Karaoke"] = karaoke_style

        for sup in supervisions:
            if sup.alignment and "word" in sup.alignment:
                # Generate \k tag text
                karaoke_text = cls._build_karaoke_text(
                    sup.alignment["word"], effect=config.style.effect
                )
                subs.append(
                    pysubs2.SSAEvent(
                        start=int(sup.start * 1000),
                        end=int(sup.end * 1000),
                        text=karaoke_text,
                        style="Karaoke",
                    )
                )
            else:
                # No alignment data, fallback to normal subtitle
                text = sup.text or ""
                if cls._should_include_speaker(sup, include_speaker):
                    text = f"{sup.speaker} {text}"
                subs.append(
                    pysubs2.SSAEvent(
                        start=int(sup.start * 1000),
                        end=int(sup.end * 1000),
                        text=text,
                    )
                )

        return subs.to_string(format_="ass").encode("utf-8")

    @classmethod
    def _create_karaoke_style(cls, style: KaraokeStyle) -> pysubs2.SSAStyle:
        """Create pysubs2 SSAStyle from KaraokeStyle config."""
        return pysubs2.SSAStyle(
            fontname=style.font_name,
            fontsize=style.font_size,
            primarycolor=cls._hex_to_ass_color(style.primary_color),
            secondarycolor=cls._hex_to_ass_color(style.secondary_color),
            outlinecolor=cls._hex_to_ass_color(style.outline_color),
            backcolor=cls._hex_to_ass_color(style.back_color),
            bold=style.bold,
            italic=style.italic,
            outline=style.outline_width,
            shadow=style.shadow_depth,
            alignment=style.alignment,
            marginl=style.margin_l,
            marginr=style.margin_r,
            marginv=style.margin_v,
        )

    @staticmethod
    def _hex_to_ass_color(hex_color: str) -> pysubs2.Color:
        """Convert #RRGGBB to pysubs2 Color."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return pysubs2.Color(r, g, b, 0)

    @staticmethod
    def _build_karaoke_text(words: list, effect: str = "sweep") -> str:
        """Build karaoke tag text from word alignment.

        Args:
            words: List of AlignmentItem objects
            effect: "sweep" for \\kf, "instant" for \\k, "outline" for \\ko

        Returns:
            Text with karaoke tags, e.g. "{\\kf45}Hello {\\kf55}world"
        """
        tag_map = {
            "sweep": "kf",
            "instant": "k",
            "outline": "ko",
        }
        tag = tag_map.get(effect, "kf")

        parts = []
        for word in words:
            duration_cs = int(word.duration * 100)  # Convert to centiseconds
            parts.append(f"{{\\{tag}{duration_cs}}}{word.symbol}")
        return "".join(parts)
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/caption/test_ass_karaoke.py -v
```

Expected: All tests PASS

**Step 5: 提交**

```bash
git add src/lattifai/caption/formats/pysubs2.py tests/caption/test_ass_karaoke.py
git commit -m "feat(caption): add ASS karaoke tags (\\kf, \\k, \\ko) support"
```

---

## Task 4: 实现 TTML Word Timing

**Files:**
- Modify: `src/lattifai/caption/formats/ttml.py`
- Test: `tests/caption/test_ttml_word_timing.py`

**Step 1: 写失败的测试**

```python
# tests/caption/test_ttml_word_timing.py
"""Tests for TTML word-level timing."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats.ttml import TTMLFormat
from lattifai.caption.formats.karaoke import KaraokeConfig
from lattifai.caption.supervision import Supervision


class TestTTMLWordTiming:
    """Test TTML word-level timing output."""

    def test_word_timing_attribute(self):
        """Word-level should add itunes:timing attribute."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = TTMLFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        assert 'itunes:timing="Word"' in content or "timing" in content.lower()

    def test_word_spans_in_output(self):
        """Each word should be wrapped in span with begin/end."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = TTMLFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        # Should have span elements with begin/end
        assert "<span" in content
        assert 'begin="00:00:15.200"' in content
        assert "Hello" in content
        assert "world" in content

    def test_paragraph_timing(self):
        """Paragraph should have overall begin/end."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = TTMLFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        # Paragraph should have begin/end
        assert "<p " in content
        assert 'begin="00:00:15.200"' in content

    def test_fallback_without_alignment(self):
        """Without alignment, should output normal TTML."""
        sups = [Supervision(text="No alignment", start=10.0, duration=2.0)]
        result = TTMLFormat.to_bytes(sups, word_level=True)
        content = result.decode("utf-8")

        assert "No alignment" in content
        # No span elements for words
        assert content.count("<span") <= 1  # May have speaker span

    def test_word_level_false_uses_original(self):
        """word_level=False should use original behavior."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = TTMLFormat.to_bytes(sups, word_level=False)
        content = result.decode("utf-8")

        # Should NOT have word spans
        assert 'itunes:timing="Word"' not in content
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/caption/test_ttml_word_timing.py -v
```

Expected: FAIL with "unexpected keyword argument 'word_level'"

**Step 3: 修改 ttml.py**

在 `src/lattifai/caption/formats/ttml.py` 中添加 word_level 支持：

```python
# 在文件顶部添加
from .karaoke import KaraokeConfig

# 添加 iTunes 命名空间
ITUNES_NS = "http://music.apple.com/lyric-ttml-internal"


# 修改 TTMLFormatBase._build_ttml 方法签名和实现
@classmethod
def _build_ttml(
    cls,
    supervisions: List[Supervision],
    config: TTMLConfig,
    include_speaker: bool = True,
    word_level: bool = False,
    karaoke_config: Optional[KaraokeConfig] = None,
) -> ET.Element:
    """Build TTML document structure."""
    kconfig = karaoke_config or KaraokeConfig()

    ET.register_namespace("", TTML_NS)
    ET.register_namespace("tts", TTML_STYLE_NS)
    ET.register_namespace("ttp", TTML_PARAM_NS)
    ET.register_namespace("xml", XML_NS)

    root = ET.Element(
        f"{{{TTML_NS}}}tt",
        attrib={
            f"{{{XML_NS}}}lang": config.language,
            f"{{{TTML_PARAM_NS}}}timeBase": "media",
        },
    )

    # Add itunes:timing attribute for word-level mode
    if word_level:
        ET.register_namespace("itunes", ITUNES_NS)
        root.set(f"{{{ITUNES_NS}}}timing", kconfig.ttml_timing_mode)

    if config.profile == "imsc1":
        root.set(f"{{{TTML_PARAM_NS}}}profile", "http://www.w3.org/ns/ttml/profile/imsc1/text")
    elif config.profile == "ebu-tt-d":
        root.set(f"{{{TTML_PARAM_NS}}}profile", "urn:ebu:tt:distribution:2014-01")

    # Head section (unchanged)
    head = ET.SubElement(root, f"{{{TTML_NS}}}head")
    styling = ET.SubElement(head, f"{{{TTML_NS}}}styling")
    cls._create_style_element(styling, "default", config.default_style)

    for speaker, style in config.speaker_styles.items():
        style_id = f"speaker_{speaker.replace(' ', '_')}"
        cls._create_style_element(styling, style_id, style)

    layout = ET.SubElement(head, f"{{{TTML_NS}}}layout")
    cls._create_region_element(layout, config.default_region)

    for speaker, region in config.speaker_regions.items():
        cls._create_region_element(layout, region)

    # Body section
    body = ET.SubElement(root, f"{{{TTML_NS}}}body")
    div = ET.SubElement(body, f"{{{TTML_NS}}}div")

    for sup in supervisions:
        begin = cls._seconds_to_ttml_time(sup.start)
        end = cls._seconds_to_ttml_time(sup.end)

        p = ET.SubElement(div, f"{{{TTML_NS}}}p")
        p.set("begin", begin)
        p.set("end", end)

        if sup.speaker and sup.speaker in config.speaker_regions:
            p.set("region", config.speaker_regions[sup.speaker].id)
        else:
            p.set("region", config.default_region.id)

        if sup.speaker and sup.speaker in config.speaker_styles:
            style_id = f"speaker_{sup.speaker.replace(' ', '_')}"
            p.set("style", style_id)
        else:
            p.set("style", "default")

        # Word-level timing mode
        if word_level and sup.alignment and "word" in sup.alignment:
            words = sup.alignment["word"]
            for i, word in enumerate(words):
                span = ET.SubElement(p, f"{{{TTML_NS}}}span")
                span.set("begin", cls._seconds_to_ttml_time(word.start))
                span.set("end", cls._seconds_to_ttml_time(word.start + word.duration))
                # Add space after word (except last)
                if i < len(words) - 1:
                    span.text = word.symbol + " "
                else:
                    span.text = word.symbol
        else:
            # Line timing mode (original behavior)
            include_this_speaker = cls._should_include_speaker(sup, include_speaker)

            if include_this_speaker and config.profile != "basic":
                span = ET.SubElement(p, f"{{{TTML_NS}}}span")
                span.set(f"{{{TTML_STYLE_NS}}}fontWeight", "bold")
                span.text = f"{sup.speaker} "
                span.tail = sup.text.strip() if sup.text else ""
            else:
                p.text = sup.text.strip() if sup.text else ""

    return root


# 修改 TTMLFormat.to_bytes 方法
@classmethod
def to_bytes(
    cls,
    supervisions: List[Supervision],
    include_speaker: bool = True,
    config: Optional[TTMLConfig] = None,
    word_level: bool = False,
    karaoke_config: Optional[KaraokeConfig] = None,
    **kwargs,
) -> bytes:
    """Convert to TTML format bytes."""
    if config is None:
        config = TTMLConfig()

    root = cls._build_ttml(
        supervisions,
        config,
        include_speaker=include_speaker,
        word_level=word_level,
        karaoke_config=karaoke_config,
    )
    xml_content = cls._prettify_xml(root)
    return xml_content.encode("utf-8")
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/caption/test_ttml_word_timing.py -v
```

Expected: All tests PASS

**Step 5: 提交**

```bash
git add src/lattifai/caption/formats/ttml.py tests/caption/test_ttml_word_timing.py
git commit -m "feat(caption): add TTML word-level timing (itunes:timing=\"Word\")"
```

---

## Task 5: 集成测试和文档

**Files:**
- Test: `tests/caption/test_word_level_integration.py`
- Modify: Update existing tests if needed

**Step 1: 写集成测试**

```python
# tests/caption/test_word_level_integration.py
"""Integration tests for word-level export across all formats."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats import get_writer
from lattifai.caption.formats.karaoke import KaraokeConfig, KaraokeStyle, KaraokeFonts
from lattifai.caption.supervision import Supervision


@pytest.fixture
def supervision_with_alignment():
    """Create test supervision with word alignment."""
    return Supervision(
        text="Hello beautiful world",
        start=15.2,
        duration=3.3,
        alignment={
            "word": [
                AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                AlignmentItem(symbol="beautiful", start=15.65, duration=0.75),
                AlignmentItem(symbol="world", start=16.4, duration=2.1),
            ]
        },
    )


class TestWordLevelIntegration:
    """Test word-level export across all supported formats."""

    def test_lrc_writer_registered(self):
        """LRC format should be registered."""
        writer = get_writer("lrc")
        assert writer is not None

    def test_all_formats_support_word_level(self, supervision_with_alignment):
        """All karaoke formats should support word_level parameter."""
        formats = ["lrc", "ass", "ttml"]
        for fmt in formats:
            writer = get_writer(fmt)
            assert writer is not None, f"Format {fmt} not registered"

            # Should not raise TypeError
            result = writer.to_bytes([supervision_with_alignment], word_level=True)
            assert isinstance(result, bytes)
            assert len(result) > 0

    def test_custom_karaoke_config(self, supervision_with_alignment):
        """Custom karaoke config should work across formats."""
        style = KaraokeStyle(
            effect="instant",
            primary_color="#FF00FF",
            font_name=KaraokeFonts.NOTO_SANS_SC,
        )
        config = KaraokeConfig(
            style=style,
            lrc_metadata={"ar": "Test Artist"},
        )

        # LRC
        lrc_writer = get_writer("lrc")
        lrc_result = lrc_writer.to_bytes(
            [supervision_with_alignment], word_level=True, karaoke_config=config
        )
        assert b"[ar:Test Artist]" in lrc_result

        # ASS
        ass_writer = get_writer("ass")
        ass_result = ass_writer.to_bytes(
            [supervision_with_alignment], word_level=True, karaoke_config=config
        )
        assert b"{\\k" in ass_result  # instant effect uses \k

    def test_graceful_fallback(self):
        """Formats should gracefully handle missing alignment."""
        sup_no_alignment = Supervision(text="No alignment data", start=0.0, duration=1.0)

        for fmt in ["lrc", "ass", "ttml"]:
            writer = get_writer(fmt)
            # Should not raise, should output line-level
            result = writer.to_bytes([sup_no_alignment], word_level=True)
            assert b"No alignment data" in result or b"No alignment" in result
```

**Step 2: 运行集成测试**

```bash
pytest tests/caption/test_word_level_integration.py -v
```

Expected: All tests PASS

**Step 3: 运行完整测试套件**

```bash
pytest tests/caption/ -v
```

Expected: All existing tests should still pass

**Step 4: 提交**

```bash
git add tests/caption/test_word_level_integration.py
git commit -m "test(caption): add word-level export integration tests"
```

---

## 完成检查清单

- [ ] Task 1: KaraokeConfig 配置模块
- [ ] Task 2: Enhanced LRC 格式
- [ ] Task 3: ASS 卡拉OK标签
- [ ] Task 4: TTML Word Timing
- [ ] Task 5: 集成测试

## 执行命令

完整测试：
```bash
cd /Users/feiteng/GEEK/OmniCaptions/lattifai-python
pytest tests/caption/test_karaoke_config.py tests/caption/test_lrc_format.py tests/caption/test_ass_karaoke.py tests/caption/test_ttml_word_timing.py tests/caption/test_word_level_integration.py -v
```
