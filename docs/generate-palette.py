#!/usr/bin/env python
"""Generate speaker color palette and karaoke preset preview images.

Usage:
    python docs/generate-palette.py

Output:
    docs/speaker-palette-zh.png   — Speaker colors (Chinese)
    docs/speaker-palette-en.png   — Speaker colors (English)
    docs/karaoke-presets-zh.png   — Karaoke presets (Chinese)
    docs/karaoke-presets-en.png   — Karaoke presets (English)
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ─── Speaker palette ───────────────────────────────────────────────
# Keep in sync with lattifai-captions pysubs2.py ASSFormat._SPEAKER_PALETTE
SPEAKER_COLORS = [
    ("#1387C0", "晴空海蓝", "Azure"),
    ("#FFC209", "金柠暖阳", "Warm Yellow"),
    ("#F7C3D9", "柔樱粉", "Soft Pink"),
    ("#9DC92A", "苹果绿", "Apple Green"),
    ("#A1FEEF", "薄荷冰青", "Mint Ice"),
    ("#F4520D", "暖橙光", "Warm Orange"),
    ("#658AE4", "柔空蓝", "Sky Blue"),
    ("#FBC03D", "栀子黄", "Gardenia"),
    ("#CC5D84", "琅玕紫", "Langgan Purple"),
    ("#008C8C", "马尔斯绿", "Mars Green"),
]

# ─── Karaoke presets ───────────────────────────────────────────────
# Keep in sync with lattifai-captions config.py KARAOKE_PRESETS
KARAOKE_PRESETS = [
    ("azure-gold", "晴空金柠", "Azure Gold", "#FFFFFF", "#FFC209", "#1387C0"),
    ("sakura-purple", "樱花紫鸢", "Sakura Purple", "#F7C3D9", "#7953B1", "#063C85"),
    ("mint-ocean", "薄荷深海", "Mint Ocean", "#A1FEEF", "#658AE4", "#28314E"),
    ("gardenia-green", "栀子碧山", "Gardenia Green", "#FFFFFF", "#9DC92A", "#77964A"),
    ("sunset-warm", "暖阳橙光", "Sunset Warm", "#FAEDD1", "#F4520D", "#1387C0"),
    ("prussian-elegant", "普鲁士蓝", "Prussian Blue", "#FFFFFF", "#FBC03D", "#003153"),
    ("burgundy-classic", "勃艮第红", "Burgundy Classic", "#F7F2DF", "#CC5D84", "#800020"),
    ("china-red", "提香红", "China Red", "#FFFFFF", "#FEA72E", "#B05923"),
    ("mars-teal", "马尔斯绿", "Mars Teal", "#FFFFFF", "#008C8C", "#003153"),
    ("spring-field", "春野绿", "Spring Field", "#FBFFF2", "#46B065", "#008E6B"),
    ("navy-pink", "藏蓝樱粉", "Navy Pink", "#FFFFFF", "#F7C3D9", "#063C85"),
    ("apricot-dark", "杏黄玄青", "Apricot Dark", "#FEA72E", "#F7F2DF", "#3A3C50"),
]

# ─── Fonts ─────────────────────────────────────────────────────────
CJK_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]
MONO_FONT = "/System/Library/Fonts/Menlo.ttc"
LATIN_FONT = "/System/Library/Fonts/Helvetica.ttc"
BG = "#1A1A2E"


def find_cjk_font():
    return next((f for f in CJK_CANDIDATES if os.path.exists(f)), None)


def load_font(path, size):
    if path and os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def luma(hex_color: str) -> float:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b


def text_color(bg_hex: str) -> str:
    return "#1A1A2E" if luma(bg_hex) > 140 else "#FFFFFF"


# ─── Speaker palette generator ────────────────────────────────────
def generate_speaker(lang: str, out_path: Path, scale: int = 5):
    S = scale
    cjk = find_cjk_font()

    zh = lang == "zh"
    ft_title = load_font(cjk if zh else LATIN_FONT, 22 * S)
    ft_name = load_font(cjk if zh else LATIN_FONT, (16 if zh else 14) * S)
    ft_sub = load_font(cjk if zh else LATIN_FONT, 12 * S)
    ft_hex = load_font(MONO_FONT, 11 * S)

    title = "LattifAI 说话人配色方案" if zh else "LattifAI Speaker Color Palette"
    subtitle = (
        f"{len(SPEAKER_COLORS)} 色高级感配色 · speaker_color=auto"
        if zh
        else f"{len(SPEAKER_COLORS)}-color palette · speaker_color=auto"
    )

    cols = 5
    sw, sh = 150 * S, 100 * S
    px, py = 18 * S, 50 * S
    W = cols * sw + (cols - 1) * px + 80 * S
    rows = (len(SPEAKER_COLORS) + cols - 1) // cols
    H = 70 * S + rows * (sh + py)

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((W // 2, 22 * S), title, fill="#FFFFFF", font=ft_title, anchor="mt")
    draw.text((W // 2, 48 * S), subtitle, fill="#777788", font=ft_sub, anchor="mt")

    sx = (W - cols * sw - (cols - 1) * px) // 2
    sy = 70 * S

    for i, (hx, cn, en) in enumerate(SPEAKER_COLORS):
        c, r = i % cols, i // cols
        x, y = sx + c * (sw + px), sy + r * (sh + py)
        draw.rounded_rectangle([x, y, x + sw, y + sh], radius=12 * S, fill=hx)
        tc = text_color(hx)
        cx = x + sw // 2
        draw.text((cx, y + sh // 2 - 10 * S), cn if zh else en, fill=tc, font=ft_name, anchor="mt")
        draw.text((cx, y + sh // 2 + 14 * S), hx, fill=tc, font=ft_hex, anchor="mt")

    img.save(out_path, "PNG")
    print(f"Saved: {out_path} ({W}x{H})")


# ─── Karaoke presets generator ─────────────────────────────────────
def generate_karaoke(lang: str, out_path: Path, scale: int = 5):
    S = scale
    cjk = find_cjk_font()
    zh = lang == "zh"

    ft_title = load_font(cjk if zh else LATIN_FONT, 22 * S)
    ft_sub = load_font(cjk if zh else LATIN_FONT, 12 * S)
    ft_preset_name = load_font(cjk, 13 * S)  # Always CJK — shows Chinese name in both langs
    ft_sample = load_font(cjk if zh else LATIN_FONT, 16 * S)
    ft_label = load_font(LATIN_FONT, 10 * S)
    ft_hex = load_font(MONO_FONT, 9 * S)

    title = "LattifAI 卡拉OK配色方案" if zh else "LattifAI Karaoke Color Presets"
    subtitle = "karaoke.preset=<name> · 8 套预设配色" if zh else "karaoke.preset=<name> · 8 presets"

    cols = 4
    cw, ch = 230 * S, 130 * S
    px, py = 14 * S, 56 * S
    W = cols * cw + (cols - 1) * px + 80 * S
    rows = (len(KARAOKE_PRESETS) + cols - 1) // cols
    H = 70 * S + rows * (ch + py)

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((W // 2, 22 * S), title, fill="#FFFFFF", font=ft_title, anchor="mt")
    draw.text((W // 2, 48 * S), subtitle, fill="#777788", font=ft_sub, anchor="mt")

    sx = (W - cols * cw - (cols - 1) * px) // 2
    sy = 70 * S

    sample_unsung = "Hello 你好" if zh else "Hello World"
    sample_sung = "World 世界" if zh else "Karaoke"

    for i, (key, cn, en, primary, secondary, outline) in enumerate(KARAOKE_PRESETS):
        c, r = i % cols, i // cols
        x, y = sx + c * (cw + px), sy + r * (ch + py)

        # Card background (dark, simulating video)
        draw.rounded_rectangle([x, y, x + cw, y + ch], radius=12 * S, fill="#0D0D1A")

        # Outline border to hint the outline color
        draw.rounded_rectangle([x, y, x + cw, y + ch], radius=12 * S, outline=outline, width=2 * S)

        cx = x + cw // 2

        # Sample text: unsung (primary) + sung (secondary), centered
        full_text = sample_unsung + " " + sample_sung
        unsung_w = draw.textlength(sample_unsung, ft_sample)
        full_w = draw.textlength(full_text, ft_sample)
        tx = x + (cw - full_w) // 2
        ty = y + 18 * S
        gap = draw.textlength(" ", ft_sample)
        # Outline shadow
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)]:
            d = S
            draw.text((tx + dx * d, ty + dy * d), sample_unsung, fill=outline, font=ft_sample)
            draw.text((tx + dx * d + unsung_w + gap, ty + dy * d), sample_sung, fill=outline, font=ft_sample)
        draw.text((tx, ty), sample_unsung, fill=primary, font=ft_sample)
        draw.text((tx + unsung_w + gap, ty), sample_sung, fill=secondary, font=ft_sample)

        # Color swatches row, centered
        sw_y = y + 66 * S
        sw_sz = 14 * S
        swatch_gap = 60 * S
        total_sw_w = 3 * (sw_sz + 4 * S + draw.textlength("#FFFFFF", ft_hex)) + 2 * (
            swatch_gap - sw_sz - 4 * S - draw.textlength("#FFFFFF", ft_hex)
        )
        # Simpler: just compute total width of 3 swatches at fixed spacing
        total_sw_w = 2 * swatch_gap + sw_sz + draw.textlength("#FFFFFF", ft_hex) + 4 * S
        sw_start_x = x + (cw - total_sw_w) // 2
        labels = [("P", primary), ("S", secondary), ("O", outline)]
        for j, (lbl, clr) in enumerate(labels):
            lx = sw_start_x + j * swatch_gap
            draw.rounded_rectangle([lx, sw_y, lx + sw_sz, sw_y + sw_sz], radius=3 * S, fill=clr)
            draw.text((lx + sw_sz + 4 * S, sw_y + 1 * S), clr, fill="#777788", font=ft_hex)

        # Preset name below card (always show both zh + en)
        draw.text((cx, y + ch + 6 * S), cn, fill="#BBBBCC", font=ft_preset_name, anchor="mt")
        draw.text((cx, y + ch + 22 * S), key, fill="#555566", font=ft_label, anchor="mt")

    img.save(out_path, "PNG")
    print(f"Saved: {out_path} ({W}x{H})")


def main():
    docs = Path(__file__).parent
    generate_speaker("zh", docs / "speaker-palette-zh.png")
    generate_speaker("en", docs / "speaker-palette-en.png")
    generate_karaoke("zh", docs / "karaoke-presets-zh.png")
    generate_karaoke("en", docs / "karaoke-presets-en.png")


if __name__ == "__main__":
    main()
