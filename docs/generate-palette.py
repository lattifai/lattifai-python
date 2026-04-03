#!/usr/bin/env python
"""Generate speaker color palette and karaoke color scheme preview images.

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
# Format: (key, cn_name, en_name, primary, secondary, outline, back)
KARAOKE_PRESETS = [
    ("azure-gold", "晴空金柠", "Azure Gold", "#FFFFFF", "#FFC209", "#1387C0", "#0A3D5C"),
    ("sakura-purple", "樱花紫鸢", "Sakura Purple", "#F7C3D9", "#7953B1", "#063C85", "#1A1A2E"),
    ("mint-ocean", "薄荷深海", "Mint Ocean", "#A1FEEF", "#658AE4", "#28314E", "#0A0A1A"),
    ("gardenia-green", "栀子碧山", "Gardenia Green", "#FFFFFF", "#9DC92A", "#77964A", "#1C2B1A"),
    ("sunset-warm", "暖阳橙光", "Sunset Warm", "#FAEDD1", "#F4520D", "#1387C0", "#0A1628"),
    ("prussian-elegant", "普鲁士蓝", "Prussian Blue", "#FFFFFF", "#FBC03D", "#003153", "#001A2C"),
    ("burgundy-classic", "勃艮第红", "Burgundy Classic", "#F7F2DF", "#CC5D84", "#800020", "#2A000D"),
    ("langgan-spring", "琅玕春辰", "Langgan Spring", "#C1D796", "#CC5D84", "#8A3A5A", "#2A1020"),
    ("mars-teal", "马尔斯绿", "Mars Teal", "#FFFFFF", "#008C8C", "#003153", "#001A1A"),
    ("spring-field", "春野绿", "Spring Field", "#FBFFF2", "#46B065", "#008E6B", "#0A2A1A"),
    ("navy-pink", "藏蓝樱粉", "Navy Pink", "#FFFFFF", "#F7C3D9", "#063C85", "#021A3A"),
    ("apricot-dark", "杏黄玄青", "Apricot Dark", "#FEA72E", "#F7F2DF", "#3A3C50", "#1A1A28"),
]

# ─── Fonts ─────────────────────────────────────────────────────────
CJK_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]
CJK_BOLD_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
MONO_FONT = "/System/Library/Fonts/Menlo.ttc"
LATIN_FONT = "/System/Library/Fonts/Helvetica.ttc"
BG = "#F5F5F8"


def find_cjk_font():
    return next((f for f in CJK_CANDIDATES if os.path.exists(f)), None)


def find_cjk_bold_font():
    return next((f for f in CJK_BOLD_CANDIDATES if os.path.exists(f)), None)


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
    draw.text((W // 2, 22 * S), title, fill="#1A1A2E", font=ft_title, anchor="mt")
    draw.text((W // 2, 48 * S), subtitle, fill="#888899", font=ft_sub, anchor="mt")

    sx = (W - cols * sw - (cols - 1) * px) // 2
    sy = 70 * S

    for i, (hx, cn, en) in enumerate(SPEAKER_COLORS):
        c, r = i % cols, i // cols
        x, y = sx + c * (sw + px), sy + r * (sh + py)
        # Add subtle shadow for depth on light background
        draw.rounded_rectangle([x + 2 * S, y + 2 * S, x + sw + 2 * S, y + sh + 2 * S], radius=12 * S, fill="#D0D0D8")
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
    cjk_bold = find_cjk_bold_font() or cjk
    ft_scheme_name = load_font(cjk_bold, 14 * S)  # Bold CJK — scheme name below card
    ft_sample = load_font(cjk if zh else LATIN_FONT, 18 * S)
    ft_label = load_font(LATIN_FONT, 11 * S)
    ft_hex = load_font(MONO_FONT, 10 * S)
    ft_swatch_label = load_font(cjk if zh else LATIN_FONT, 9 * S)

    title = "LattifAI 卡拉OK配色方案" if zh else "LattifAI Karaoke Color Schemes"
    subtitle = "karaoke.color_scheme=<name> · 12 套配色方案" if zh else "karaoke.color_scheme=<name> · 12 schemes"

    cols = 4
    cw, ch = 260 * S, 160 * S
    px, py = 14 * S, 46 * S
    W = cols * cw + (cols - 1) * px + 80 * S
    rows = (len(KARAOKE_PRESETS) + cols - 1) // cols
    H = 70 * S + rows * (ch + py)

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((W // 2, 22 * S), title, fill="#1A1A2E", font=ft_title, anchor="mt")
    draw.text((W // 2, 48 * S), subtitle, fill="#888899", font=ft_sub, anchor="mt")

    sx = (W - cols * cw - (cols - 1) * px) // 2
    sy = 70 * S

    sample_unsung = "Hello 你好" if zh else "Hello World"
    sample_sung = "World 世界" if zh else "Karaoke"

    for i, (key, cn, en, primary, secondary, outline, back) in enumerate(KARAOKE_PRESETS):
        c, r = i % cols, i // cols
        x, y = sx + c * (cw + px), sy + r * (ch + py)

        # Card background: use back_color (each scheme's designed shadow tone)
        # Lighten it slightly so it's not too dark for a preview card
        br, bg_, bb = int(back[1:3], 16), int(back[3:5], 16), int(back[5:7], 16)
        card_bg = f"#{min(br + 40, 255):02x}{min(bg_ + 40, 255):02x}{min(bb + 40, 255):02x}"
        draw.rounded_rectangle([x + 3 * S, y + 3 * S, x + cw + 3 * S, y + ch + 3 * S], radius=12 * S, fill="#C0C0C8")
        draw.rounded_rectangle([x, y, x + cw, y + ch], radius=12 * S, fill=card_bg)

        # Border from secondary color for visual pop
        draw.rounded_rectangle([x, y, x + cw, y + ch], radius=12 * S, outline=secondary, width=2 * S)

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

        # Color swatch bars — 3 rounded pills with full label + hex
        zh_labels = ["文字色", "高亮色", "描边色"]
        en_labels = ["Text", "Highlight", "Outline"]
        bar_y = y + 78 * S
        bar_h = 28 * S
        bar_gap = 6 * S
        bar_w = (cw - 2 * 12 * S - 2 * bar_gap) // 3
        labels_colors = list(zip(zh_labels if zh else en_labels, [primary, secondary, outline]))
        for j, (lbl, clr) in enumerate(labels_colors):
            bx = x + 12 * S + j * (bar_w + bar_gap)
            draw.rounded_rectangle([bx, bar_y, bx + bar_w, bar_y + bar_h], radius=4 * S, fill=clr)
            tc = "#1A1A2E" if luma(clr) > 120 else "#FFFFFF"
            bcx = bx + bar_w // 2
            bcy = bar_y + bar_h // 2
            draw.text((bcx, bcy), clr, fill=tc, font=ft_hex, anchor="mm")
            # Label above the bar — use white for visibility on any card bg
            draw.text((bcx, bar_y - 3 * S), lbl, fill="#DDDDEE", font=ft_swatch_label, anchor="mb")

        # Scheme name below card
        # Ensure text is readable on light page background: darken light colors
        sr, sg, sb = int(secondary[1:3], 16), int(secondary[3:5], 16), int(secondary[5:7], 16)
        name_color = secondary if luma(secondary) < 180 else f"#{sr*2//3:02x}{sg*2//3:02x}{sb*2//3:02x}"
        dim = (
            f"#{sr*2//3:02x}{sg*2//3:02x}{sb*2//3:02x}"
            if luma(secondary) < 180
            else f"#{sr//2:02x}{sg//2:02x}{sb//2:02x}"
        )
        draw.text((cx, y + ch + 8 * S), cn, fill=name_color, font=ft_scheme_name, anchor="mt")
        draw.text((cx, y + ch + 26 * S), key, fill=dim, font=ft_label, anchor="mt")

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
