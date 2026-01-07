# Multilingual punctuation characters (no duplicates)
PUNCTUATION = (
    # ASCII punctuation
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    # Chinese/CJK punctuation
    "。，、？！：；·￥…—～“”‘’"
    "《》〈〉【】〔〕〖〗（）"  # Brackets
    # Japanese punctuation (unique only)
    "「」『』・"
    # CJK punctuation (unique only)
    "〃〆〇〒〓〘〙〚〛〜〝〞〟"
    # Arabic punctuation
    "،؛؟"
    # Thai punctuation
    "๏๚๛"
    # Hebrew punctuation
    "־׀׃׆"
    # Other common punctuation
    "¡¿"  # Spanish inverted marks
    "«»‹›"  # Guillemets
    "‐‑‒–―"  # Dashes (excluding — already above)
    "‚„"  # Low quotation marks
    "†‡•‣"  # Daggers and bullets
    "′″‴"  # Prime marks
    "‰‱"  # Per mille
)
PUNCTUATION_SPACE = PUNCTUATION + " "
STAR_TOKEN = "※"

# End of sentence punctuation marks (multilingual)
# - ASCII: .!?"']）
# - Chinese/CJK: 。！？"】」』〗〙〛 (including right double quote U+201D)
# - Japanese: ｡ (halfwidth period)
# - Arabic: ؟
# - Ellipsis: …
END_PUNCTUATION = ".!?\"']）。！？\u201d】」』〗〙〛｡؟…"

GROUPING_SEPARATOR = "✹"
