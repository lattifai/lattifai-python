import logging
import re
from typing import Optional, Tuple

# 来自于字幕中常见的说话人标记格式
SPEAKER_PATTERN = re.compile(r"((?:>>|&gt;&gt;|>|&gt;).*?[:：])\s*(.*)")

# Transcriber Output Example:
# 26:19.919 --> 26:34.921
# [SPEAKER_01]: 越来越多的科技巨头入...
SPEAKER_LATTIFAI = re.compile(r"(^\[SPEAKER_.*?\][:：])\s*(.*)")

# NISHTHA BHATIA: Hey, everyone.
# DIETER: Oh, hey, Nishtha.
# GEMINI: That might
SPEAKER_PATTERN2 = re.compile(r"^([A-Z]{1,15}(?:\s+[A-Z]{1,15})?[:：])\s*(.*)$")


def normalize_html_text(text: str) -> str:
    """Normalize HTML text by decoding entities and stripping whitespace."""
    html_entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "\\N": " ",
        "…": " ",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space

    # Convert curly apostrophes to straight apostrophes for common English contractions
    # Handles: 't 's 'll 're 've 'd 'm
    # For example, convert "don't" to "don't"
    text = re.sub(r"([a-zA-Z])’([tsdm]|ll|re|ve)\b", r"\1'\2", text, flags=re.IGNORECASE)
    # For example, convert "5’s" to "5's"
    text = re.sub(r"([0-9])’([s])\b", r"\1'\2", text, flags=re.IGNORECASE)

    return text.strip()


def parse_speaker_text(line) -> Tuple[Optional[str], str]:
    """Parse a line of text to extract speaker and content."""

    if ":" not in line and "：" not in line:
        return None, line

    # 匹配以 >> 开头的行，并去除开头的名字和冒号
    match = SPEAKER_PATTERN.match(line)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    match = SPEAKER_LATTIFAI.match(line)
    if match:
        assert len(match.groups()) == 2, match.groups()
        if not match.group(1):
            logging.error(f"ParseSub LINE [{line}]")
        else:
            return match.group(1).strip(), match.group(2).strip()

    match = SPEAKER_PATTERN2.match(line)
    if match:
        assert len(match.groups()) == 2, match.groups()
        return match.group(1).strip(), match.group(2).strip()

    return None, line


if __name__ == "__main__":
    pattern = re.compile(r">>\s*(.*?)\s*[:：]\s*(.*)")
    pattern = re.compile(r"(>>.*?[:：])\s*(.*)")

    test_strings = [
        ">>Key: Value",
        ">>  Key with space : Value with space ",
        ">>  全角键 ： 全角值",
        ">>Key：Value xxx. >>Key：Value",
    ]

    for text in test_strings:
        match = pattern.match(text)
        if match:
            print(f"Input: '{text}'")
            print(f"  Key:   '{match.group(1)}'")
            print(f"  Value: '{match.group(2)}'")
            print("-------------")

    # pattern2
    test_strings2 = ["NISHTHA BHATIA: Hey, everyone.", "DIETER: Oh, hey, Nishtha.", "GEMINI: That might"]
    for text in test_strings2:
        match = SPEAKER_PATTERN2.match(text)
        if match:
            print(f"  Input: '{text}'")
            print(f"Speaker: '{match.group(1)}'")
            print(f"Content: '{match.group(2)}'")
            print("-------------")
        else:
            raise ValueError(f"No match for: '{text}'")
