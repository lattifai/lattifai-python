import logging
import re
from typing import Optional, Tuple

# Timestamp pattern: [start-end] text
# Example: [1.23-4.56] Hello world
TIMESTAMP_PATTERN = re.compile(r"^\[([\d.]+)-([\d.]+)\]\s*(.*)$")

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


def normalize_text(text: str) -> str:
    """Normalize caption text by:
    - Decoding common HTML entities
    - Removing HTML tags (e.g., <i>, <font>, <b>, <br>)
    - Collapsing multiple whitespace into a single space
    - Converting curly apostrophes to straight ones in common contractions
    """
    if not text:
        return ""

    # # Remove HTML tags first (replace with space to avoid concatenation)
    # text = re.sub(r"<[^>]+>", " ", text)

    html_entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "\\N": " ",
        "…": " ",  # replace ellipsis with space to avoid merging words
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    # Convert curly apostrophes to straight apostrophes for common English contractions
    text = re.sub(r"([a-zA-Z])’([tsdm]|ll|re|ve)\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"([0-9])’([s])\b", r"\1'\2", text, flags=re.IGNORECASE)

    # Collapse whitespace (after replacements)
    text = re.sub(r"\s+", " ", text)

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


def parse_timestamp_text(line: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Parse a line of text to extract timestamp and content.

    Format: [start-end] text
    Example: [1.23-4.56] Hello world

    Args:
        line: Input line to parse

    Returns:
        Tuple of (start_time, end_time, text)
        - start_time: Start timestamp in seconds, or None if not found
        - end_time: End timestamp in seconds, or None if not found
        - text: The text content after the timestamp
    """
    match = TIMESTAMP_PATTERN.match(line)
    if match:
        try:
            start = float(match.group(1))
            end = float(match.group(2))
            text = match.group(3).strip()
            return start, end, text
        except ValueError:
            # If conversion fails, treat as plain text
            return None, None, line

    return None, None, line


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
            print(f"Speaker:   '{match.group(1)}'")
            print(f"Content: '{match.group(2)}'")
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
