"""JSON format handler for structured caption data.

JSON is the most flexible format for storing caption data, supporting:
- Segment-level timing (start, end)
- Word-level alignment (words array with per-word timestamps)
- Speaker labels
- Custom metadata

Example JSON structure:
```json
[
    {
        "text": "Hello world",
        "start": 0.0,
        "end": 2.5,
        "speaker": "Speaker 1",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 2.5}
        ]
    }
]
```
"""

import json
from pathlib import Path
from typing import List

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


@register_format("json")
class JSONFormat(FormatHandler):
    """JSON format for structured caption data.

    Features:
    - Preserves full segment structure with timing
    - Supports word-level alignment in 'words' field
    - Round-trip compatible (read/write preserves all data)
    - Human-readable with indentation

    Input format (read):
    - Array of objects with: text, start, duration/end
    - Optional: speaker, words (array of word timing objects)
    - Words can have: word, start, duration or end

    Output format (write):
    - word_level=False: Standard segment output
    - word_level=True: Includes 'words' array with per-word timestamps
    """

    extensions = [".json"]
    description = "JSON - structured caption data with word-level support"

    @classmethod
    def read(cls, source, normalize_text: bool = True, **kwargs) -> List[Supervision]:
        """Read JSON format.

        Args:
            source: File path or JSON string content
            normalize_text: Whether to normalize text content

        Returns:
            List of Supervision objects with alignment data if present

        Supports word-level alignment data in the 'words' field.
        Each word item should have: word, start, duration (or end).
        """
        from lhotse.supervision import AlignmentItem

        if cls.is_content(source):
            data = json.loads(source)
        else:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)

        supervisions = []
        for item in data:
            text = item.get("text", "")
            if normalize_text:
                text = normalize_text_fn(text)

            # Parse word-level alignment if present
            alignment = None
            if "words" in item and item["words"]:
                word_alignments = []
                for word_item in item["words"]:
                    word_text = word_item.get("word", "")
                    word_start = word_item.get("start", 0)
                    # Support both 'duration' and 'end' fields
                    if "duration" in word_item:
                        word_duration = word_item["duration"]
                    elif "end" in word_item:
                        word_duration = word_item["end"] - word_start
                    else:
                        word_duration = 0
                    word_alignments.append(AlignmentItem(symbol=word_text, start=word_start, duration=word_duration))
                if word_alignments:
                    alignment = {"word": word_alignments}

            # Support both 'duration' and 'end' fields for segment timing
            start = item.get("start", 0)
            if "duration" in item:
                duration = item["duration"]
            elif "end" in item:
                duration = item["end"] - start
            else:
                duration = 0

            supervisions.append(
                Supervision(
                    text=text,
                    start=start,
                    duration=duration,
                    speaker=item.get("speaker"),
                    alignment=alignment,
                )
            )

        return supervisions

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        word_level: bool = False,
        **kwargs,
    ) -> Path:
        """Write JSON format.

        Args:
            supervisions: List of Supervision objects
            output_path: Output file path
            include_speaker: Whether to include speaker field
            word_level: If True, include 'words' field with word-level timestamps

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker, word_level=word_level)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls, supervisions: List[Supervision], include_speaker: bool = True, word_level: bool = False, **kwargs
    ) -> bytes:
        """Convert to JSON format bytes.

        Args:
            supervisions: List of Supervision objects
            include_speaker: Whether to include speaker field
            word_level: If True, include 'words' field with word-level timestamps

        Returns:
            JSON content as UTF-8 encoded bytes

        Note:
            Unlike other formats (SRT, VTT, LRC) that expand word_level=True to
            one segment per word, JSON preserves the original structure and adds
            a 'words' array inside each segment. This allows round-trip compatibility
            and preserves all timing information.
        """
        data = []
        for sup in supervisions:
            item = {
                "text": sup.text,
                "start": sup.start,
                "end": sup.end,
            }
            if include_speaker and sup.speaker:
                item["speaker"] = sup.speaker

            # Add words field when word_level=True and alignment exists
            if word_level and sup.alignment and "word" in sup.alignment:
                item["words"] = [
                    {
                        "word": w.symbol,
                        "start": w.start,
                        "end": w.start + w.duration,
                    }
                    for w in sup.alignment["word"]
                ]

            data.append(item)

        return json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8")
