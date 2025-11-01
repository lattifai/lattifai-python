import json
from abc import ABCMeta
from typing import Any, List, Optional

import pysubs2
from lhotse.supervision import AlignmentItem
from lhotse.utils import Pathlike

from .reader import Supervision


def s_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    ms = pysubs2.make_time(s=seconds)
    return pysubs2.time.ms_to_str(ms)


class SubtitleWriter(ABCMeta):
    """Class for writing subtitle files with optional word-level alignment."""

    @classmethod
    def write(cls, alignments: List[Supervision], output_path: Pathlike) -> Pathlike:
        if str(output_path)[-4:].lower() == '.txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sup in alignments:
                    word_items = parse_alignment_from_supervision(sup)
                    if word_items:
                        for item in word_items:
                            f.write(f'[{item.start:.2f}-{item.end:.2f}] {item.symbol}\n')
                    else:
                        f.write(f'[{sup.start:.2f}-{sup.end:.2f}] {sup.text}\n')

        elif str(output_path)[-5:].lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                # Enhanced JSON export with word-level alignment
                json_data = []
                for sup in alignments:
                    sup_dict = sup.to_dict()
                    json_data.append(sup_dict)
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        elif str(output_path).endswith('.TextGrid') or str(output_path).endswith('.textgrid'):
            from tgt import Interval, IntervalTier, TextGrid, write_to_file

            tg = TextGrid()
            supervisions, words = [], []
            for supervision in sorted(alignments, key=lambda x: x.start):
                supervisions.append(Interval(supervision.start, supervision.end, supervision.text or ''))
                # Extract word-level alignment using helper function
                word_items = parse_alignment_from_supervision(supervision)
                if word_items:
                    for item in word_items:
                        words.append(Interval(item.start, item.end, item.symbol))

            tg.add_tier(IntervalTier(name='utterances', objects=supervisions))
            if words:
                tg.add_tier(IntervalTier(name='words', objects=words))
            write_to_file(tg, output_path, format='long')
        else:
            subs = pysubs2.SSAFile()
            for sup in alignments:
                # Add word-level timing as metadata in the subtitle text
                word_items = parse_alignment_from_supervision(sup)
                if word_items:
                    for word in word_items:
                        subs.append(
                            pysubs2.SSAEvent(start=int(word.start * 1000), end=int(word.end * 1000), text=word.symbol)
                        )
                else:
                    subs.append(
                        pysubs2.SSAEvent(start=int(sup.start * 1000), end=int(sup.end * 1000), text=sup.text or '')
                    )
            subs.save(output_path)

        return output_path


def parse_alignment_from_supervision(supervision: Any) -> Optional[List[AlignmentItem]]:
    """
    Extract word-level alignment items from Supervision object.

    Args:
        supervision: Supervision object with potential alignment data

    Returns:
        List of AlignmentItem objects, or None if no alignment data present
    """
    if not hasattr(supervision, 'alignment') or not supervision.alignment:
        return None

    if 'word' not in supervision.alignment:
        return None

    return supervision.alignment['word']
