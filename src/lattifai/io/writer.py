from abc import ABCMeta
from typing import List

from lhotse.utils import Pathlike

from .reader import SubtitleFormat, Supervision


class SubtitleWriter(ABCMeta):
    """Class for writing subtitle files."""

    @classmethod
    def write(cls, alignments: List[Supervision], output_path: Pathlike) -> Pathlike:
        if str(output_path)[-4:].lower() == '.txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sup in alignments:
                    f.write(f'{sup.text}\n')
        elif str(output_path)[-5:].lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                import json

                json.dump([sup.to_dict() for sup in alignments], f, ensure_ascii=False, indent=4)
        elif str(output_path).endswith('.TextGrid') or str(output_path).endswith('.textgrid'):
            from tgt import Interval, IntervalTier, TextGrid, write_to_file

            tg = TextGrid()
            supervisions, words = [], []
            for supervision in sorted(alignments, key=lambda x: x.start):
                supervisions.append(Interval(supervision.start, supervision.end, supervision.text or ''))
                if supervision.alignment and 'word' in supervision.alignment:
                    for alignment in supervision.alignment['word']:
                        words.append(Interval(alignment.start, alignment.end, alignment.symbol))

            tg.add_tier(IntervalTier(name='utterances', objects=supervisions))
            if words:
                tg.add_tier(IntervalTier(name='words', objects=words))
            write_to_file(tg, output_path, format='long')
        else:
            import pysubs2

            subs = pysubs2.SSAFile()
            for sup in alignments:
                start = int(sup.start * 1000)
                end = int(sup.end * 1000)
                text = sup.text or ''
                subs.append(pysubs2.SSAEvent(start=start, end=end, text=text))
            subs.save(output_path)

        return output_path
