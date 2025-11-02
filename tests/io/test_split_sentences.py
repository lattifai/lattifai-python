import copy
import sys
import types
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pytest

if 'k2' not in sys.modules:
    sys.modules['k2'] = types.ModuleType('k2')


from lattifai.io import Supervision
from lattifai.tokenizer.tokenizer import LatticeTokenizer


class FakeSplitter:
    def __init__(self, outputs: List[List[str]]):
        self._outputs = outputs
        self.calls = []

    def split(self, texts: List[str], threshold: float, strip_whitespace: bool) -> List[List[str]]:
        self.calls.append(
            {
                'texts': list(texts),
                'threshold': threshold,
                'strip_whitespace': strip_whitespace,
            }
        )
        return copy.deepcopy(self._outputs)


def make_supervision(idx: int, text: str, speaker: Optional[str]) -> Supervision:
    return Supervision(
        id=f'sup-{idx}',
        recording_id='rec',
        start=float(idx),
        duration=1.0,
        channel=0,
        text=text,
        speaker=speaker,
    )


def texts_and_speakers(items: List[Supervision]) -> List[Tuple[str, Optional[str]]]:
    return [(sup.text, sup.speaker) for sup in items]


def test_split_sentences_keeps_initial_speaker_for_multi_sentence_chunk():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([['Hello world.', 'This is second sentence!']])

    supervisions = [
        make_supervision(0, 'Hello world.', speaker='Alice'),
        make_supervision(1, 'This is second sentence!', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ('Hello world.', 'Alice'),
        ('This is second sentence!', None),
    ]


def test_split_sentences_emits_trailing_remainder_without_punctuation():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([['Trailing remainder']])

    supervisions = [make_supervision(0, 'Trailing remainder', speaker=None)]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [('Trailing remainder', None)]


def test_split_sentences_resplits_special_colon_sequences():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([['[APPLAUSE] &gt;&gt; SPEAKER:', 'We are live.']])

    supervisions = [
        make_supervision(0, '[APPLAUSE] &gt;&gt; SPEAKER:', speaker='Host'),
        make_supervision(1, 'We are live.', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ('[APPLAUSE]', 'Host'),
        ('&gt;&gt; SPEAKER: We are live.', None),
    ]


def test_split_sentences_outputs_remainder_before_next_speaker():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([['Incomplete thought'], ['Replies with closure.']])

    supervisions = [
        make_supervision(0, 'Incomplete thought', speaker='Alice'),
        make_supervision(1, '', speaker=None),
        make_supervision(2, 'Replies with closure.', speaker='Bob'),
        make_supervision(3, '', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ('Incomplete thought', 'Alice'),
        ('Replies with closure.', 'Bob'),
    ]


def test_split_sentences_carries_speaker_to_next_chunk_when_missing():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([['Lead-in chunk'], ['Next sentence finishes.']])

    supervisions = [
        make_supervision(0, 'Lead-in', speaker='Alice'),
        make_supervision(1, 'x' * 2000, speaker=None),
        make_supervision(2, 'Next sentence finishes.', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [('Lead-in chunk Next sentence finishes.', 'Alice')]


def test_split_sentences_respects_strip_whitespace_flag_and_length_split():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([['a' * 2000], ['Final sentence.']])

    supervisions = [
        make_supervision(0, 'a' * 2000, speaker=None),
        make_supervision(1, 'Final sentence.', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions, strip_whitespace=False)

    assert tokenizer.sentence_splitter.calls[0]['strip_whitespace'] is False
    assert tokenizer.sentence_splitter.calls[0]['texts'] == ['a' * 2000, 'Final sentence.']
    assert texts_and_speakers(result) == [('a' * 2000 + ' Final sentence.', None)]


def test_split_sentences_inserts_remainder_before_new_speaker():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([['Chunk one start still going'], ['Bob begins now', 'Wraps up.']])

    supervisions = [
        make_supervision(0, 'Chunk one start', speaker='Alice'),
        make_supervision(1, 'still going', speaker=None),
        make_supervision(2, 'Bob begins now', speaker='Bob'),
        make_supervision(3, 'Wraps up.', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ('Chunk one start still going', 'Alice'),
        ('Bob begins now', 'Bob'),
        ('Wraps up.', None),
    ]


def test_split_sentences_propagates_speaker_across_length_split():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    long_chunk = 'Intro ' + 'a' * 1000 + ' ' + 'b' * 1000
    tokenizer.sentence_splitter = FakeSplitter([[long_chunk], ['Continuation picks up', 'Wrap-up here.']])

    supervisions = [
        make_supervision(0, 'Intro', speaker='Alice'),
        make_supervision(1, 'a' * 1000, speaker=None),
        make_supervision(2, 'b' * 1000, speaker=None),
        make_supervision(3, 'Continuation picks up', speaker=None),
        make_supervision(4, 'Wrap-up here.', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    expected_first = f'{long_chunk} Continuation picks up'
    assert texts_and_speakers(result) == [
        (expected_first, 'Alice'),
        ('Wrap-up here.', None),
    ]


def test_split_sentences_handles_resplit_and_remainder_with_next_speaker():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter(
        [['[APPLAUSE] >> HOST:', 'Welcome everyone', 'Let us begin'], ['Tonight we feature highlights.']]
    )

    supervisions = [
        make_supervision(0, '[APPLAUSE] >> HOST:', speaker='MC'),
        make_supervision(1, 'Welcome everyone', speaker=None),
        make_supervision(2, 'Let us begin', speaker=None),
        make_supervision(3, 'Tonight we feature highlights.', speaker='Narrator'),
        make_supervision(4, '', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ('[APPLAUSE]', 'MC'),
        ('>> HOST: Welcome everyone', None),
        ('Let us begin', None),
        ('Tonight we feature highlights.', 'Narrator'),
    ]


def test_split_sentences_retains_speaker_for_final_remainder():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([['Closing thought that trails']])

    supervisions = [
        make_supervision(0, 'Closing thought that trails', speaker='Alice'),
        make_supervision(1, '', speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [('Closing thought that trails', 'Alice')]


def test_split_sentences_text_integrity():
    from pathlib import Path

    subtitle_file = Path('~/Downloads/lattifai_youtube_google/eIUqw3_YcCI.en.vtt').expanduser()
    subtitle_file = Path('~/Downloads/lattifai_youtube_google/7nv1snJRCEI.en.vtt').expanduser()
    if not subtitle_file.exists():
        pytest.skip('Subtitle file not found, skipping integrity test.')
        return

    from lattifai.io import SubtitleIO
    from lattifai.utils import _resolve_model_path

    model_path = _resolve_model_path('Lattifai/Lattice-1-Alpha')
    tokenizer = LatticeTokenizer.from_pretrained(None, model_path, device='cpu')

    supervisions = SubtitleIO.read(subtitle_file)
    tokenizer.init_sentence_splitter()
    splits = tokenizer.split_sentences(supervisions)

    origin_text = ''.join([(sup.speaker or '').strip() + sup.text for sup in supervisions]).replace(' ', '')
    split_text = ''.join([(sup.speaker or '').strip() + sup.text for sup in splits]).replace(' ', '')

    if origin_text != split_text:
        open(str(subtitle_file) + '.debug.supervisions.txt', 'w', encoding='utf-8').write(
            '\n'.join([f'[{sup.speaker}] {sup.text}' for sup in supervisions])
        )
        open(str(subtitle_file) + '.debug.splits.txt', 'w', encoding='utf-8').write(
            '\n'.join([f'[{sup.speaker}] {sup.text}' for sup in splits])
        )

        open(str(subtitle_file) + '.debug.supervisions_text', 'w', encoding='utf-8').write(
            origin_text
        )
        open(str(subtitle_file) + '.debug.splits_text', 'w', encoding='utf-8').write(
            split_text
        )

    assert origin_text == split_text, 'Text integrity check failed after sentence splitting.'
