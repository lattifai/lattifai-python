import copy
import sys
import types
from typing import List, Optional, Tuple

from lattifai import Caption
from lattifai.alignment.tokenizer import LatticeTokenizer
from lattifai.caption import Supervision


class FakeSplitter:
    def __init__(self, outputs: List[List[str]]):
        self._outputs = outputs
        self.calls = []
        self._splitter = self  # Mock the internal splitter

    def split(self, texts: List[str], threshold: float, strip_whitespace: bool, batch_size: int = 8) -> List[List[str]]:
        self.calls.append(
            {
                "texts": list(texts),
                "threshold": threshold,
                "strip_whitespace": strip_whitespace,
                "batch_size": batch_size,
            }
        )
        return copy.deepcopy(self._outputs)

    def split_sentences(self, supervisions: List[Supervision], strip_whitespace: bool = True) -> List[Supervision]:
        """Mock split_sentences method that delegates to the real implementation."""
        from lattifai.alignment.sentence_splitter import SentenceSplitter

        # Call the actual split_sentences logic from SentenceSplitter
        # but use our mock split() method
        texts, speakers = [], []
        text_len, sidx = 0, 0

        def flush_segment(end_idx: int, speaker: Optional[str] = None):
            """Flush accumulated text from sidx to end_idx with given speaker."""
            nonlocal text_len, sidx
            if sidx <= end_idx:
                if len(speakers) < len(texts) + 1:
                    speakers.append(speaker)
                text = " ".join(sup.text for sup in supervisions[sidx : end_idx + 1])
                texts.append(text)
                sidx = end_idx + 1
                text_len = 0

        for s, supervision in enumerate(supervisions):
            text_len += len(supervision.text)
            is_last = s == len(supervisions) - 1

            if supervision.speaker:
                # Flush previous segment without speaker (if any)
                if sidx < s:
                    flush_segment(s - 1, None)
                    text_len = len(supervision.text)

                # Check if we should flush this speaker's segment now
                next_has_speaker = not is_last and supervisions[s + 1].speaker
                if is_last or next_has_speaker:
                    flush_segment(s, supervision.speaker)
                else:
                    speakers.append(supervision.speaker)

            elif text_len >= 2000 or is_last:
                flush_segment(s, None)

        assert len(speakers) == len(texts), f"len(speakers)={len(speakers)} != len(texts)={len(texts)}"
        sentences = self.split(texts, threshold=0.15, strip_whitespace=strip_whitespace, batch_size=8)

        supervisions, remainder = [], ""
        for k, (_speaker, _sentences) in enumerate(zip(speakers, sentences)):
            # Prepend remainder from previous iteration to the first sentence
            if _sentences and remainder:
                _sentences[0] = remainder + _sentences[0]
                remainder = ""

            if not _sentences:
                continue

            # Process and re-split special sentence types
            processed_sentences = []
            for s, _sentence in enumerate(_sentences):
                if remainder:
                    _sentence = remainder + _sentence
                    remainder = ""
                # Detect and split special sentence types
                resplit_parts = SentenceSplitter._resplit_special_sentence_types(_sentence)
                if any(resplit_parts[-1].endswith(sp) for sp in [":", "："]):
                    if s < len(_sentences) - 1:
                        _sentences[s + 1] = resplit_parts[-1] + " " + _sentences[s + 1]
                    else:  # last part
                        remainder = resplit_parts[-1] + " "
                    processed_sentences.extend(resplit_parts[:-1])
                else:
                    processed_sentences.extend(resplit_parts)
            _sentences = processed_sentences

            if not _sentences:
                if remainder:
                    _sentences, remainder = [remainder.strip()], ""
                else:
                    continue

            END_PUNCTUATION = '.!?"]。！？"】'
            if any(_sentences[-1].endswith(ep) for ep in END_PUNCTUATION):
                supervisions.extend(
                    Supervision(text=text, speaker=(_speaker if s == 0 else None)) for s, text in enumerate(_sentences)
                )
                _speaker = None  # reset speaker after use
            else:
                supervisions.extend(
                    Supervision(text=text, speaker=(_speaker if s == 0 else None))
                    for s, text in enumerate(_sentences[:-1])
                )
                remainder = _sentences[-1] + " " + remainder
                if k < len(speakers) - 1 and speakers[k + 1] is not None:  # next speaker is set
                    supervisions.append(
                        Supervision(text=remainder.strip(), speaker=_speaker if len(_sentences) == 1 else None)
                    )
                    remainder = ""
                elif len(_sentences) == 1:
                    if k == len(speakers) - 1:
                        pass  # keep _speaker for the last supervision
                    else:
                        assert speakers[k + 1] is None
                        speakers[k + 1] = _speaker
                else:
                    assert len(_sentences) > 1
                    _speaker = None  # reset speaker if sentence not ended

        if remainder.strip():
            supervisions.append(Supervision(text=remainder.strip(), speaker=_speaker))

        return supervisions


def make_supervision(idx: int, text: str, speaker: Optional[str]) -> Supervision:
    return Supervision(
        id=f"sup-{idx}",
        recording_id="rec",
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
    tokenizer.sentence_splitter = FakeSplitter([["Hello world.", "This is second sentence!"]])

    supervisions = [
        make_supervision(0, "Hello world.", speaker="Alice"),
        make_supervision(1, "This is second sentence!", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ("Hello world.", "Alice"),
        ("This is second sentence!", None),
    ]


def test_split_sentences_emits_trailing_remainder_without_punctuation():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([["Trailing remainder"]])

    supervisions = [make_supervision(0, "Trailing remainder", speaker=None)]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [("Trailing remainder", None)]


def test_split_sentences_resplits_special_colon_sequences():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([["[APPLAUSE] &gt;&gt; SPEAKER:", "We are live."]])

    supervisions = [
        make_supervision(0, "[APPLAUSE] &gt;&gt; SPEAKER:", speaker="Host"),
        make_supervision(1, "We are live.", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ("[APPLAUSE]", "Host"),
        ("&gt;&gt; SPEAKER: We are live.", None),
    ]


def test_split_sentences_outputs_remainder_before_next_speaker():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([["Incomplete thought"], ["Replies with closure."]])

    supervisions = [
        make_supervision(0, "Incomplete thought", speaker="Alice"),
        make_supervision(1, "", speaker=None),
        make_supervision(2, "Replies with closure.", speaker="Bob"),
        make_supervision(3, "", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ("Incomplete thought", "Alice"),
        ("Replies with closure.", "Bob"),
    ]


def test_split_sentences_carries_speaker_to_next_chunk_when_missing():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([["Lead-in chunk"], ["Next sentence finishes."]])

    supervisions = [
        make_supervision(0, "Lead-in", speaker="Alice"),
        make_supervision(1, "x" * 2000, speaker=None),
        make_supervision(2, "Next sentence finishes.", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [("Lead-in chunk Next sentence finishes.", "Alice")]


def test_split_sentences_respects_strip_whitespace_flag_and_length_split():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([["a" * 2000], ["Final sentence."]])

    supervisions = [
        make_supervision(0, "a" * 2000, speaker=None),
        make_supervision(1, "Final sentence.", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions, strip_whitespace=False)

    assert tokenizer.sentence_splitter.calls[0]["strip_whitespace"] is False
    assert tokenizer.sentence_splitter.calls[0]["texts"] == ["a" * 2000, "Final sentence."]
    assert texts_and_speakers(result) == [("a" * 2000 + " Final sentence.", None)]


def test_split_sentences_inserts_remainder_before_new_speaker():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([["Chunk one start still going"], ["Bob begins now", "Wraps up."]])

    supervisions = [
        make_supervision(0, "Chunk one start", speaker="Alice"),
        make_supervision(1, "still going", speaker=None),
        make_supervision(2, "Bob begins now", speaker="Bob"),
        make_supervision(3, "Wraps up.", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ("Chunk one start still going", "Alice"),
        ("Bob begins now", "Bob"),
        ("Wraps up.", None),
    ]


def test_split_sentences_propagates_speaker_across_length_split():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    long_chunk = "Intro " + "a" * 1000 + " " + "b" * 1000
    tokenizer.sentence_splitter = FakeSplitter([[long_chunk], ["Continuation picks up", "Wrap-up here."]])

    supervisions = [
        make_supervision(0, "Intro", speaker="Alice"),
        make_supervision(1, "a" * 1000, speaker=None),
        make_supervision(2, "b" * 1000, speaker=None),
        make_supervision(3, "Continuation picks up", speaker=None),
        make_supervision(4, "Wrap-up here.", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    expected_first = f"{long_chunk} Continuation picks up"
    assert texts_and_speakers(result) == [
        (expected_first, "Alice"),
        ("Wrap-up here.", None),
    ]


def test_split_sentences_handles_resplit_and_remainder_with_next_speaker():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter(
        [["[APPLAUSE] >> HOST:", "Welcome everyone", "Let us begin"], ["Tonight we feature highlights."]]
    )

    supervisions = [
        make_supervision(0, "[APPLAUSE] >> HOST:", speaker="MC"),
        make_supervision(1, "Welcome everyone", speaker=None),
        make_supervision(2, "Let us begin", speaker=None),
        make_supervision(3, "Tonight we feature highlights.", speaker="Narrator"),
        make_supervision(4, "", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [
        ("[APPLAUSE]", "MC"),
        (">> HOST: Welcome everyone", None),
        ("Let us begin", None),
        ("Tonight we feature highlights.", "Narrator"),
    ]


def test_split_sentences_retains_speaker_for_final_remainder():
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.sentence_splitter = FakeSplitter([["Closing thought that trails"]])

    supervisions = [
        make_supervision(0, "Closing thought that trails", speaker="Alice"),
        make_supervision(1, "", speaker=None),
    ]

    result = tokenizer.split_sentences(supervisions)

    assert texts_and_speakers(result) == [("Closing thought that trails", "Alice")]


def test_split_sentences_text_integrity():
    import tempfile
    import zipfile
    from pathlib import Path

    tokenizer = LatticeTokenizer(client_wrapper=None)

    for caption_file in [
        "tests/data/captions/7nv1snJRCEI.en.vtt.zip",
        "tests/data/captions/eIUqw3_YcCI.en.vtt.zip",
        "tests/data/captions/_xYSQe9oq6c.en.vtt.zip",
    ]:
        # Unzip the caption file
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(caption_file, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find the extracted .vtt file
            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            if not vtt_files:
                raise FileNotFoundError(f"No .vtt file found in {caption_file}")

            extracted_file = str(vtt_files[0])
            caption = Caption.read(extracted_file)
            supervisions = caption.supervisions

            tokenizer.init_sentence_splitter()
            splits = tokenizer.split_sentences(supervisions)

            origin_text = "".join([(sup.speaker or "").strip() + sup.text for sup in supervisions]).replace(" ", "")
            split_text = "".join([(sup.speaker or "").strip() + sup.text for sup in splits]).replace(" ", "")

            if origin_text != split_text:
                open(str(caption_file) + ".debug.supervisions.txt", "w", encoding="utf-8").write(
                    "\n".join([f"[{sup.speaker}] {sup.text}" for sup in supervisions])
                )
                open(str(caption_file) + ".debug.splits.txt", "w", encoding="utf-8").write(
                    "\n".join([f"[{sup.speaker}] {sup.text}" for sup in splits])
                )

                open(str(caption_file) + ".debug.supervisions_text", "w", encoding="utf-8").write(origin_text)
                open(str(caption_file) + ".debug.splits_text", "w", encoding="utf-8").write(split_text)

            assert origin_text == split_text, "Text integrity check failed after sentence splitting."
