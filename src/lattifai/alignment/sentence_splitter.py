import re
from typing import List, Optional

from lattifai.caption import Supervision
from lattifai.utils import _resolve_model_path

END_PUNCTUATION = '.!?"]。！？"】'


class SentenceSplitter:
    """Lazy-initialized sentence splitter using wtpsplit."""

    def __init__(self, device: str = "cpu", model_hub: Optional[str] = None, lazy_init: bool = True):
        """Initialize sentence splitter with lazy loading.

        Args:
            device: Device to run the model on (cpu, cuda, mps)
            model_hub: Model hub to use (None for huggingface, "modelscope" for modelscope)
        """
        self.device = device
        self.model_hub = model_hub
        if lazy_init:
            self._splitter = None
        else:
            self._init_splitter()

    def _init_splitter(self):
        """Initialize the sentence splitter model on first use."""
        if self._splitter is not None:
            return

        import onnxruntime as ort
        from wtpsplit import SaT

        providers = []
        device = self.device
        if device.startswith("cuda") and ort.get_all_providers().count("CUDAExecutionProvider") > 0:
            providers.append("CUDAExecutionProvider")
        elif device.startswith("mps") and ort.get_all_providers().count("MPSExecutionProvider") > 0:
            providers.append("MPSExecutionProvider")

        if self.model_hub == "modelscope":
            downloaded_path = _resolve_model_path("LattifAI/OmniTokenizer", model_hub="modelscope")
            sat = SaT(
                f"{downloaded_path}/sat-3l-sm",
                tokenizer_name_or_path=f"{downloaded_path}/xlm-roberta-base",
                ort_providers=providers + ["CPUExecutionProvider"],
            )
        else:
            sat_path = _resolve_model_path("segment-any-text/sat-3l-sm", model_hub="huggingface")
            sat = SaT(
                sat_path,
                tokenizer_name_or_path="facebookAI/xlm-roberta-base",
                hub_prefix="segment-any-text",
                ort_providers=providers + ["CPUExecutionProvider"],
            )
        self._splitter = sat

    @staticmethod
    def _resplit_special_sentence_types(sentence: str) -> List[str]:
        """
        Re-split special sentence types.

        Examples:
        '[APPLAUSE] &gt;&gt; MIRA MURATI:' -> ['[APPLAUSE]', '&gt;&gt; MIRA MURATI:']
        '[MUSIC] &gt;&gt; SPEAKER:' -> ['[MUSIC]', '&gt;&gt; SPEAKER:']

        Special handling patterns:
        1. Separate special marks at the beginning (e.g., [APPLAUSE], [MUSIC], etc.) from subsequent speaker marks
        2. Use speaker marks (&gt;&gt; or other separators) as split points

        Args:
            sentence: Input sentence string

        Returns:
            List of re-split sentences. If no special marks are found, returns the original sentence in a list
        """
        # Detect special mark patterns: [SOMETHING] &gt;&gt; SPEAKER:
        # or other forms like [SOMETHING] SPEAKER:

        # Pattern 1: [mark] HTML-encoded separator speaker:
        pattern1 = r"^(\[[^\]]+\])\s+(&gt;&gt;|>>)\s+(.+)$"
        match1 = re.match(pattern1, sentence.strip())
        if match1:
            special_mark = match1.group(1)
            separator = match1.group(2)
            speaker_part = match1.group(3)
            return [special_mark, f"{separator} {speaker_part}"]

        # Pattern 2: [mark] speaker:
        pattern2 = r"^(\[[^\]]+\])\s+([^:]+:)(.*)$"
        match2 = re.match(pattern2, sentence.strip())
        if match2:
            special_mark = match2.group(1)
            speaker_label = match2.group(2)
            remaining = match2.group(3).strip()
            if remaining:
                return [special_mark, f"{speaker_label} {remaining}"]
            else:
                return [special_mark, speaker_label]

        # If no special pattern matches, return the original sentence
        return [sentence]

    def split_sentences(self, supervisions: List[Supervision], strip_whitespace=True) -> List[Supervision]:
        """Split supervisions into sentences using the sentence splitter.

        Careful about speaker changes.

        Args:
            supervisions: List of Supervision objects to split
            strip_whitespace: Whether to strip whitespace from split sentences

        Returns:
            List of Supervision objects with split sentences
        """
        self._init_splitter()

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
        sentences = self._splitter.split(texts, threshold=0.15, strip_whitespace=strip_whitespace, batch_size=8)

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
                # Detect and split special sentence types: e.g., '[APPLAUSE] &gt;&gt; MIRA MURATI:' -> ['[APPLAUSE]', '&gt;&gt; MIRA MURATI:']  # noqa: E501
                resplit_parts = self._resplit_special_sentence_types(_sentence)
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
