import re
from typing import List, Optional

from lattifai.alignment.punctuation import END_PUNCTUATION
from lattifai.caption import Supervision
from lattifai.utils import _resolve_model_path


class SentenceSplitter:
    """Lazy-initialized sentence splitter using wtpsplit."""

    def __init__(self, device: str = "cpu", model_hub: Optional[str] = "modelscope", lazy_init: bool = True):
        """Initialize sentence splitter with lazy loading.

        Args:
            device: Device to run the model on (cpu, cuda, mps)
            model_hub: Model hub to use (None for huggingface, "modelscope" for modelscope)
        """
        self.device = device
        self.model_hub = model_hub
        self._splitter = None
        if not lazy_init:
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
    def _distribute_time_info(
        input_supervisions: List[Supervision],
        split_texts: List[str],
    ) -> List[Supervision]:
        """Distribute time information from input supervisions to split sentences.

        Args:
            input_supervisions: Original supervisions with time information
            split_texts: List of split sentence texts

        Returns:
            List of Supervision objects with distributed time information.
            Custom attributes are inherited from first_sup with conflict markers.
        """
        if not input_supervisions:
            return [Supervision(text=text, id="", recording_id="", start=0, duration=0) for text in split_texts]

        # Build concatenated input text
        input_text = " ".join(sup.text for sup in input_supervisions)

        # Pre-compute supervision position mapping for O(1) lookup
        # Format: [(start_pos, end_pos, supervision), ...]
        sup_ranges = []
        char_pos = 0
        for sup in input_supervisions:
            sup_start = char_pos
            sup_end = char_pos + len(sup.text)
            sup_ranges.append((sup_start, sup_end, sup))
            char_pos = sup_end + 1  # +1 for space separator

        # Process each split text
        result = []
        search_start = 0
        sup_idx = 0  # Track current supervision index to skip processed ones

        for split_text in split_texts:
            text_start = input_text.find(split_text, search_start)
            if text_start == -1:
                raise ValueError(f"Could not find split text '{split_text}' in input supervisions.")

            text_end = text_start + len(split_text)
            search_start = text_end

            # Find overlapping supervisions, starting from last used index
            first_sup = None
            last_sup = None
            first_char_idx = None
            last_char_idx = None
            overlapping_customs = []  # Track all custom dicts for conflict detection

            # Start from sup_idx, which is the first supervision that might overlap
            for i in range(sup_idx, len(sup_ranges)):
                sup_start, sup_end, sup = sup_ranges[i]

                # Skip if no overlap (before text_start)
                if sup_end <= text_start:
                    sup_idx = i + 1  # Update starting point for next iteration
                    continue

                # Stop if no overlap (after text_end)
                if sup_start >= text_end:
                    break

                # Found overlap
                if first_sup is None:
                    first_sup = sup
                    first_char_idx = max(0, text_start - sup_start)

                last_sup = sup
                last_char_idx = min(len(sup.text) - 1, text_end - 1 - sup_start)

                # Collect custom dict for conflict detection
                if getattr(sup, "custom", None):
                    overlapping_customs.append(sup.custom)

            if first_sup is None or last_sup is None:
                raise ValueError(f"Could not find supervisions for split text: {split_text}")

            # Calculate timing
            start_time = first_sup.start + (first_char_idx / len(first_sup.text)) * first_sup.duration
            end_time = last_sup.start + ((last_char_idx + 1) / len(last_sup.text)) * last_sup.duration

            # Inherit custom from first_sup, mark conflicts if multiple sources
            merged_custom = None
            if overlapping_customs:
                # Start with first_sup's custom (inherit strategy)
                merged_custom = overlapping_customs[0].copy() if overlapping_customs[0] else {}

                # Detect conflicts if multiple overlapping supervisions have different custom values
                if len(overlapping_customs) > 1:
                    has_conflict = False
                    for other_custom in overlapping_customs[1:]:
                        if other_custom and other_custom != overlapping_customs[0]:
                            has_conflict = True
                            break

                    if has_conflict:
                        # Mark that this supervision spans multiple sources with different customs
                        merged_custom["_split_from_multiple"] = True
                        merged_custom["_source_count"] = len(overlapping_customs)

            result.append(
                Supervision(
                    id="",
                    text=split_text,
                    start=start_time,
                    duration=end_time - start_time,
                    recording_id=first_sup.recording_id,
                    custom=merged_custom,
                )
            )

        return result

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

        if len(speakers) != len(texts):
            raise ValueError(f"len(speakers)={len(speakers)} != len(texts)={len(texts)}")
        sentences = self._splitter.split(texts, threshold=0.15, strip_whitespace=strip_whitespace, batch_size=8)

        # First pass: collect all split texts with their speakers
        split_texts_with_speakers = []
        remainder = ""
        remainder_speaker = None

        for k, (_speaker, _sentences) in enumerate(zip(speakers, sentences)):
            # Prepend remainder from previous iteration to the first sentence
            if _sentences and remainder:
                _sentences[0] = remainder + _sentences[0]
                _speaker = remainder_speaker if remainder_speaker else _speaker
                remainder = ""
                remainder_speaker = None

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
                split_texts_with_speakers.extend(
                    (text, _speaker if s == 0 else None) for s, text in enumerate(_sentences)
                )
                _speaker = None  # reset speaker after use
            else:
                split_texts_with_speakers.extend(
                    (text, _speaker if s == 0 else None) for s, text in enumerate(_sentences[:-1])
                )
                remainder = _sentences[-1] + " " + remainder
                if k < len(speakers) - 1 and speakers[k + 1] is not None:  # next speaker is set
                    split_texts_with_speakers.append((remainder.strip(), _speaker if len(_sentences) == 1 else None))
                    remainder = ""
                    remainder_speaker = None
                elif len(_sentences) == 1:
                    remainder_speaker = _speaker
                    if k == len(speakers) - 1:
                        pass  # keep _speaker for the last supervision
                    elif speakers[k + 1] is not None:
                        raise ValueError(f"Expected speakers[{k + 1}] to be None, got {speakers[k + 1]}")
                    else:
                        speakers[k + 1] = _speaker
                elif len(_sentences) > 1:
                    _speaker = None  # reset speaker if sentence not ended
                    remainder_speaker = None
                else:
                    raise ValueError(f"Unexpected state: len(_sentences)={len(_sentences)}")

        if remainder.strip():
            split_texts_with_speakers.append((remainder.strip(), remainder_speaker))

        # Second pass: distribute time information
        split_texts = [text for text, _ in split_texts_with_speakers]
        result_supervisions = self._distribute_time_info(supervisions, split_texts)

        # Third pass: add speaker information
        for sup, (_, speaker) in zip(result_supervisions, split_texts_with_speakers):
            if speaker:
                sup.speaker = speaker

        return result_supervisions
