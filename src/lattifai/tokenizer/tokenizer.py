import gzip
import inspect
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import torch

from lattifai.errors import LATTICE_DECODING_FAILURE_HELP, LatticeDecodingError
from lattifai.io import Supervision, normalize_html_text
from lattifai.tokenizer.phonemizer import G2Phonemizer

PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
END_PUNCTUATION = '.!?"]。！？”】'
PUNCTUATION_SPACE = PUNCTUATION + " "
STAR_TOKEN = "※"

GROUPING_SEPARATOR = "✹"

MAXIMUM_WORD_LENGTH = 40


TokenizerT = TypeVar("TokenizerT", bound="LatticeTokenizer")


class LatticeTokenizer:
    """Tokenizer for converting Lhotse Cut to LatticeGraph."""

    def __init__(self, client_wrapper: Any):
        self.client_wrapper = client_wrapper
        self.words: List[str] = []
        self.g2p_model: Any = None  # Placeholder for G2P model
        self.dictionaries = defaultdict(lambda: [])
        self.oov_word = "<unk>"
        self.sentence_splitter = None
        self.device = "cpu"

    def init_sentence_splitter(self):
        if self.sentence_splitter is not None:
            return

        import onnxruntime as ort
        from wtpsplit import SaT

        providers = []
        device = self.device
        if device.startswith("cuda") and ort.get_all_providers().count("CUDAExecutionProvider") > 0:
            providers.append("CUDAExecutionProvider")
        elif device.startswith("mps") and ort.get_all_providers().count("MPSExecutionProvider") > 0:
            providers.append("MPSExecutionProvider")

        sat = SaT(
            "sat-3l-sm",
            ort_providers=providers + ["CPUExecutionProvider"],
        )
        self.sentence_splitter = sat

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

    @classmethod
    def from_pretrained(
        cls: Type[TokenizerT],
        client_wrapper: Any,
        model_path: str,
        device: str = "cpu",
        compressed: bool = True,
    ) -> TokenizerT:
        """Load tokenizer from exported binary file"""
        from pathlib import Path

        words_model_path = f"{model_path}/words.bin"
        if compressed:
            with gzip.open(words_model_path, "rb") as f:
                data = pickle.load(f)
        else:
            with open(words_model_path, "rb") as f:
                data = pickle.load(f)

        tokenizer = cls(client_wrapper=client_wrapper)
        tokenizer.words = data["words"]
        tokenizer.dictionaries = defaultdict(list, data["dictionaries"])
        tokenizer.oov_word = data["oov_word"]

        g2p_model_path = f"{model_path}/g2p.bin" if Path(f"{model_path}/g2p.bin").exists() else None
        if g2p_model_path:
            tokenizer.g2p_model = G2Phonemizer(g2p_model_path, device=device)

        tokenizer.device = device
        tokenizer.add_special_tokens()
        return tokenizer

    def add_special_tokens(self):
        tokenizer = self
        for special_token in ["&gt;&gt;", "&gt;"]:
            if special_token not in tokenizer.dictionaries:
                tokenizer.dictionaries[special_token] = tokenizer.dictionaries[tokenizer.oov_word]
        return self

    def prenormalize(self, texts: List[str], language: Optional[str] = None) -> List[str]:
        if not self.g2p_model:
            raise ValueError("G2P model is not loaded, cannot prenormalize texts")

        oov_words = []
        for text in texts:
            text = normalize_html_text(text)
            words = text.lower().replace("-", " ").replace("—", " ").replace("–", " ").split()
            oovs = [w.strip(PUNCTUATION) for w in words if w not in self.words]
            if oovs:
                oov_words.extend([w for w in oovs if (w not in self.words and len(w) <= MAXIMUM_WORD_LENGTH)])

        oov_words = list(set(oov_words))
        if oov_words:
            indexs = []
            for k, _word in enumerate(oov_words):
                if any(_word.startswith(p) and _word.endswith(q) for (p, q) in [("(", ")"), ("[", "]")]):
                    self.dictionaries[_word] = self.dictionaries[self.oov_word]
                else:
                    _word = _word.strip(PUNCTUATION_SPACE)
                    if not _word or _word in self.words:
                        indexs.append(k)
            for idx in sorted(indexs, reverse=True):
                del oov_words[idx]

            g2p_words = [w for w in oov_words if w not in self.dictionaries]
            if g2p_words:
                predictions = self.g2p_model(words=g2p_words, lang=language, batch_size=len(g2p_words), num_prons=4)
                for _word, _predictions in zip(g2p_words, predictions):
                    for pronuncation in _predictions:
                        if pronuncation and pronuncation not in self.dictionaries[_word]:
                            self.dictionaries[_word].append(pronuncation)
                    if not self.dictionaries[_word]:
                        self.dictionaries[_word] = self.dictionaries[self.oov_word]

            pronunciation_dictionaries: Dict[str, List[List[str]]] = {
                w: self.dictionaries[w] for w in oov_words if self.dictionaries[w]
            }
            return pronunciation_dictionaries

        return {}

    def split_sentences(self, supervisions: List[Supervision], strip_whitespace=True) -> List[str]:
        """Split supervisions into sentences using the sentence splitter.

        Carefull about speaker changes.
        """
        texts, text_len, sidx = [], 0, 0
        speakers = []
        for s, supervision in enumerate(supervisions):
            text_len += len(supervision.text)
            if supervision.speaker:
                if sidx < s:
                    if len(speakers) < len(texts) + 1:
                        speakers.append(None)
                    text = " ".join([sup.text for sup in supervisions[sidx:s]])
                    texts.append(text)
                    sidx = s
                    text_len = len(supervision.text)
                speakers.append(supervision.speaker)

            else:
                if text_len >= 2000 or s == len(supervisions) - 1:
                    if len(speakers) < len(texts) + 1:
                        speakers.append(None)
                    text = " ".join([sup.text for sup in supervisions[sidx : s + 1]])
                    texts.append(text)
                    sidx = s + 1
                    text_len = 0

        assert len(speakers) == len(texts), f"len(speakers)={len(speakers)} != len(texts)={len(texts)}"
        sentences = self.sentence_splitter.split(texts, threshold=0.15, strip_whitespace=strip_whitespace)

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

    def tokenize(self, supervisions: List[Supervision], split_sentence: bool = False) -> Tuple[str, Dict[str, Any]]:
        if split_sentence:
            self.init_sentence_splitter()
            supervisions = self.split_sentences(supervisions)

        pronunciation_dictionaries = self.prenormalize([s.text for s in supervisions])
        response = self.client_wrapper.post(
            "tokenize",
            json={
                "supervisions": [s.to_dict() for s in supervisions],
                "pronunciation_dictionaries": pronunciation_dictionaries,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Failed to tokenize texts: {response.text}")
        result = response.json()
        lattice_id = result["id"]
        return (
            supervisions,
            lattice_id,
            (result["lattice_graph"], result["final_state"], result.get("acoustic_scale", 1.0)),
        )

    def detokenize(
        self,
        lattice_id: str,
        lattice_results: Tuple[torch.Tensor, Any, Any, float, float],
        supervisions: List[Supervision],
        return_details: bool = False,
    ) -> List[Supervision]:
        emission, results, labels, frame_shift, offset, channel = lattice_results  # noqa: F841
        response = self.client_wrapper.post(
            "detokenize",
            json={
                "lattice_id": lattice_id,
                "frame_shift": frame_shift,
                "results": [t.to_dict() for t in results[0]],
                "labels": labels[0],
                "offset": offset,
                "channel": channel,
                "return_details": return_details,
                "destroy_lattice": True,
            },
        )
        if response.status_code == 422:
            raise LatticeDecodingError(
                lattice_id,
                original_error=Exception(LATTICE_DECODING_FAILURE_HELP),
            )
        if response.status_code != 200:
            raise Exception(f"Failed to detokenize lattice: {response.text}")

        result = response.json()
        if not result.get("success"):
            raise Exception("Failed to detokenize the alignment results.")

        alignments = [Supervision.from_dict(s) for s in result["supervisions"]]

        if return_details:
            # Add emission confidence scores for segments and word-level alignments
            _add_confidence_scores(alignments, emission, labels[0], frame_shift)

        alignments = _update_alignments_speaker(supervisions, alignments)

        return alignments


class AsyncLatticeTokenizer(LatticeTokenizer):
    async def _post_async(self, endpoint: str, **kwargs):
        response = self.client_wrapper.post(endpoint, **kwargs)
        if inspect.isawaitable(response):
            return await response
        return response

    async def tokenize(
        self, supervisions: List[Supervision], split_sentence: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        if split_sentence:
            self.init_sentence_splitter()
            supervisions = self.split_sentences(supervisions)

        pronunciation_dictionaries = self.prenormalize([s.text for s in supervisions])
        response = await self._post_async(
            "tokenize",
            json={
                "supervisions": [s.to_dict() for s in supervisions],
                "pronunciation_dictionaries": pronunciation_dictionaries,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Failed to tokenize texts: {response.text}")
        result = response.json()
        lattice_id = result["id"]
        return (
            supervisions,
            lattice_id,
            (result["lattice_graph"], result["final_state"], result.get("acoustic_scale", 1.0)),
        )

    async def detokenize(
        self,
        lattice_id: str,
        lattice_results: Tuple[torch.Tensor, Any, Any, float, float],
        supervisions: List[Supervision],
        return_details: bool = False,
    ) -> List[Supervision]:
        emission, results, labels, frame_shift, offset, channel = lattice_results  # noqa: F841
        response = await self._post_async(
            "detokenize",
            json={
                "lattice_id": lattice_id,
                "frame_shift": frame_shift,
                "results": [t.to_dict() for t in results[0]],
                "labels": labels[0],
                "offset": offset,
                "channel": channel,
                "return_details": return_details,
                "destroy_lattice": True,
            },
        )
        if response.status_code == 422:
            raise LatticeDecodingError(
                lattice_id,
                original_error=Exception(LATTICE_DECODING_FAILURE_HELP),
            )
        if response.status_code != 200:
            raise Exception(f"Failed to detokenize lattice: {response.text}")

        result = response.json()
        if not result.get("success"):
            return Exception("Failed to detokenize the alignment results.")

        alignments = [Supervision.from_dict(s) for s in result["supervisions"]]

        if return_details:
            # Add emission confidence scores for segments and word-level alignments
            _add_confidence_scores(alignments, emission, labels[0], frame_shift)

        alignments = _update_alignments_speaker(supervisions, alignments)

        return alignments


def _add_confidence_scores(
    supervisions: List[Supervision],
    emission: torch.Tensor,
    labels: List[int],
    frame_shift: float,
) -> None:
    """
    Add confidence scores to supervisions and their word-level alignments.

    This function modifies supervisions in-place by:
    1. Computing segment-level confidence scores based on emission probabilities
    2. Computing word-level confidence scores for each aligned word

    Args:
        supervisions: List of Supervision objects to add scores to (modified in-place)
        emission: Emission tensor with shape [batch, time, vocab_size]
        labels: Token labels corresponding to aligned tokens
        frame_shift: Frame shift in seconds for converting frames to time
    """
    tokens = torch.tensor(labels, dtype=torch.int64, device=emission.device)

    for supervision in supervisions:
        start_frame = int(supervision.start / frame_shift)
        end_frame = int(supervision.end / frame_shift)

        # Compute segment-level confidence
        probabilities = emission[0, start_frame:end_frame].softmax(dim=-1)
        aligned = probabilities[range(0, end_frame - start_frame), tokens[start_frame:end_frame]]
        diffprobs = (probabilities.max(dim=-1).values - aligned).cpu()
        supervision.score = round(1.0 - diffprobs.mean().item(), ndigits=4)

        # Compute word-level confidence if alignment exists
        if hasattr(supervision, "alignment") and supervision.alignment:
            words = supervision.alignment.get("word", [])
            for w, item in enumerate(words):
                start = int(item.start / frame_shift) - start_frame
                end = int(item.end / frame_shift) - start_frame
                words[w] = item._replace(score=round(1.0 - diffprobs[start:end].mean().item(), ndigits=4))


def _update_alignments_speaker(supervisions: List[Supervision], alignments: List[Supervision]) -> List[Supervision]:
    """
    Update the speaker attribute for a list of supervisions.

    Args:
        supervisions: List of Supervision objects to get speaker info from
        alignments: List of aligned Supervision objects to update speaker info to
    """
    for supervision, alignment in zip(supervisions, alignments):
        alignment.speaker = supervision.speaker
    return alignments
