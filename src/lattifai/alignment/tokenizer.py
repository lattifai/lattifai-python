import gzip
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np

from lattifai.caption import Supervision
from lattifai.caption import normalize_text as normalize_html_text
from lattifai.errors import (
    LATTICE_DECODING_FAILURE_HELP,
    LatticeDecodingError,
    ModelLoadError,
    QuotaExceededError,
)

from .phonemizer import G2Phonemizer
from .sentence_splitter import SentenceSplitter

PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
PUNCTUATION_SPACE = PUNCTUATION + " "
STAR_TOKEN = "※"

GROUPING_SEPARATOR = "✹"

MAXIMUM_WORD_LENGTH = 40


TokenizerT = TypeVar("TokenizerT", bound="LatticeTokenizer")


def _is_punctuation(char: str) -> bool:
    """Check if a character is punctuation (not space, not alphanumeric, not CJK)."""
    if len(char) != 1:
        return False
    if char.isspace():
        return False
    if char.isalnum():
        return False
    # Check if it's a CJK character
    if "\u4e00" <= char <= "\u9fff":
        return False
    # Check if it's an accented Latin character
    if "\u00c0" <= char <= "\u024f":
        return False
    return True


def tokenize_multilingual_text(text: str, keep_spaces: bool = True, attach_punctuation: bool = False) -> list[str]:
    """
    Tokenize a mixed Chinese-English string into individual units.

    Tokenization rules:
    - Chinese characters (CJK) are split individually
    - Consecutive Latin letters (including accented characters) and digits are grouped as one unit
    - English contractions ('s, 't, 'm, 'll, 're, 've) are kept with the preceding word
    - Other characters (punctuation, spaces) are split individually by default
    - If attach_punctuation=True, punctuation marks are attached to the preceding token

    Args:
        text: Input string containing mixed Chinese and English text
        keep_spaces: If True, spaces are included in the output as separate tokens.
                     If False, spaces are excluded from the output. Default is True.
        attach_punctuation: If True, punctuation marks are attached to the preceding token.
                            For example, "Hello, World!" becomes ["Hello,", " ", "World!"].
                            Default is False.

    Returns:
        List of tokenized units

    Examples:
        >>> tokenize_multilingual_text("Hello世界")
        ['Hello', '世', '界']
        >>> tokenize_multilingual_text("I'm fine")
        ["I'm", ' ', 'fine']
        >>> tokenize_multilingual_text("I'm fine", keep_spaces=False)
        ["I'm", 'fine']
        >>> tokenize_multilingual_text("Kühlschrank")
        ['Kühlschrank']
        >>> tokenize_multilingual_text("Hello, World!", attach_punctuation=True)
        ['Hello,', ' ', 'World!']
    """
    # Regex pattern:
    # - [a-zA-Z0-9\u00C0-\u024F]+ matches Latin letters (including accented chars like ü, ö, ä, ß, é, etc.)
    # - (?:'[a-zA-Z]{1,2})? optionally matches contractions like 's, 't, 'm, 'll, 're, 've
    # - [\u4e00-\u9fff] matches CJK characters
    # - . matches any other single character
    # Unicode ranges:
    # - \u00C0-\u00FF: Latin-1 Supplement (À-ÿ)
    # - \u0100-\u017F: Latin Extended-A
    # - \u0180-\u024F: Latin Extended-B
    pattern = re.compile(r"([a-zA-Z0-9\u00C0-\u024F]+(?:'[a-zA-Z]{1,2})?|[\u4e00-\u9fff]|.)")

    # filter(None, ...) removes any empty strings from re.findall results
    tokens = list(filter(None, pattern.findall(text)))

    if attach_punctuation and len(tokens) > 1:
        # Attach punctuation to the preceding token
        # Punctuation characters (excluding spaces) are merged with the previous token
        merged_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # Look ahead to collect consecutive punctuation (non-space, non-alphanumeric, non-CJK)
            if merged_tokens and _is_punctuation(token):
                merged_tokens[-1] = merged_tokens[-1] + token
            else:
                merged_tokens.append(token)
            i += 1
        tokens = merged_tokens

    if not keep_spaces:
        tokens = [t for t in tokens if not t.isspace()]

    return tokens


class LatticeTokenizer:
    """Tokenizer for converting Lhotse Cut to LatticeGraph."""

    def __init__(self, client_wrapper: Any):
        self.client_wrapper = client_wrapper
        self.model_name = ""
        self.model_hub: Optional[str] = None
        self.words: List[str] = []
        self.g2p_model: Any = None  # Placeholder for G2P model
        self.dictionaries = defaultdict(lambda: [])
        self.oov_word = "<unk>"
        self.sentence_splitter: Optional[SentenceSplitter] = None
        self.device = "cpu"

    def init_sentence_splitter(self):
        if self.sentence_splitter is None:
            self.sentence_splitter = SentenceSplitter(device=self.device, model_hub=self.model_hub)

    @classmethod
    def from_pretrained(
        cls: Type[TokenizerT],
        client_wrapper: Any,
        model_path: str,
        model_name: str,
        model_hub: Optional[str] = None,
        device: str = "cpu",
        compressed: bool = True,
    ) -> TokenizerT:
        """Load tokenizer from exported binary file"""
        from pathlib import Path

        words_model_path = f"{model_path}/words.bin"
        try:
            if compressed:
                with gzip.open(words_model_path, "rb") as f:
                    data = pickle.load(f)
            else:
                with open(words_model_path, "rb") as f:
                    data = pickle.load(f)
        except Exception as e:
            del e
            import msgpack

            if compressed:
                with gzip.open(words_model_path, "rb") as f:
                    data = msgpack.unpack(f, raw=False, strict_map_key=False)
            else:
                with open(words_model_path, "rb") as f:
                    data = msgpack.unpack(f, raw=False, strict_map_key=False)

        tokenizer = cls(client_wrapper=client_wrapper)
        tokenizer.model_name = model_name
        tokenizer.model_hub = model_hub
        tokenizer.words = data["words"]
        tokenizer.dictionaries = defaultdict(list, data["dictionaries"])
        tokenizer.oov_word = data["oov_word"]

        g2pp_model_path = f"{model_path}/g2pp.bin" if Path(f"{model_path}/g2pp.bin").exists() else None
        if g2pp_model_path:
            tokenizer.g2p_model = G2Phonemizer(g2pp_model_path, device=device)
        else:
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
            # support english, chinese and german tokenization
            words = tokenize_multilingual_text(
                text.lower().replace("-", " ").replace("—", " ").replace("–", " "), keep_spaces=False
            )
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

    def split_sentences(self, supervisions: List[Supervision], strip_whitespace=True) -> List[Supervision]:
        """Split supervisions into sentences using the sentence splitter.

        Careful about speaker changes.
        """
        self.init_sentence_splitter()
        return self.sentence_splitter.split_sentences(supervisions, strip_whitespace=strip_whitespace)

    def tokenize(self, supervisions: List[Supervision], split_sentence: bool = False) -> Tuple[str, Dict[str, Any]]:
        if split_sentence:
            supervisions = self.split_sentences(supervisions)

        pronunciation_dictionaries = self.prenormalize([s.text for s in supervisions])
        response = self.client_wrapper.post(
            "tokenize",
            json={
                "model_name": self.model_name,
                "supervisions": [s.to_dict() for s in supervisions],
                "pronunciation_dictionaries": pronunciation_dictionaries,
            },
        )
        if response.status_code == 402:
            raise QuotaExceededError(response.json().get("detail", "Quota exceeded"))
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
        lattice_results: Tuple[np.ndarray, Any, Any, float, float],
        supervisions: List[Supervision],
        return_details: bool = False,
        start_margin: float = 0.08,
        end_margin: float = 0.20,
    ) -> List[Supervision]:
        emission, results, labels, frame_shift, offset, channel = lattice_results  # noqa: F841
        response = self.client_wrapper.post(
            "detokenize",
            json={
                "model_name": self.model_name,
                "lattice_id": lattice_id,
                "frame_shift": frame_shift,
                "results": [t.to_dict() for t in results[0]],
                "labels": labels[0],
                "offset": offset,
                "channel": channel,
                "return_details": False if return_details is None else return_details,
                "destroy_lattice": True,
                "start_margin": start_margin,
                "end_margin": end_margin,
            },
        )
        if response.status_code == 400:
            raise LatticeDecodingError(
                lattice_id,
                original_error=Exception(LATTICE_DECODING_FAILURE_HELP),
            )
        if response.status_code == 402:
            raise QuotaExceededError(response.json().get("detail", "Quota exceeded"))
        if response.status_code != 200:
            raise Exception(f"Failed to detokenize lattice: {response.text}")

        result = response.json()
        if not result.get("success"):
            raise Exception("Failed to detokenize the alignment results.")

        alignments = [Supervision.from_dict(s) for s in result["supervisions"]]

        if emission is not None and return_details:
            # Add emission confidence scores for segments and word-level alignments
            _add_confidence_scores(alignments, emission, labels[0], frame_shift, offset)

        alignments = _update_alignments_speaker(supervisions, alignments)

        return alignments


def _add_confidence_scores(
    supervisions: List[Supervision],
    emission: np.ndarray,
    labels: List[int],
    frame_shift: float,
    offset: float = 0.0,
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
    tokens = np.array(labels, dtype=np.int64)

    for supervision in supervisions:
        start_frame = int((supervision.start - offset) / frame_shift)
        end_frame = int((supervision.end - offset) / frame_shift)

        # Compute segment-level confidence
        probabilities = np.exp(emission[0, start_frame:end_frame])
        aligned = probabilities[range(0, end_frame - start_frame), tokens[start_frame:end_frame]]
        diffprobs = np.max(probabilities, axis=-1) - aligned
        supervision.score = round(1.0 - diffprobs.mean(), ndigits=4)

        # Compute word-level confidence if alignment exists
        if hasattr(supervision, "alignment") and supervision.alignment:
            words = supervision.alignment.get("word", [])
            for w, item in enumerate(words):
                start = int((item.start - offset) / frame_shift) - start_frame
                end = int((item.end - offset) / frame_shift) - start_frame
                words[w] = item._replace(score=round(1.0 - diffprobs[start:end].mean(), ndigits=4))


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


def _load_tokenizer(
    client_wrapper: Any,
    model_path: str,
    model_name: str,
    device: str,
    *,
    model_hub: Optional[str] = None,
    tokenizer_cls: Type[LatticeTokenizer] = LatticeTokenizer,
) -> LatticeTokenizer:
    """Instantiate tokenizer with consistent error handling."""
    return tokenizer_cls.from_pretrained(
        client_wrapper=client_wrapper,
        model_path=model_path,
        model_name=model_name,
        model_hub=model_hub,
        device=device,
    )
