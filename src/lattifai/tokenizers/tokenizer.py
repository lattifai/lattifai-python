import gzip
import pickle
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import torch

from lattifai.base_client import SyncAPIClient
from lattifai.io import Supervision
from lattifai.tokenizers.phonemizer import G2Phonemizer

PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
PUNCTUATION_SPACE = PUNCTUATION + ' '
STAR_TOKEN = '※'

GROUPING_SEPARATOR = '✹'

MAXIMUM_WORD_LENGTH = 40


class LatticeTokenizer:
    """Tokenizer for converting Lhotse Cut to LatticeGraph."""

    def __init__(self, client_wrapper: SyncAPIClient):
        self.client_wrapper = client_wrapper
        self.words: List[str] = []
        self.g2p_model: Any = None  # Placeholder for G2P model
        self.dictionaries = defaultdict(lambda: [])
        self.oov_word = '<unk>'

    @staticmethod
    def from_pretrained(
        client_wrapper: SyncAPIClient,
        model_path: str,
        g2p_model_path: Optional[str] = None,
        device: str = 'cpu',
        compressed: bool = True,
    ):
        """Load tokenizer from exported binary file"""
        if compressed:
            with gzip.open(model_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

        tokenizer = LatticeTokenizer(client_wrapper=client_wrapper)
        tokenizer.words = data['words']
        tokenizer.dictionaries = defaultdict(list, data['dictionaries'])
        tokenizer.oov_word = data['oov_word']
        if g2p_model_path:
            tokenizer.g2p_model = G2Phonemizer(g2p_model_path, device=device)
        return tokenizer

    def prenormalize(self, texts: List[str], language: Optional[str] = None) -> List[str]:
        if not self.g2p_model:
            raise ValueError('G2P model is not loaded, cannot prenormalize texts')

        oov_words = []
        for text in texts:
            words = text.lower().replace('-', ' ').replace('—', ' ').replace('–', ' ').split()
            oovs = [w for w in words if w not in self.words]
            if oovs:
                oov_words.extend([w for w in oovs if (w not in self.words and len(w) <= MAXIMUM_WORD_LENGTH)])

        oov_words = list(set(oov_words))
        if oov_words:
            indexs = []
            for k, _word in enumerate(oov_words):
                if any(_word.startswith(p) and _word.endswith(q) for (p, q) in [('(', ')'), ('[', ']')]):
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

            pronunciation_dictionaries: Dict[str, List[List[str]]] = {
                w: self.dictionaries[w] for w in oov_words if self.dictionaries[w]
            }
            return pronunciation_dictionaries

        return {}

    def tokenize(self, supervisions: List[Supervision]) -> Tuple[str, Dict[str, Any]]:
        pronunciation_dictionaries = self.prenormalize([s.text for s in supervisions])
        response = self.client_wrapper.post(
            'tokenize',
            json={
                'supervisions': [s.to_dict() for s in supervisions],
                'pronunciation_dictionaries': pronunciation_dictionaries,
            },
        )
        if response.status_code != 200:
            raise Exception(f'Failed to tokenize texts: {response.text}')
        result = response.json()
        lattice_id = result['id']
        return lattice_id, (result['lattice_graph'], result['final_state'], result.get('acoustic_scale', 1.0))

    def detokenize(
        self,
        lattice_id: str,
        lattice_results: Tuple[torch.Tensor, Any, Any, float, float],
        # return_supervisions: bool = True,
        # return_details: bool = False,
    ) -> List[Supervision]:
        emission, results, labels, frame_shift, offset, channel = lattice_results  # noqa: F841
        response = self.client_wrapper.post(
            'detokenize',
            json={
                'lattice_id': lattice_id,
                'frame_shift': frame_shift,
                'results': [t.to_dict() for t in results[0]],
                'labels': labels[0],
                'offset': offset,
                'channel': channel,
                'destroy_lattice': True,
            },
        )
        if response.status_code != 200:
            raise Exception(f'Failed to detokenize lattice: {response.text}')
        result = response.json()
        # if return_details:
        #     raise NotImplementedError("return_details is not implemented yet")
        return [Supervision.from_dict(s) for s in result['supervisions']]


# Compute average score weighted by the span length
def _score(spans):
    if not spans:
        return 0.0
    # TokenSpan(token=token, start=start, end=end, score=scores[start:end].mean().item())
    return round(sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans), ndigits=4)
