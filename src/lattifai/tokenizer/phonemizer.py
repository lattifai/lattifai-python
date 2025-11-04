import re
from typing import List, Optional, Union

from dp.phonemizer import Phonemizer  # g2p-phonemizer
from num2words import num2words

LANGUAGE = "omni"


class G2Phonemizer:
    def __init__(self, model_checkpoint, device):
        self.phonemizer = Phonemizer.from_checkpoint(model_checkpoint, device=device).predictor
        self.pattern = re.compile(r"\d+")

    def num2words(self, word, lang: str):
        matches = self.pattern.findall(word)
        for match in matches:
            word_equivalent = num2words(int(match), lang=lang)
            word = word.replace(match, word_equivalent)
        return word

    def remove_special_tokens(self, decoded: List[str]) -> List[str]:
        return [d for d in decoded if d not in self.phonemizer.phoneme_tokenizer.special_tokens]

    def __call__(
        self, words: Union[str, List[str]], lang: Optional[StopIteration], batch_size: int = 0, num_prons: int = 1
    ):
        is_list = True
        if not isinstance(words, list):
            words = [words]
            is_list = False

        predictions = self.phonemizer(
            [self.num2words(word.replace(" .", ".").replace(".", " ."), lang=lang or "en") for word in words],
            lang=LANGUAGE,
            batch_size=min(batch_size or len(words), 128),
            num_prons=num_prons,
        )
        if num_prons > 1:
            predictions = [
                [self.remove_special_tokens(_prediction.phoneme_tokens) for _prediction in prediction]
                for prediction in predictions
            ]
        else:
            predictions = [self.remove_special_tokens(prediction.phoneme_tokens) for prediction in predictions]

        if is_list:
            return predictions

        return predictions[0]
