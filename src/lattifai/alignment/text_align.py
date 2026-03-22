# Align caption.supervisions with transcription
import logging
import string  # noqa: F401
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import regex
from error_align import error_align
from error_align.utils import DELIMITERS, NUMERIC_TOKEN, STANDARD_TOKEN, Alignment, OpType

from lattifai.caption import Supervision
from lattifai.data import Caption
from lattifai.utils import safe_print

from .punctuation import PUNCTUATION

Symbol = TypeVar("Symbol")
EPSILON = "`"

JOIN_TOKEN = "❄"
if JOIN_TOKEN not in DELIMITERS:
    DELIMITERS.add(JOIN_TOKEN)


def custom_tokenizer(text: str) -> list:
    """Default tokenizer that splits text into words based on whitespace.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list: A list of tokens (words).

    """
    # Escape JOIN_TOKEN for use in regex pattern
    escaped_join_token = regex.escape(JOIN_TOKEN)
    return list(
        regex.finditer(
            rf"({NUMERIC_TOKEN})|({STANDARD_TOKEN}|{escaped_join_token})",
            text,
            regex.UNICODE | regex.VERBOSE,
        )
    )


def equal_ratio(chunk: List[Alignment]):
    return sum(a.op_type == OpType.MATCH for a in chunk) / max(len(chunk), 1)


def equal(chunk: List[Alignment]):
    return all(a.op_type == OpType.MATCH for a in chunk)


def group_alignments(
    supervisions: List[Supervision],
    transcription: List[Supervision],
    max_silence_gap: float = 10.0,
    mini_num_supervisions: int = 1,
    mini_num_transcription: int = 1,
    equal_threshold: float = 0.5,
    verbose: bool = False,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], List[Alignment]]]:
    # TABLE = str.maketrans(dict.fromkeys(string.punctuation))
    # sup.text.lower().translate(TABLE)
    ref = "".join(sup.text.lower() + JOIN_TOKEN for sup in supervisions)
    hyp = "".join(sup.text.lower() + JOIN_TOKEN for sup in transcription)
    alignments = error_align(ref, hyp, tokenizer=custom_tokenizer)

    matches = []
    # segment start index
    ss_start, ss_idx = 0, 0
    tr_start, tr_idx = 0, 0

    idx_start = 0
    for idx, ali in enumerate(alignments):
        if ali.ref == JOIN_TOKEN:
            ss_idx += 1
        if ali.hyp == JOIN_TOKEN:
            tr_idx += 1

        if ali.ref == JOIN_TOKEN and ali.hyp == JOIN_TOKEN:
            chunk = alignments[idx_start:idx]

            split_at_silence = False
            greater_two_silence_gap = False
            if tr_idx > 0 and tr_idx < len(transcription):
                gap = transcription[tr_idx].start - transcription[tr_idx - 1].end
                if gap > max_silence_gap:
                    split_at_silence = True
                if gap > 2 * max_silence_gap:
                    greater_two_silence_gap = True

            if (
                (equal_ratio(chunk[:10]) > equal_threshold or equal_ratio(chunk[-10:]) > equal_threshold)
                and (
                    (ss_idx - ss_start >= mini_num_supervisions and tr_idx - tr_start >= mini_num_transcription)
                    or split_at_silence
                )
            ) or greater_two_silence_gap:
                matches.append(((ss_start, ss_idx), (tr_start, tr_idx), chunk))

                if verbose:
                    sub_align = supervisions[ss_start:ss_idx]
                    asr_align = transcription[tr_start:tr_idx]
                    safe_print("========================================================================")
                    safe_print(f"   Caption [{ss_start:>4d}, {ss_idx:>4d}): {[sup.text for sup in sub_align]}")
                    safe_print(f"Transcript [{tr_start:>4d}, {tr_idx:>4d}): {[sup.text for sup in asr_align]}")
                    safe_print("========================================================================\n\n")

                ss_start = ss_idx
                tr_start = tr_idx
                idx_start = idx + 1

        if ss_start == len(supervisions) and tr_start == len(transcription):
            break

        # remainder
        if ss_idx == len(supervisions) or tr_idx == len(transcription):
            chunk = alignments[idx_start:]
            matches.append(((ss_start, len(supervisions)), (tr_start, len(transcription)), chunk))
            break

    return matches


class AlignQuality(namedtuple("AlignQuality", ["FW", "LW", "PREFIX", "SUFFIX", "WER"])):
    def __repr__(self) -> str:
        quality = f"WORD[{self.FW}][{self.LW}]_WER[{self.WER}]"
        return quality

    @property
    def info(self) -> str:
        info = f"WER {self.WER.WER:.4f} accuracy [{self.PREFIX:.2f}, {self.SUFFIX:.2f}] {self.WER}"
        return info

    @property
    def first_word_equal(self) -> bool:
        return self.FW == "FE"

    @property
    def last_word_equal(self) -> bool:
        return self.LW == "LE"

    @property
    def wer(self) -> float:
        return self.WER.WER

    @property
    def qwer(self):
        wer = self.wer
        # 考虑 ref_len
        if wer == 0.0:
            return "WZ"  # zero
        elif wer < 0.1:
            return "WL"  # low
        elif wer < 0.32:
            return "WM"  # medium
        else:
            return "WH"  # high

    @property
    def qprefix(self) -> str:
        if self.PREFIX >= 0.7:
            return "PH"  # high
        elif self.PREFIX >= 0.5:
            return "PM"  # medium
        else:
            return "PL"  # low

    @property
    def qsuffix(self) -> str:
        if self.SUFFIX > 0.7:
            return "SH"  # high
        elif self.SUFFIX > 0.5:
            return "SM"  # medium
        else:
            return "SL"  # low


class TimestampQuality(namedtuple("TimestampQuality", ["start", "end"])):
    @property
    def start_diff(self):
        return abs(self.start[0] - self.start[1])

    @property
    def end_diff(self):
        return abs(self.end[0] - self.end[1])

    @property
    def diff(self):
        return max(self.start_diff, self.end_diff)


TextAlignResult = Tuple[Optional[List[Supervision]], Optional[List[Supervision]], AlignQuality, TimestampQuality, int]


def align_supervisions_and_transcription(
    caption: Caption,
    max_duration: Optional[float] = None,
    verbose: bool = False,
) -> List[TextAlignResult]:
    """Align caption.supervisions with caption.transcription.

    Args:
        caption: Caption object containing supervisions and transcription.

    """
    groups = group_alignments(caption.supervisions, caption.transcription, verbose=False)

    if max_duration is None:
        max_duration = max(caption.transcription[-1].end, caption.supervisions[-1].end)
    else:
        max_duration = min(
            max_duration,
            max(caption.transcription[-1].end, caption.supervisions[-1].end) + 10.0,
        )

    def next_start(alignments: List[Supervision], idx: int) -> float:
        if idx < len(alignments):
            return alignments[idx].start
        return min(alignments[-1].end + 2.0, max_duration)

    wer_filter = WERFilter()

    matches = []
    for idx, ((sub_start, sub_end), (asr_start, asr_end), chunk) in enumerate(groups):
        sub_align = caption.supervisions[sub_start:sub_end]
        asr_align = caption.transcription[asr_start:asr_end]

        if not sub_align or not asr_align:
            if sub_align:
                if matches:
                    _asr_start = matches[-1][-2].end[1]
                else:
                    _asr_start = 0.0
                startends = [
                    (sub_align[0].start, sub_align[-1].end),
                    (_asr_start, next_start(caption.transcription, asr_end)),
                ]
            elif asr_align:
                if matches:
                    _sub_start = matches[-1][-2].end[0]
                else:
                    _sub_start = 0.0
                startends = [
                    (_sub_start, next_start(caption.supervisions, sub_end)),
                    (asr_align[0].start, asr_align[-1].end),
                ]
            else:
                raise ValueError(
                    f"Never Here! subtitles[{len(caption.supervisions)}] {sub_start}-{sub_end} asrs[{len(caption.transcription)}] {asr_start}-{asr_end}"
                )

            if verbose:
                safe_print("oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
                safe_print(
                    f"   Caption idx=[{sub_start:>4d}, {sub_end:>4d}) timestamp=[{startends[0][0]:>8.2f}, {startends[0][1]:>8.2f}]: {[sup.text for sup in sub_align]}"
                )
                safe_print(
                    f"Transcript idx=[{asr_start:>4d}, {asr_end:>4d}) timestamp=[{startends[1][0]:>8.2f}, {startends[1][1]:>8.2f}]: {[sup.text for sup in asr_align]}"
                )
                safe_print("oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo\n\n")

            aligned, timestamp = quality(chunk, startends[0], startends[1], wer_fn=wer_filter)
            matches.append([sub_align, asr_align, aligned, timestamp, chunk])
            continue
        else:
            aligned, timestamp = quality(
                chunk,
                [sub_align[0].start, sub_align[-1].end],
                [asr_align[0].start, asr_align[-1].end],
                wer_fn=wer_filter,
            )
            matches.append([sub_align, asr_align, aligned, timestamp, chunk])

            if verbose and aligned.wer > 0.0:
                safe_print(
                    f"===================================WER={aligned.wer:>4.2f}====================================="
                )
                safe_print(
                    f"   Caption idx=[{sub_start:>4d}, {sub_end:>4d}) timestamp=[{sub_align[0].start:>8.2f}, {sub_align[-1].end:>8.2f}]: {[sup.text for sup in sub_align]}"
                )
                safe_print(
                    f"Transcript idx=[{asr_start:>4d}, {asr_end:>4d}) timestamp=[{asr_align[0].start:>8.2f}, {asr_align[-1].end:>8.2f}]: {[sup.text for sup in asr_align]}"
                )
                safe_print("========================================================================\n\n")

    return matches


class AlignFilter(ABC):

    def __init__(
        self, PUNCTUATION: str = PUNCTUATION, IGNORE: str = "", SPACE=" ", EPSILON=EPSILON, SEPARATOR=JOIN_TOKEN
    ):
        super().__init__()
        self._name = self.__class__.__name__
        self.PUNCTUATION = PUNCTUATION

        self.IGNORE = IGNORE
        self.SPACE = SPACE
        self.EPSILON = EPSILON
        self.SEPARATOR = SEPARATOR

        self.PUNCTUATION_SEPARATOR = PUNCTUATION + SEPARATOR
        self.PUNCTUATION_SPACE = PUNCTUATION + SPACE
        self.PUNCTUATION_SPACE_SEPARATOR = PUNCTUATION + SPACE + SEPARATOR

    @abstractmethod
    def __call__(self, chunk: List[Alignment]) -> str:
        pass

    @property
    def name(self):
        return self._name


class WERStats(namedtuple("AlignStats", ["WER", "ins_errs", "del_errs", "sub_errs", "ref_len"])):
    def to_dict(self):
        return {
            "WER": self.WER,
            "ins_errs": self.ins_errs,
            "del_errs": self.del_errs,
            "sub_errs": self.sub_errs,
            "ref_len": self.ref_len,
        }


# https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
def compute_align_stats(
    ali: List[Tuple[str, str]],
    ERR: str = "*",
    IGNORE: str = "",
    enable_log: bool = True,
) -> WERStats:
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0

    ref_len = 0
    skip = 0
    for k, (ref_word, hyp_word) in enumerate(ali):
        if skip > 0:
            skip -= 1
            continue

        # compute_align_stats(ali, ERR=EPSILON, IGNORE=PUNCTUATION_SPACE_SEPARATOR)
        if ali[k : k + 1] in [
            [("is", "'s")],  # what is -> what's
            [("am", "'m")],  # I am -> I'm
            [("are", "'re")],  # they are -> they're
            [("would", "'d")],  # they would -> they'd don't
            [("had", "'d")],  # I had -> I'd
            # we will -> we'll
            [("will", "'ll")],
            # I have -> I've
            [("have", "'ve")],
            # ok -> okay
            [("ok", "okay")],
            # okay -> ok
            [("okay", "ok")],
        ]:
            skip = 1
            ref_len += 1
            continue
        elif ali[k : k + 2] in [
            # let us -> let's
            [("let", "let"), ("us", "'s")],
            # do not -> don't
            [("do", "do"), ("not", "n't")],
        ]:
            skip = 2
            ref_len += 2
            continue
        elif (ref_word and ref_word in IGNORE) and (hyp_word and hyp_word in IGNORE):
            continue
        else:
            ref_len += 1

        if ref_word == ERR:
            ins[hyp_word] += 1
            words[hyp_word][3] += 1
        elif hyp_word == ERR:
            dels[ref_word] += 1
            words[ref_word][4] += 1
        elif hyp_word != ref_word:
            subs[(ref_word, hyp_word)] += 1
            words[ref_word][1] += 1
            words[hyp_word][2] += 1
        else:
            words[ref_word][0] += 1
            num_corr += 1

    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs

    stats = WERStats(
        WER=round(tot_errs / max(ref_len, 1), ndigits=4),
        ins_errs=ins_errs,
        del_errs=del_errs,
        sub_errs=sub_errs,
        ref_len=ref_len,
    )

    if enable_log:
        logging.info(
            f"%WER {stats.WER:.4%} "
            f"[{tot_errs} / {max(ref_len, 1)}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    return stats


class WERFilter(AlignFilter):
    def __call__(self, chunk: List[Alignment]) -> WERStats:
        ali = [(a.ref, a.hyp) for a in chunk]
        stats = compute_align_stats(ali, ERR=JOIN_TOKEN, IGNORE=self.PUNCTUATION_SPACE_SEPARATOR, enable_log=False)
        return stats


def quality(
    chunk: List[Alignment], supervision: Tuple[float, float], transcript: Tuple[float, float], wer_fn: Callable
) -> Tuple[AlignQuality, TimestampQuality]:
    _quality = AlignQuality(
        FW="FE" if chunk and chunk[0].op_type == OpType.MATCH else "FN",
        LW="LE" if chunk and chunk[-1].op_type == OpType.MATCH else "LN",
        PREFIX=equal_ratio(chunk[:4]),
        SUFFIX=equal_ratio(chunk[-4:]),
        WER=wer_fn(chunk),
    )
    timestamp = TimestampQuality(start=(supervision[0], transcript[0]), end=(supervision[1], transcript[1]))
    return _quality, timestamp


# ---------------------------------------------------------------------------
# Nearby duplicate block detection
# ---------------------------------------------------------------------------

DuplicateBlock = namedtuple("DuplicateBlock", ["first", "second", "matched_words", "time_gap"])


def detect_duplicate_blocks(
    supervisions: List[Supervision],
    ngram: int = 8,
    min_match_words: int = 10,
    max_word_gap: int = 300,
    max_time_gap: float = 300.0,
) -> List[DuplicateBlock]:
    """Detect nearby duplicate text blocks caused by editing errors.

    Concatenates all supervision text into a word stream, then uses N-gram
    fingerprinting to find repeated subsequences that are close in both word
    distance and timestamp.

    Args:
        supervisions: List of Supervision segments.
        ngram: N-gram size for fingerprinting.
        min_match_words: Minimum number of consecutive matching words.
        max_word_gap: Maximum word-level distance between two blocks.
        max_time_gap: Maximum time distance (seconds) between two blocks.

    Returns:
        List of DuplicateBlock(first=(seg_start, seg_end),
                               second=(seg_start, seg_end),
                               matched_words, time_gap).
    """
    if len(supervisions) < 2:
        return []

    # Build word stream with segment index mapping (multilingual-aware)
    from .tokenizer import tokenize_multilingual_text

    words: List[str] = []
    word_to_seg: List[int] = []
    for seg_idx, sup in enumerate(supervisions):
        tokens = tokenize_multilingual_text(sup.text.lower(), keep_spaces=False)
        for w in tokens:
            words.append(w)
            word_to_seg.append(seg_idx)

    if len(words) < ngram * 2:
        return []

    # Build N-gram fingerprint index
    fp_map: Dict[tuple, List[int]] = defaultdict(list)
    for i in range(len(words) - ngram):
        fp_map[tuple(words[i : i + ngram])].append(i)

    # Find nearby matching pairs
    seen: set = set()
    candidates = []
    for positions in fp_map.values():
        if len(positions) < 2:
            continue
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                pos_a, pos_b = positions[a], positions[b]
                if pos_b - pos_a > max_word_gap:
                    continue
                if abs(supervisions[word_to_seg[pos_b]].start - supervisions[word_to_seg[pos_a]].start) > max_time_gap:
                    continue
                key = (pos_a // 10, pos_b // 10)
                if key in seen:
                    continue
                seen.add(key)

                # Extend match forward (exact word match)
                match_len = ngram
                limit = min(len(words) - pos_a, len(words) - pos_b)
                while match_len < limit and words[pos_a + match_len] == words[pos_b + match_len]:
                    match_len += 1

                if match_len >= min_match_words:
                    time_gap = abs(supervisions[word_to_seg[pos_b]].start - supervisions[word_to_seg[pos_a]].start)
                    candidates.append((pos_a, pos_b, match_len, time_gap))

    # Keep longest non-overlapping blocks
    candidates.sort(key=lambda x: -x[2])
    results: List[DuplicateBlock] = []
    used: set = set()
    for pa, pb, mlen, tgap in candidates:
        if any(i in used for i in range(pa, pa + mlen)) or any(i in used for i in range(pb, pb + mlen)):
            continue
        used.update(range(pa, pa + mlen))
        used.update(range(pb, pb + mlen))

        seg_a = (word_to_seg[pa], word_to_seg[min(pa + mlen - 1, len(word_to_seg) - 1)])
        seg_b = (word_to_seg[pb], word_to_seg[min(pb + mlen - 1, len(word_to_seg) - 1)])
        # Skip intra-segment matches (e.g., parallel structures like
        # "接受X作为输入，并生成X的输出" where X repeats within the same supervision)
        if seg_a == seg_b:
            continue
        results.append(DuplicateBlock(first=seg_a, second=seg_b, matched_words=mlen, time_gap=tgap))

    return sorted(results, key=lambda x: x.first[0])


def deduplicate_supervisions(
    supervisions: List[Supervision],
) -> Tuple[List[Supervision], List[DuplicateBlock]]:
    """Detect and remove nearby duplicate blocks from supervisions.

    For each duplicate pair, removes the block whose segments span a shorter
    time range (likely the compressed/edited copy).

    Returns:
        Tuple of (cleaned_supervisions, detected_duplicates).
    """
    duplicates = detect_duplicate_blocks(supervisions)

    if not duplicates:
        return supervisions, duplicates

    # Collect segment indices to remove (shorter-duration copy)
    remove_indices: set = set()
    for dup in duplicates:
        a_start, a_end = dup.first
        b_start, b_end = dup.second
        a_duration = supervisions[a_end].end - supervisions[a_start].start
        b_duration = supervisions[b_end].end - supervisions[b_start].start
        if b_duration < a_duration:
            remove_indices.update(range(b_start, b_end + 1))
        else:
            remove_indices.update(range(a_start, a_end + 1))

    cleaned = [sup for i, sup in enumerate(supervisions) if i not in remove_indices]
    return cleaned, duplicates
