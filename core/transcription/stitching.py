import re
from difflib import SequenceMatcher
from typing import List, Tuple

MIN_MATCHED_RUN = 3


def _normalize_word(word: str) -> str:
    return re.sub(r'[^\w]', '', word).lower()


def _normalize_words(words: List[str]) -> List[str]:
    return [_normalize_word(w) for w in words]


def stitch_texts_aligned(texts: List[str], spans: List[Tuple[float, float]]) -> str:
    if len(texts) == 1:
        return texts[0]

    words: List[str] = []
    last = None
    for i, text in enumerate(texts):
        current = text.split()
        if not current:
            continue
        overlap = spans[last][1] - spans[i][0] if last is not None else 0.0
        if overlap <= 0 or not words:
            words += current
            last = i
            continue

        prev_rate = len(texts[last].split()) / max(spans[last][1] - spans[last][0], 1e-6)
        curr_rate = len(current) / max(spans[i][1] - spans[i][0], 1e-6)
        tail_len = min(len(words), 2 * max(1, round(overlap * prev_rate)) + 5)
        head = current[:2 * max(1, round(overlap * curr_rate)) + 5]

        matcher = SequenceMatcher(
            None, _normalize_words(words[-tail_len:]), _normalize_words(head), autojunk=False
        )
        run = matcher.find_longest_match(0, tail_len, 0, len(head))
        if run.size >= MIN_MATCHED_RUN:
            cut = len(words) - tail_len + run.a + run.size // 2
            head_idx = run.b + run.size // 2
        else:
            cut = len(words)
            head_idx = 0

        words = words[:cut] + current[head_idx:]
        last = i

    return " ".join(words)


def stitch_words_at_midpoints(
    chunk_words: List[List[Tuple[float, float, str]]],
    chunk_spans: List[Tuple[float, float]],
) -> List[Tuple[float, float, str]]:
    merged = []
    last = len(chunk_words) - 1
    for i, words in enumerate(chunk_words):
        lo = (chunk_spans[i][0] + chunk_spans[i - 1][1]) / 2 if i > 0 else float("-inf")
        hi = (chunk_spans[i + 1][0] + chunk_spans[i][1]) / 2 if i < last else float("inf")
        merged.extend(w for w in words if lo <= (w[0] + w[1]) / 2 < hi)
    return merged
