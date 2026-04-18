import re
from difflib import SequenceMatcher
from typing import List, Tuple

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "dept", "est", "govt", "inc", "corp", "ltd", "co", "vs", "etc",
    "approx", "dept", "fig", "vol", "no", "gen", "gov", "sgt", "cpl",
    "pvt", "capt", "lt", "col", "maj", "cmdr", "adm", "rev", "hon",
}


def _normalize_word(word: str) -> str:
    return re.sub(r'[^\w]', '', word).lower()


def _normalize_words(words: List[str]) -> List[str]:
    return [_normalize_word(w) for w in words]


def _is_boundary_punctuation_artifact(last_word: str, next_word: str) -> bool:
    if not last_word or not next_word:
        return False

    if not re.search(r'[.!?]$', last_word):
        return False

    stripped = re.sub(r'[.!?]+$', '', last_word)
    if not stripped:
        return False

    if stripped.lower() in ABBREVIATIONS:
        return False

    if re.match(r'^[A-Z]{1,4}$', stripped):
        return False

    if len(stripped) <= 2 and stripped[0].isupper():
        return False

    if next_word[0].islower():
        return True

    return False


def _strip_trailing_sentence_punctuation(word: str) -> str:
    return re.sub(r'[.!?]+$', '', word)


def stitch_texts(texts: List[str], had_overlap: List[bool],
                 model_type: str = "parakeet") -> str:
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    min_match_words = 3 if model_type == "canary" else 5

    result = texts[0]
    for i in range(1, len(texts)):
        if not had_overlap[i] or not texts[i]:
            if texts[i]:
                result = result + " " + texts[i]
            continue

        prev_words = result.split()
        curr_words = texts[i].split()

        overlap_word_estimate = 70
        prev_tail = prev_words[-overlap_word_estimate:]
        curr_head = curr_words[:overlap_word_estimate]

        best_match_len = _find_exact_overlap(prev_tail, curr_head, min_match_words)

        if best_match_len == 0:
            best_match_len = _find_fuzzy_overlap(prev_tail, curr_head, min_match_words)

        if best_match_len > 0:
            remaining = curr_words[best_match_len:]
            if (remaining and prev_words
                    and _normalize_word(remaining[0]) == _normalize_word(prev_words[-1])):
                remaining = remaining[1:]

            if remaining and prev_words:
                if _is_boundary_punctuation_artifact(prev_words[-1], remaining[0]):
                    fixed_word = _strip_trailing_sentence_punctuation(prev_words[-1])
                    prev_words[-1] = fixed_word
                    result = " ".join(prev_words)

            if remaining:
                result = result + " " + " ".join(remaining)
        else:
            result = result + " " + texts[i]

    return result


def _find_exact_overlap(prev_tail: List[str], curr_head: List[str], min_match: int) -> int:
    prev_norm = _normalize_words(prev_tail)
    curr_norm = _normalize_words(curr_head)
    search_window = min(len(curr_norm), len(prev_norm))

    for match_len in range(search_window, min_match - 1, -1):
        if prev_norm[-match_len:] == curr_norm[:match_len]:
            return match_len

    return 0


def _find_fuzzy_overlap(prev_tail: List[str], curr_head: List[str], min_match: int) -> int:
    prev_norm = _normalize_words(prev_tail)
    curr_norm = _normalize_words(curr_head)

    matcher = SequenceMatcher(None, prev_norm, curr_norm, autojunk=False)

    best = matcher.find_longest_match(0, len(prev_norm), 0, len(curr_norm))

    if best.size < min_match:
        return 0

    overlap_end_in_curr = best.b + best.size

    match_near_prev_end = (best.a + best.size) >= len(prev_norm) - 5
    match_near_curr_start = best.b <= 5

    if not (match_near_prev_end and match_near_curr_start):
        return 0

    return overlap_end_in_curr


def stitch_timestamp_segments(
    all_chunk_segments: List[List[Tuple[float, float, str]]],
    had_overlap: List[bool],
) -> List[Tuple[float, float, str]]:
    if not all_chunk_segments:
        return []

    result = list(all_chunk_segments[0])

    for i in range(1, len(all_chunk_segments)):
        curr_segments = all_chunk_segments[i]
        if not curr_segments:
            continue

        if not had_overlap[i] or not result:
            result.extend(curr_segments)
            continue

        last_end_time = result[-1][1] if result else 0.0

        first_kept = True
        for seg in curr_segments:
            if seg[0] >= last_end_time - 0.05:
                if first_kept and result and _normalize_word(seg[2]) == _normalize_word(result[-1][2]):
                    first_kept = False
                    continue
                first_kept = False
                result.append(seg)

    return result
