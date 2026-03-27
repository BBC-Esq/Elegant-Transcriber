from typing import List, Tuple


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

        overlap_word_estimate = 35
        prev_tail = prev_words[-overlap_word_estimate:]
        curr_head = curr_words[:overlap_word_estimate]
        search_window = min(len(curr_head), len(prev_tail))

        best_match_len = 0
        for match_len in range(search_window, min_match_words - 1, -1):
            prev_suffix = [w.lower() for w in prev_tail[-match_len:]]
            curr_prefix = [w.lower() for w in curr_head[:match_len]]
            if prev_suffix == curr_prefix:
                best_match_len = match_len
                break

        if best_match_len > 0:
            remaining = curr_words[best_match_len:]
            if remaining and prev_words and remaining[0].lower().rstrip(".,;:!?") == prev_words[-1].lower().rstrip(".,;:!?"):
                remaining = remaining[1:]
            if remaining:
                result = result + " " + " ".join(remaining)
        else:
            result = result + " " + texts[i]

    return result


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
                if first_kept and result and seg[2].lower().rstrip(".,;:!?") == result[-1][2].lower().rstrip(".,;:!?"):
                    first_kept = False
                    continue
                first_kept = False
                result.append(seg)

    return result
