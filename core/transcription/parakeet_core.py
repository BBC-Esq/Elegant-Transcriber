from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import numpy as np
import torch

from core.transcription.stitching import stitch_texts_aligned, stitch_words_at_midpoints


SR = 16000
CANARY_OVERLAP_SECONDS = 5
PARAKEET_OVERLAP_SECONDS = 15
PARAKEET_BATCH_SIZE = 1


def parakeet_overlap_seconds(segment_length_seconds: float) -> float:
    return min(PARAKEET_OVERLAP_SECONDS, segment_length_seconds / 3)


def get_chunks(audio: np.ndarray, segment_length_seconds: float, overlap_seconds: float):
    segment_samples = int(segment_length_seconds * SR)
    step = segment_samples - int(overlap_seconds * SR)
    if step <= 0:
        raise ValueError(
            f"Segment length {segment_length_seconds}s must exceed the {overlap_seconds}s overlap"
        )
    audio_length = len(audio)

    if audio_length <= segment_samples:
        return [(audio, 0.0, False)]

    chunks = []
    start = 0
    while True:
        end = min(start + segment_samples, audio_length)
        chunks.append((audio[start:end], start / SR, start > 0))
        if end >= audio_length:
            break
        start += step

    return chunks


def extract_text(output) -> str:
    if not output:
        return ""
    if isinstance(output, (list, tuple)):
        first = output[0]
        if isinstance(first, str):
            return first.strip()
        if hasattr(first, 'text'):
            return first.text.strip()
        return str(first).strip()
    return str(output).strip()


def extract_word_timestamps(output, time_offset: float) -> List[Tuple[float, float, str]]:
    segments = []
    if not output or not isinstance(output, (list, tuple)):
        return segments
    hyp = output[0]
    if not hasattr(hyp, 'timestamp') or not hyp.timestamp:
        return segments
    ts = hyp.timestamp
    if not isinstance(ts, dict):
        return segments
    word_ts = ts.get('word', [])
    if not word_ts:
        word_ts = ts.get('segment', [])
    for entry in word_ts:
        start = entry.get('start', 0) + time_offset
        end = entry.get('end', start) + time_offset
        text = entry.get('word', '') or entry.get('char', '') or entry.get('segment', '')
        if text.strip():
            segments.append((start, end, text.strip()))
    return segments


def group_words_into_segments(
    word_segments: List[Tuple[float, float, str]],
    max_duration: float = 10.0,
) -> List[Tuple[float, float, str]]:
    if not word_segments:
        return []
    grouped = []
    current_words = []
    current_start = word_segments[0][0]
    for start, end, word in word_segments:
        if current_words and (end - current_start) > max_duration:
            grouped.append((
                current_start,
                current_words[-1][1],
                " ".join(w[2] for w in current_words),
            ))
            current_words = []
            current_start = start
        current_words.append((start, end, word))
    if current_words:
        grouped.append((
            current_start,
            current_words[-1][1],
            " ".join(w[2] for w in current_words),
        ))
    return grouped


def _transcribe_chunk_words(model, chunk_audio: np.ndarray, time_offset: float) -> List[Tuple[float, float, str]]:
    with torch.inference_mode():
        hypotheses = model.transcribe(
            [chunk_audio],
            batch_size=PARAKEET_BATCH_SIZE,
            timestamps=True,
            return_hypotheses=True,
            verbose=False,
        )
    words = extract_word_timestamps(hypotheses, time_offset)
    if not words:
        txt = extract_text(hypotheses)
        if txt:
            words = [(time_offset, time_offset + len(chunk_audio) / SR, txt)]
    return words


def _transcribe_words(
    model,
    audio: np.ndarray,
    segment_length: float,
    cancel_check=None,
    progress_callback=None,
) -> List[Tuple[float, float, str]]:
    chunks = get_chunks(audio, segment_length, parakeet_overlap_seconds(segment_length))
    chunk_words = []
    chunk_spans = []
    for i, (chunk_audio, time_offset, _has_overlap) in enumerate(chunks):
        if cancel_check and cancel_check():
            break
        if progress_callback:
            progress_callback(i, len(chunks))
        chunk_words.append(_transcribe_chunk_words(model, chunk_audio, time_offset))
        chunk_spans.append((time_offset, time_offset + len(chunk_audio) / SR))
    return stitch_words_at_midpoints(chunk_words, chunk_spans)


def transcribe_text(
    model,
    audio: np.ndarray,
    segment_length: int,
    cancel_check=None,
    progress_callback=None,
) -> str:
    if len(audio) > int(segment_length * SR):
        words = _transcribe_words(model, audio, segment_length, cancel_check, progress_callback)
        return " ".join(w[2] for w in words)
    if cancel_check and cancel_check():
        return ""
    if progress_callback:
        progress_callback(0, 1)
    with torch.inference_mode():
        output = model.transcribe(
            [audio],
            batch_size=PARAKEET_BATCH_SIZE,
            timestamps=False,
            verbose=False,
        )
    return extract_text(output)


def transcribe_canary(
    model,
    audio: np.ndarray,
    segment_length: int,
    cancel_check=None,
    progress_callback=None,
) -> str:
    import soundfile as sf

    chunks = get_chunks(audio, segment_length, CANARY_OVERLAP_SECONDS)
    all_texts: list[str] = []
    chunk_spans: list[tuple[float, float]] = []
    temp_files: list[str] = []

    try:
        for i, (chunk_audio, offset, _has_overlap) in enumerate(chunks):
            if cancel_check and cancel_check():
                break
            if progress_callback:
                progress_callback(i, len(chunks))

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_files.append(tmp.name)
            sf.write(tmp.name, chunk_audio, SR)
            tmp.close()

            chunk_duration = len(chunk_audio) / SR
            max_tokens = max(128, int(chunk_duration * 10))

            with torch.inference_mode():
                answer_ids = model.generate(
                    prompts=[[{
                        "role": "user",
                        "content": f"Transcribe the following: {model.audio_locator_tag}",
                        "audio": [tmp.name],
                    }]],
                    max_new_tokens=max_tokens,
                )

            txt = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
            all_texts.append(txt)
            chunk_spans.append((offset, offset + chunk_duration))
    finally:
        for tf_path in temp_files:
            try:
                os.remove(tf_path)
            except OSError:
                pass

    return stitch_texts_aligned(all_texts, chunk_spans)


def transcribe_with_timestamps(
    model,
    audio: np.ndarray,
    segment_length: int,
    segment_duration: float,
    cancel_check=None,
    progress_callback=None,
) -> List[Tuple[float, float, str]]:
    words = _transcribe_words(model, audio, segment_length, cancel_check, progress_callback)
    if words:
        return group_words_into_segments(words, max_duration=float(segment_duration))
    return []
