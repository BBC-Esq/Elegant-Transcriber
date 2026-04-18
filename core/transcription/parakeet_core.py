from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import numpy as np
import torch

from core.transcription.stitching import stitch_texts, stitch_timestamp_segments


SR = 16000
CHUNK_OVERLAP_SECONDS = 7
PARAKEET_BATCH_SIZE = 1


def get_chunks(audio: np.ndarray, segment_length_seconds: int):
    segment_samples = segment_length_seconds * SR
    overlap_samples = CHUNK_OVERLAP_SECONDS * SR
    audio_length = len(audio)

    if audio_length <= segment_samples:
        return [(audio, 0.0, False)]

    chunks = []
    start = 0
    while start < audio_length:
        end = min(start + segment_samples, audio_length)
        is_not_first = start > 0
        chunks.append((audio[start:end], start / SR, is_not_first))

        next_start = start + segment_samples - overlap_samples
        if next_start >= audio_length:
            break
        if audio_length - next_start < overlap_samples * 2:
            chunks.append((audio[next_start:audio_length], next_start / SR, True))
            break
        start = next_start

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


def transcribe_text(
    model,
    audio: np.ndarray,
    segment_length: int,
    cancel_check=None,
    progress_callback=None,
) -> str:
    chunks = get_chunks(audio, segment_length)
    all_texts = []
    had_overlap = []
    for i, (chunk_audio, _offset, has_overlap) in enumerate(chunks):
        if cancel_check and cancel_check():
            break
        if progress_callback:
            progress_callback(i, len(chunks))
        with torch.inference_mode():
            output = model.transcribe(
                [chunk_audio],
                batch_size=PARAKEET_BATCH_SIZE,
                timestamps=False,
                verbose=False,
            )
        txt = extract_text(output)
        all_texts.append(txt)
        had_overlap.append(has_overlap)
    return stitch_texts(all_texts, had_overlap, "parakeet")


def transcribe_canary(
    model,
    audio: np.ndarray,
    segment_length: int,
    cancel_check=None,
    progress_callback=None,
) -> str:
    import soundfile as sf

    chunks = get_chunks(audio, segment_length)
    all_texts: list[str] = []
    had_overlap: list[bool] = []
    temp_files: list[str] = []

    try:
        for i, (chunk_audio, _offset, has_overlap) in enumerate(chunks):
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
            had_overlap.append(has_overlap)
    finally:
        for tf_path in temp_files:
            try:
                os.remove(tf_path)
            except OSError:
                pass

    return stitch_texts(all_texts, had_overlap, "canary")


def transcribe_with_timestamps(
    model,
    audio: np.ndarray,
    segment_length: int,
    segment_duration: float,
    cancel_check=None,
    progress_callback=None,
) -> List[Tuple[float, float, str]]:
    chunks = get_chunks(audio, segment_length)
    all_chunk_words = []
    had_overlap = []
    for i, (chunk_audio, time_offset, has_overlap) in enumerate(chunks):
        if cancel_check and cancel_check():
            break
        if progress_callback:
            progress_callback(i, len(chunks))
        with torch.inference_mode():
            hypotheses = model.transcribe(
                [chunk_audio],
                batch_size=PARAKEET_BATCH_SIZE,
                timestamps=True,
                return_hypotheses=True,
                verbose=False,
            )
        word_segs = extract_word_timestamps(hypotheses, time_offset)
        if not word_segs:
            txt = extract_text(hypotheses)
            if txt:
                chunk_end = time_offset + len(chunk_audio) / SR
                word_segs = [(time_offset, chunk_end, txt)]
        all_chunk_words.append(word_segs or [])
        had_overlap.append(has_overlap)

    stitched_words = stitch_timestamp_segments(all_chunk_words, had_overlap)
    if stitched_words:
        return group_words_into_segments(stitched_words, max_duration=float(segment_duration))
    return []
