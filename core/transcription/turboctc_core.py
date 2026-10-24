from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from core.transcription.parakeet_core import SR, group_words_into_segments, transcribe_words


CTC_FRAME_SECONDS = 0.08


class TurboCTCModel:

    def __init__(self, model, processor, device: str, dtype) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype


def _group_pieces(pieces: List[Tuple[str, float, float]], time_offset: float) -> List[Tuple[float, float, str]]:
    words = []
    for text, start, end in pieces:
        if not words or text.startswith(" "):
            words.append([start + time_offset, end + time_offset, text.strip()])
        else:
            words[-1][1] = end + time_offset
            words[-1][2] += text
    return [tuple(w) for w in words if w[2]]


def _ctc_words(tokenizer, frame_ids: List[int], time_offset: float) -> List[Tuple[float, float, str]]:
    skip = set(tokenizer.all_special_ids)
    runs, prev = [], None
    for frame, token_id in enumerate(frame_ids):
        if token_id == prev:
            if runs and token_id not in skip:
                runs[-1][2] = frame + 1
            continue
        prev = token_id
        if token_id not in skip:
            runs.append([token_id, frame, frame + 1])
    pieces = []
    for (_, first, last), piece in zip(runs, tokenizer.convert_ids_to_tokens([r[0] for r in runs])):
        text = tokenizer.convert_tokens_to_string([piece])
        if piece.startswith("▁"):
            text = " " + text.lstrip()
        pieces.append((text, first * CTC_FRAME_SECONDS, last * CTC_FRAME_SECONDS))
    return _group_pieces(pieces, time_offset)


def _transcribe_chunk_words(model: TurboCTCModel, chunk_audio: np.ndarray, time_offset: float) -> List[Tuple[float, float, str]]:
    inputs = model.processor(chunk_audio, sampling_rate=SR, device=model.device, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        frame_ids = model.model.generate(**inputs)
    return _ctc_words(model.processor.tokenizer, frame_ids[0].tolist(), time_offset)


def transcribe_turboctc(
    model: TurboCTCModel,
    audio: np.ndarray,
    segment_length: int,
    cancel_check=None,
    progress_callback=None,
) -> str:
    words = transcribe_words(
        model, audio, segment_length, cancel_check, progress_callback, _transcribe_chunk_words
    )
    return " ".join(w[2] for w in words)


def transcribe_turboctc_with_timestamps(
    model: TurboCTCModel,
    audio: np.ndarray,
    segment_length: int,
    segment_duration: float,
    cancel_check=None,
    progress_callback=None,
) -> List[Tuple[float, float, str]]:
    words = transcribe_words(
        model, audio, segment_length, cancel_check, progress_callback, _transcribe_chunk_words
    )
    return group_words_into_segments(words, max_duration=float(segment_duration))
