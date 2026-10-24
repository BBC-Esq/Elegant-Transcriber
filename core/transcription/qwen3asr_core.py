from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

import numpy as np
import torch

from core.logging_config import get_logger
from core.models.metadata import LANGUAGE_AUTO, ModelMetadata
from core.transcription.parakeet_core import SR, get_chunks
from core.transcription.stitching import stitch_texts_aligned

logger = get_logger(__name__)


QWEN3_OVERLAP_SECONDS = 5
QWEN3_TOKENS_PER_SECOND = 16
QWEN3_MIN_NEW_TOKENS = 200


class Qwen3ASRModel:

    def __init__(self, model, processor, device: str, dtype) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.pad_token_id = model.generation_config.pad_token_id


def _transcribe_chunk(model: Qwen3ASRModel, chunk_audio: np.ndarray, language: Optional[str]) -> Tuple[str, Optional[str]]:
    inputs = model.processor.apply_transcription_request(
        audio=[np.ascontiguousarray(chunk_audio, dtype=np.float32)], language=language
    )
    inputs = inputs.to(model.device, model.dtype)
    chunk_seconds = len(chunk_audio) / SR
    max_new_tokens = max(QWEN3_MIN_NEW_TOKENS, int(chunk_seconds * QWEN3_TOKENS_PER_SECOND))
    with torch.inference_mode():
        output_ids = model.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    generated = output_ids[:, inputs["input_ids"].shape[-1]:]
    if int((generated[0] != model.pad_token_id).sum()) >= max_new_tokens:
        logger.warning(
            f"Qwen3-ASR used its whole {max_new_tokens}-token budget on a {chunk_seconds:.0f}s chunk; "
            "that part of the transcript may repeat itself"
        )
    if language is None:
        parsed = model.processor.decode(generated, return_format="parsed")[0]
        return parsed["transcription"].strip(), parsed.get("language")
    return model.processor.decode(generated, return_format="transcription_only")[0].strip(), language


def transcribe_qwen3asr(
    model: Qwen3ASRModel,
    audio: np.ndarray,
    segment_length: int,
    language: Optional[str] = None,
    cancel_check=None,
    progress_callback=None,
) -> Tuple[str, Optional[str]]:
    forced = None if not language or language == LANGUAGE_AUTO else language
    chunks = get_chunks(audio, segment_length, QWEN3_OVERLAP_SECONDS)
    texts: list[str] = []
    spans: list[tuple[float, float]] = []
    detected: list[str] = []
    for i, (chunk_audio, offset, _has_overlap) in enumerate(chunks):
        if cancel_check and cancel_check():
            break
        if progress_callback:
            progress_callback(i, len(chunks))
        text, chunk_language = _transcribe_chunk(model, chunk_audio, forced)
        texts.append(text)
        spans.append((offset, offset + len(chunk_audio) / SR))
        if chunk_language:
            detected.append(chunk_language)

    if forced:
        reported = forced
    elif detected:
        name = Counter(detected).most_common(1)[0][0]
        reported = ModelMetadata.resolve_language(name) or name
    else:
        reported = None
    return " ".join(stitch_texts_aligned(texts, spans).split()), reported
