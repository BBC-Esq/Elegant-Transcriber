from __future__ import annotations

from pathlib import Path
import numpy as np
import av


PARAKEET_SAMPLE_RATE = 16000


def load_audio_for_parakeet(audio_path: str | Path, target_sr: int = PARAKEET_SAMPLE_RATE) -> np.ndarray:
    container = av.open(str(audio_path))
    try:
        resampler = av.AudioResampler(format='s16', layout='mono', rate=target_sr)
        frames = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                frames.append(r.to_ndarray())
    finally:
        container.close()
    if not frames:
        raise ValueError(f"No audio frames decoded from {audio_path}")
    return np.concatenate(frames, axis=1).flatten().astype(np.float32) / 32768.0
