<div align="center">

<img width="1312" height="508" alt="image" src="https://github.com/user-attachments/assets/3ae59567-e2dc-48de-8e7f-b9a5faed82f7" />

</div>

<div align="center">

### Requires [Python 3.10](https://www.python.org/downloads/release/python-31011/), [3.11](https://www.python.org/downloads/release/python-3119/), [3.12](https://www.python.org/downloads/release/python-31213/), or [3.13](https://www.python.org/downloads/release/python-31312/)

</div>

## Features

- Batch transcribe multiple files recursively in directories or sub-directories.
- NVIDIA Parakeet TDT 0.6B v2 (English) and v3 (Multilingual + Translation)
- Optional timestamps with configurable segment intervals
- GPU (CUDA) and CPU inference support
- Supported file types: AAC, AMR, ASF, AVI, FLAC, M4A, MKV, MP3, MP4, WAV, WEBM, WMA

## Benchmarks

Transcription of a ~2.5 hour file [`sam_altman_lex_podcast_367.flac`](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac):

| Library | Model | Batch | Chunk | VRAM Usage | Time | [Quality Ranking](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) |
|---|---|---|---|---|---|---|
| **[Elegant Transcriber](https://github.com/BBC-Esq/Elegant-Audio-Transcriber)** | **[Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** | 1 | **90s** | **~3.3 GB** | **8.2s** | #8 |
| [Transformers](https://github.com/huggingface/transformers) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 32 | Default | ~12.4 GB | 52.2s | #32 |
| [WhisperS2T Reborn](https://github.com/BBC-Esq/whisper-s2t-reborn) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 32 | Default | ~13.4 GB | 66.9s | #32 |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 32 | Default | ~12.5 GB | 75.9s | #32 |
| [WhisperX](https://github.com/m-bain/whisperX) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 32 | Default | ~12.8 GB | 71.8s | #32 |
| [Granite Speech](https://github.com/ibm-granite/granite-speech) | [Granite 4.0 1B Speech](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) | 12 | 30s | ~6.3 GB | 97.7s | #1 |

> All models were run in ```bfloat16```.<br>
> All VRAM measurements include model weights and inference overhead.<br>
> All parameters were chosen to achieve a maximum throughput of ~90% CUDA core usage on an RTX 4090.

## Installation


### 1. Windows (installer)
> Download and run [```Elegant_Transcriber_Setup.exe```](https://github.com/BBC-Esq/Elegant-Audio-Transcriber/releases/latest/download/Elegant_Transcriber_Setup.exe).

### 2. Windows (from source)

> Download the latest release, unzip and extract, then navigate to the directory containing ```main.py``` and run:

```
python -m venv .
```
```
.\Scripts\activate
```
```
python install.py
```
```
python main.py
```

### 3. Linux (from wource)

> Download the latest release, unzip and extract, then navigate to the directory containing ```main.py``` and run:

```
python3 -m venv .
```
```
source bin/activate
```
```
python install.py
```
```
python main.py
```

