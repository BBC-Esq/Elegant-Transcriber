<div align="center">

<img width="1312" height="508" alt="image" src="https://github.com/user-attachments/assets/3ae59567-e2dc-48de-8e7f-b9a5faed82f7" />

</div>

## 4x faster than the fastest Whisper implementation AND higher accuracy.
- Batch transcribe multiple files recursively in directories or sub-directories.
- Optional timestamps with configurable segment intervals
- Works on GPU (CUDA) and CPU, Windows or Linux
- Supported file types: AAC, AMR, ASF, AVI, FLAC, M4A, MKV, MP3, MP4, WAV, WEBM, WMA

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

### 3. Linux (from source)

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

## Benchmarks (GPU)
* 2.5 hour audio file here: [`sam_altman_lex_podcast_367.flac`](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac)

| Library | Model | Batch | Chunk | VRAM Usage | Time | Real Time | [Quality Ranking](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) |
|---|---|---|---|---|---|---|---|
| **[Elegant Transcriber](https://github.com/BBC-Esq/Elegant-Audio-Transcriber)** | **[Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** | 1 | **90s** | **~3.3 GB** | **14.9s** | 580x | #8 |
| [Transformers](https://github.com/huggingface/transformers) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 32 | Default | ~12.4 GB | 52.2s | 166x | #32 |
| [WhisperS2T Reborn](https://github.com/BBC-Esq/whisper-s2t-reborn) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 32 | Default | ~13.4 GB | 66.9s | 129x | #32 |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 32 | Default | ~12.5 GB | 75.9s | 114x | #32 |
| [WhisperX](https://github.com/m-bain/whisperX) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 32 | Default | ~12.8 GB | 71.8s | 120x | #32 |
| [Granite Speech](https://github.com/ibm-granite/granite-speech) | [Granite 4.0 1B Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | 12 | 30s | ~6.3 GB | 97.7s | 88x | #1 |

> All models were run in ```bfloat16```.<br>
> All VRAM measurements include model weights and inference overhead and subtract background usage.<br>
> All parameters were chosen to achieve a maximum throughput of ~90% CUDA core usage on an RTX 4090.

## Benchmarks (CPU)
* 12 minute 58 second private audio file.
* CPU tests use a shorter audio sample to keep runtimes manageable.

| Library | Model | Batch | Chunk | RAM Usage | Time | Real Time | [Quality Ranking](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) |
|---|---|---|---|---|---|---|---|
| **[Elegant Transcriber](https://github.com/BBC-Esq/Elegant-Audio-Transcriber)** | **[Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** | 1 | **90s** | **~5.6 GB** | **29.0s** | **26.8x** | #8 |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 1 | Default | ~6.5 GB | 211.8s | 3.67x | #32 |
| [WhisperS2T Reborn](https://github.com/BBC-Esq/whisper-s2t-reborn) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 1 | Default | ~6.6 GB | 257.9s | 3.02x | #32 |
| [Transformers](https://github.com/huggingface/transformers) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 1 | Default | ~6.6 GB | 311.1s | 2.50x | #32 |
| [WhisperX](https://github.com/m-bain/whisperX) | [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) | 1 | Default | ~7.3 GB | 396.4s | 1.96x | #32 |

> All models were loaded in ```float32``` for CPU compatibility.<br>
> 20 threads were used on an Intel 13900k resulting in ~90% CPU usage.<br>
> I couldn't get Granite Speech to run...

## Special Thanks
* Nvidia for the Canary and Parkeet models, which are the best balance of accuracy and compute time for commonfolk like myself.
* IBM for Granite Speech Models, which, as of March, 2026, rank #1 on the ASR leaderboard in terms of accuracy.
* OpenAI for the older Whisper models.
