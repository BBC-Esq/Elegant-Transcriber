<div align="center">

<img width="1312" height="508" alt="image" src="https://github.com/user-attachments/assets/3ae59567-e2dc-48de-8e7f-b9a5faed82f7" />

</div>

<div align="center">

### Requires [Python 3.10](https://www.python.org/downloads/release/python-31011/), [3.11](https://www.python.org/downloads/release/python-3119/), [3.12](https://www.python.org/downloads/release/python-31213/), or [3.13](https://www.python.org/downloads/release/python-31312/)

</div>

## Features

- Batch transcribe multiple files recursivelyk in directories or sub-directories.
- NVIDIA Parakeet TDT 0.6B v2 (English) and v3 (Multilingual + Translation)
- Optional timestamps with configurable segment intervals
- GPU (CUDA) and CPU inference support
- Supported file types: AAC, AMR, ASF, AVI, FLAC, M4A, MKV, MP3, MP4, WAV, WEBM, WMA

## Benchmarks

Transcription of a 2.5 hour file [`sam_altman_lex_podcast_367.flac`](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac) on an RTX 4090 and 13900k CPU

| Library | Model | Device | Batch Size / Chunk Length | VRAM Usage | Time |
|---|---|---|---|---|---|
| **Elegant Transcriber** | **Parakeet TDT 0.6B v2** | **CUDA** | **90s chunks** | **~3.3 GB** | **8.2s** |
| Transformers | Whisper Large v3 | CUDA | Batch 16 | ~8.2 GB | 54.6s |
| WhisperS2T Reborn | Whisper Large v3 | CUDA | Batch 16 | ~8.8 GB | 69.4s |
| Faster-Whisper | Whisper Large v3 | CUDA | Batch 16 | ~8.4 GB | 80.1s |
| WhisperX | Whisper Large v3 | CUDA | Batch 16 | ~8.8 GB | 87.3s |
| Granite Speech | Granite 4.0 1B Speech | CUDA | Batch 8 / 45s chunks | ~18.3 GB | 136.6s |
| Elegant Transcriber | Parakeet TDT 0.6B v2 | CPU | 30s chunks | N/A | 281.7s |

> VRAM includes model weights plus inference overhead. All models run in bfloat16 except the CPU test was in float32.

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

