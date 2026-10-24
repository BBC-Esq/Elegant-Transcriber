<div align="center">

<img width="768" height="512" alt="image" src="https://github.com/user-attachments/assets/1f37e28e-cf7c-4a0a-8f57-029ba0ee312a" />

</div>

## Up to 1,949x faster than real time, or down to a 2.23% word error rate.
- Batch transcribe multiple files in a directory and optionally all sub-directories.
- Optional timestamps with configurable segment intervals (Parakeet and Granite TurboCTC models)
- Works on GPU (CUDA) and CPU, Windows or Linux (the CPU runs the Parakeet and Granite TurboCTC models)
- Supported file types: AAC, AMR, ASF, AVI, FLAC, M4A, MKV, MP3, MP4, WAV, WEBM, WMA
- [Link to article on Medium](https://medium.com/@vici0549/the-best-audio-transcriber-period-3249fa92f531)

## Installation

> GPU mode needs an NVIDIA GeForce GTX 16 / RTX 20 series card or newer, with driver version 580 or newer. Other computers can run it on the CPU.

### 1. Windows (installer)
> Download and run [```Elegant_Transcriber_Setup.exe```](https://github.com/BBC-Esq/Elegant-Transcriber/releases/latest/download/Elegant_Transcriber_Setup.exe) (right-click and run as administrator)

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
* ~2.4 hour audio file here: [`sam_altman_lex_podcast_367.flac`](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac)

| Library | Model | VRAM Usage | Real Time | WER |
|---|---|---|---|---|
| [Transformers](https://github.com/huggingface/transformers) | [Granite Speech 5.0 470M TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | ~1.4 GB | 1,949x | 3.53% |
| [NeMo](https://github.com/NVIDIA/NeMo) | [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | ~2.4 GB | 729x | 3.34% |
| [NeMo](https://github.com/NVIDIA/NeMo) | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | ~2.4 GB | 703x | 3.66% |
| [Transformers](https://github.com/huggingface/transformers) | [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | ~5.5 GB | 14.2x | 2.23% |
| [Transformers](https://github.com/huggingface/transformers) | [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | ~3.2 GB | 14.1x | 2.73% |
| [NeMo](https://github.com/NVIDIA/NeMo) | [Canary-Qwen 2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) | ~5.8 GB | 12.3x | 2.92% |

> All models were run in ```bfloat16``` on an RTX 4090 with Elegant Transcriber's default settings for each model.<br>
> All VRAM measurements include model weights and inference overhead and subtract background usage.<br>
> WER (word error rate) was measured on 15 long English recordings totaling 1 hour 45 minutes; lower is better.

## Special Thanks
* NVIDIA for the Parakeet and Canary-Qwen models. Parakeet is hands down the best balance of accuracy and compute time for most people IMHO.
* IBM for the Granite Speech models, including Granite Speech 5.0 470M TurboCTC, the fastest model in the program.
* Alibaba's Qwen team for the Qwen3-ASR models, the most accurate in the program, covering 30 languages.
* OpenAI for the older Whisper models setting the gold standard for so many years.
