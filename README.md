<div align="center">

<img width="1312" height="508" alt="image" src="https://github.com/user-attachments/assets/3ae59567-e2dc-48de-8e7f-b9a5faed82f7" />

</div>

<div align="center">

### Requires [Python 3.10](https://www.python.org/downloads/release/python-31011/), [3.11](https://www.python.org/downloads/release/python-3119/), [3.12](https://www.python.org/downloads/release/python-31210/), or [3.13](https://www.python.org/downloads/release/python-31311/)

</div>

## Features

- Batch transcribe multiple files recursivelyk in directories or sub-directories.
- NVIDIA Parakeet TDT 0.6B v2 (English) and v3 (Multilingual + Translation)
- Optional timestamps with configurable segment intervals
- GPU (CUDA) and CPU inference support
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
