# Elegant Audio Transcriber — Server API Guide

## Overview

Elegant Audio Transcriber can run as an HTTP server, allowing any program or script to send audio and receive transcriptions back. This means you can integrate transcription into your own Python scripts, web applications, or automation workflows without using the GUI.

## Starting the Server

1. Launch the Elegant Audio Transcriber GUI.
2. Click the **Server** button in the bottom control bar.
3. A dialog will ask you to choose a port (default: **7862**). Click **OK**.
4. The GUI will show "Server running on port 7862" and all other controls will be disabled.
5. The server is now accepting requests.

To stop the server, click **Stop Server**.

> **Tip:** The server binds to `0.0.0.0`, which means it is accessible from your local machine at `http://127.0.0.1:7862` and from other computers on your network at `http://<your-ip>:7862`.

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check if the server is running |
| `/status` | GET | Server status, queue depth, whether a transcription is active |
| `/models` | GET | List all available models and their properties |
| `/transcribe` | POST | Transcribe audio from a file upload (multipart form) |
| `/transcribe/raw` | POST | Transcribe audio from base64-encoded data (JSON body) |

The server also provides interactive API documentation at:
- **Swagger UI:** `http://127.0.0.1:7862/docs`
- **ReDoc:** `http://127.0.0.1:7862/redoc`

---

## Quick Start

The simplest possible request — send an audio file and get text back:

```python
import requests

response = requests.post(
    "http://127.0.0.1:7862/transcribe",
    files={"audio": open("my_audio.mp3", "rb")},
)

print(response.json()["text"])
```

That's it. The server uses its default settings (whatever was configured in the GUI when the server was started) for everything you don't specify.

---

## Accepted Audio Input Formats

The server is designed to be very flexible. You can send audio in any of these formats:

### 1. Audio Files (most common)

Standard audio files uploaded directly. Supported formats include: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.webm`, `.mp4`, `.mkv`, `.avi`, `.asf`, `.amr`.

```python
import requests

with open("recording.wav", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:7862/transcribe",
        files={"audio": ("recording.wav", f, "audio/wav")},
    )

print(response.json()["text"])
```

### 2. NumPy Arrays

If your program already has audio loaded as a NumPy array, you can send it directly without saving to disk first. Save the array with `np.save()` and upload the `.npy` file.

```python
import io
import numpy as np
import requests

# Your audio as a numpy array (float32, mono, at some sample rate)
audio_array = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds at 16kHz

# Serialize to .npy format
buffer = io.BytesIO()
np.save(buffer, audio_array)
buffer.seek(0)

response = requests.post(
    "http://127.0.0.1:7862/transcribe",
    files={"audio": ("audio.npy", buffer, "application/octet-stream")},
    data={"sample_rate": "16000"},  # tell the server what sample rate your array is
)

print(response.json()["text"])
```

> **Note:** If your audio is at a different sample rate (e.g., 44100 Hz), pass `sample_rate=44100` and the server will resample it automatically.

### 3. PyTorch Tensors

If you're working with PyTorch and have audio as a tensor:

```python
import io
import torch
import requests

# Your audio as a PyTorch tensor
audio_tensor = torch.randn(16000 * 5)  # 5 seconds at 16kHz

# Serialize to .pt format
buffer = io.BytesIO()
torch.save(audio_tensor, buffer)
buffer.seek(0)

response = requests.post(
    "http://127.0.0.1:7862/transcribe",
    files={"audio": ("audio.pt", buffer, "application/octet-stream")},
    data={"sample_rate": "16000"},
)

print(response.json()["text"])
```

### 4. Raw PCM Bytes

If you have raw audio samples as bytes (common in real-time audio pipelines):

```python
import numpy as np
import requests

# Raw PCM audio data
audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
raw_bytes = audio.tobytes()

response = requests.post(
    "http://127.0.0.1:7862/transcribe",
    files={"audio": ("audio.raw", raw_bytes, "application/octet-stream")},
    data={
        "audio_format": "pcm",
        "sample_rate": "16000",
        "dtype": "float32",  # also supports: int16, int32, float64
    },
)

print(response.json()["text"])
```

### 5. Base64-Encoded Data (JSON Endpoint)

For situations where you want to send everything as JSON (no multipart form):

```python
import base64
import io
import numpy as np
import requests

# Encode your audio as base64
audio = np.random.randn(16000 * 5).astype(np.float32)
buffer = io.BytesIO()
np.save(buffer, audio)
b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

response = requests.post(
    "http://127.0.0.1:7862/transcribe/raw",
    json={
        "audio_data": b64_data,
        "audio_format": "numpy",
        "sample_rate": 16000,
    },
)

print(response.json()["text"])
```

You can also send a complete audio file as base64:

```python
import base64
import requests

with open("my_audio.mp3", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://127.0.0.1:7862/transcribe/raw",
    json={
        "audio_data": b64_data,
        "audio_format": "file",
    },
)

print(response.json()["text"])
```

---

## Settings You Can Control

Every setting is optional. If you don't specify a value, the server uses whatever was configured in the GUI when the server was started.

| Parameter | Type | Description | Example Values |
|---|---|---|---|
| `model` | string | Which model to use | `"Parakeet TDT 0.6B v2"`, `"Parakeet TDT 0.6B v3"`, `"Canary-Qwen 2.5B"` |
| `precision` | string | Numerical precision | `"bfloat16"`, `"float16"`, `"float32"` |
| `device` | string | CPU or GPU | `"cuda"`, `"cpu"` |
| `output_format` | string | Timestamp output format | `"txt"`, `"srt"`, `"vtt"`, `"json"` |
| `word_timestamps` | boolean | Enable word-level timestamps | `"true"`, `"false"` |
| `segment_length` | integer | Audio chunk size in seconds | `"30"`, `"80"` |
| `segment_duration` | integer | Timestamp grouping interval in seconds | `"5"`, `"10"` |
| `audio_format` | string | Override auto-detection of input format | `"auto"`, `"file"`, `"numpy"`, `"tensor"`, `"pcm"` |
| `sample_rate` | integer | Sample rate of raw audio input | `"16000"`, `"44100"` |
| `dtype` | string | Data type for raw PCM input | `"float32"`, `"int16"` |

### Example with Custom Settings

```python
import requests

with open("lecture.mp3", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:7862/transcribe",
        files={"audio": ("lecture.mp3", f, "audio/mpeg")},
        data={
            "model": "Parakeet TDT 0.6B v2",
            "precision": "float16",
            "device": "cuda",
            "word_timestamps": "true",
            "output_format": "srt",
            "segment_length": "80",
            "segment_duration": "10",
        },
    )

result = response.json()
print(result["text"])
print(f"Took {result['processing_time_seconds']} seconds")
```

---

## Response Format

Every transcription request returns a JSON object with these fields:

```json
{
    "text": "The full transcription text...",
    "segments": [
        {"start": 0.0, "end": 3.52, "text": "The full"},
        {"start": 3.52, "end": 7.1, "text": "transcription text"}
    ],
    "processing_time_seconds": 1.234,
    "model_used": "Parakeet TDT 0.6B v2 - bfloat16",
    "audio_duration_seconds": 10.5
}
```

| Field | Description |
|---|---|
| `text` | The complete transcription as a single string. Always present. |
| `segments` | Timestamped segments. Only populated when `word_timestamps` is enabled and the model supports timestamps. Empty list (`[]`) otherwise. |
| `processing_time_seconds` | How long the transcription took (excludes network transfer time). |
| `model_used` | Which model and precision was used (e.g., `"Parakeet TDT 0.6B v2 - bfloat16"`). |
| `audio_duration_seconds` | Duration of the audio that was processed, in seconds. |

---

## Model Notes

| Model | Timestamps | Speed | Accuracy | VRAM |
|---|---|---|---|---|
| Parakeet TDT 0.6B v2 | Yes | Fast | High (English) | ~1.5 GB |
| Parakeet TDT 0.6B v3 | Yes | Fast | Slightly lower English, multilingual | ~1.5 GB |
| Canary-Qwen 2.5B | No | ~30x slower | Highest (English) | ~11 GB |

- **Canary-Qwen** does not support timestamps. If you request `word_timestamps=true` with Canary, the `segments` field will be empty and `text` will still contain the full transcription.
- **Canary-Qwen** has a maximum audio chunk length of 40 seconds (enforced automatically).

---

## Checking Server Status

### Health Check

```python
import requests

r = requests.get("http://127.0.0.1:7862/health")
print(r.json())
# {"status": "ok"}
```

### Server Status

```python
r = requests.get("http://127.0.0.1:7862/status")
print(r.json())
# {"server_running": true, "queue_depth": 0, "transcription_active": false}
```

- `queue_depth`: Number of requests waiting to be processed.
- `transcription_active`: Whether a transcription is currently running.

### List Available Models

```python
r = requests.get("http://127.0.0.1:7862/models")
for name, info in r.json().items():
    print(f"{name}: type={info['model_type']}, vram={info['avg_vram_usage']}, timestamps={info['supports_timestamps']}")
```

---

## Request Queuing

The server processes one transcription at a time (because the GPU is a shared resource). If you send multiple requests simultaneously, they are placed in a queue and processed in the order they were received. Each client waits for its own result — you don't need to poll.

```python
import threading
import requests

def transcribe(file_path):
    with open(file_path, "rb") as f:
        r = requests.post(
            "http://127.0.0.1:7862/transcribe",
            files={"audio": (file_path, f)},
        )
    print(f"{file_path}: {r.json()['text'][:80]}...")

# Send 3 requests at the same time — they'll be queued and processed one by one
threads = [
    threading.Thread(target=transcribe, args=("file1.mp3",)),
    threading.Thread(target=transcribe, args=("file2.mp3",)),
    threading.Thread(target=transcribe, args=("file3.mp3",)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## Using curl (Command Line)

If you prefer the command line:

```bash
# Health check
curl http://127.0.0.1:7862/health

# Transcribe a file
curl -F "audio=@my_audio.mp3" http://127.0.0.1:7862/transcribe

# Transcribe with settings
curl -F "audio=@my_audio.mp3" \
     -F "model=Parakeet TDT 0.6B v2" \
     -F "precision=float16" \
     -F "word_timestamps=true" \
     http://127.0.0.1:7862/transcribe
```

---

## Error Handling

The server returns standard HTTP status codes:

| Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Bad request (invalid model name, empty audio, unreadable format) |
| 422 | Validation error (missing required `audio` field) |
| 500 | Internal server error (model failed during transcription) |
| 503 | Server shutting down (request was in queue when server stopped) |

Errors include a `detail` field explaining what went wrong:

```python
r = requests.post(
    "http://127.0.0.1:7862/transcribe",
    files={"audio": open("test.mp3", "rb")},
    data={"model": "Nonexistent Model"},
)

if r.status_code != 200:
    print(f"Error {r.status_code}: {r.json()['detail']}")
    # Error 400: Unknown model: 'Nonexistent Model' with precision 'bfloat16'. Available: [...]
```

---

## Complete Example Script

A full script that transcribes all `.mp3` files in a folder:

```python
import requests
from pathlib import Path

SERVER = "http://127.0.0.1:7862"
AUDIO_DIR = Path("./my_audio_files")

# Check server is running
health = requests.get(f"{SERVER}/health")
if health.status_code != 200:
    print("Server is not running!")
    exit(1)

# Process each file
for audio_file in sorted(AUDIO_DIR.glob("*.mp3")):
    print(f"Transcribing: {audio_file.name}...", end=" ", flush=True)

    with open(audio_file, "rb") as f:
        response = requests.post(
            f"{SERVER}/transcribe",
            files={"audio": (audio_file.name, f, "audio/mpeg")},
        )

    if response.status_code == 200:
        result = response.json()
        text = result["text"]
        duration = result["audio_duration_seconds"]
        speed = result["processing_time_seconds"]

        # Save transcript
        output_file = audio_file.with_suffix(".txt")
        output_file.write_text(text, encoding="utf-8")
        print(f"Done ({duration:.1f}s audio in {speed:.1f}s)")
    else:
        print(f"Failed: {response.json().get('detail', 'Unknown error')}")
```
