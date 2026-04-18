import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


MODELS = {
    "nvidia/parakeet-tdt-0.6b-v2": "parakeet-tdt-0.6b-v2.nemo",
    "nvidia/parakeet-tdt-0.6b-v3": "parakeet-tdt-0.6b-v3.nemo",
    "nvidia/canary-qwen-2.5b": None,
}


def get_models_dir() -> Path:
    return Path(__file__).parent / "models"


def get_local_model_path(repo_id: str):
    if repo_id not in MODELS:
        return None
    org, name = repo_id.split("/", 1)
    filename = MODELS[repo_id]
    if filename:
        return get_models_dir() / org / name / filename
    return get_models_dir() / org / name


def find_local_model(repo_id: str):
    path = get_local_model_path(repo_id)
    if path is None:
        return None
    filename = MODELS.get(repo_id)
    if filename:
        if path.is_file():
            return str(path)
    else:
        if path.is_dir() and (path / "config.json").is_file():
            return str(path)
    return None


def download_model(repo_id: str) -> str:
    if repo_id not in MODELS:
        raise ValueError(f"Unknown model: {repo_id}")

    filename = MODELS[repo_id]
    local_path = get_local_model_path(repo_id)

    if filename:
        if local_path.is_file():
            size_mb = local_path.stat().st_size / 1024 / 1024
            print(f"Model already exists: {local_path} ({size_mb:.1f} MB)")
            return str(local_path)

        local_dir = local_path.parent
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {repo_id}/{filename} to {local_dir}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )

        if local_path.is_file():
            size_mb = local_path.stat().st_size / 1024 / 1024
            print(f"Model cached at: {local_path} ({size_mb:.1f} MB)")
            return str(local_path)
        raise RuntimeError(f"Download completed but file not found at {local_path}")

    if local_path.is_dir() and (local_path / "config.json").is_file():
        print(f"Model already exists: {local_path}")
        return str(local_path)

    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} to {local_path}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_path),
    )

    if (local_path / "config.json").is_file():
        print(f"Model cached at: {local_path}")
        return str(local_path)
    raise RuntimeError(f"Download completed but config.json not found at {local_path}")


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "nvidia/parakeet-tdt-0.6b-v2"
    try:
        path = download_model(model_id)
        print(f"Success: {path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
