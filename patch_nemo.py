"""Patch NeMo toolkit for compatibility with both pre-v5 and v5+ transformers."""
import importlib.util
import sys
from pathlib import Path


def _find_nemo_root() -> Path:
    spec = importlib.util.find_spec("nemo")
    if spec and spec.submodule_search_locations:
        return Path(spec.submodule_search_locations[0])
    raise RuntimeError("Cannot locate installed nemo package")


NEMO_ROOT = _find_nemo_root()

PATCHES = []


def patch(file_path, old, new, description):
    PATCHES.append((file_path, old, new, description))


patch(
    NEMO_ROOT / "collections" / "common" / "tokenizers" / "huggingface" / "auto_tokenizer.py",
    "from typing import List, Optional\n",
    "import os\nfrom typing import List, Optional\n",
    "auto_tokenizer.py: add 'import os'",
)

patch(
    NEMO_ROOT / "collections" / "common" / "tokenizers" / "huggingface" / "auto_tokenizer.py",
    """            self.tokenizer = AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                vocab_file=vocab_file,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )
        else:""",
    """            self.tokenizer = AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                vocab_file=vocab_file,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )
            if vocab_file and os.path.isfile(vocab_file):
                try:
                    with open(vocab_file, 'r', encoding='utf-8') as f:
                        expected_vocab_size = sum(1 for line in f if line.strip())
                    if expected_vocab_size > 0 and len(self.tokenizer) != expected_vocab_size:
                        tokenizer_class = type(self.tokenizer)
                        self.tokenizer = tokenizer_class.from_pretrained(
                            pretrained_model_name_or_path=vocab_file,
                            use_fast=use_fast,
                        )
                        logging.info(
                            f"Loaded tokenizer from custom vocab_file with {len(self.tokenizer)} tokens "
                            f"(resolved class: {tokenizer_class.__name__})"
                        )
                except Exception:
                    pass
        else:""",
    "auto_tokenizer.py: handle transformers>=5.0 ignoring vocab_file kwarg",
)

patch(
    NEMO_ROOT / "collections" / "speechlm2" / "parts" / "hf_hub.py",
    """        force_download: bool,
        proxies: Optional[dict],
        resume_download: Optional[bool],
        local_files_only: bool,""",
    """        force_download: bool,
        local_files_only: bool,
        **kwargs,""",
    "hf_hub.py: absorb removed proxies/resume_download via **kwargs",
)

patch(
    NEMO_ROOT / "collections" / "speechlm2" / "parts" / "hf_hub.py",
    """            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,""",
    """            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,""",
    "hf_hub.py: remove proxies/resume_download from cached_file call",
)

patch(
    NEMO_ROOT / "collections" / "speechlm2" / "parts" / "hf_hub.py",
    """            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,""",
    """            force_download=force_download,
            local_files_only=local_files_only,""",
    "hf_hub.py: remove proxies/resume_download from super()._from_pretrained call",
)

patch(
    NEMO_ROOT / "collections" / "tts" / "models" / "magpietts.py",
    """            tc_tokenizer = self.tokenizer.tokenizers[self.text_conditioning_tokenizer_name]
            self.context_text_embedding = nn.Embedding(tc_tokenizer.vocab_size, cfg.embedding_dim)""",
    """            tc_tokenizer = self.tokenizer.tokenizers[self.text_conditioning_tokenizer_name]
            tc_vocab_size = tc_tokenizer.vocab_size
            if hasattr(tc_tokenizer, '_extra_ids'):
                tc_vocab_size -= tc_tokenizer._extra_ids
            self.context_text_embedding = nn.Embedding(tc_vocab_size, cfg.embedding_dim)""",
    "magpietts.py: adjust vocab_size for T5 fast tokenizer sentinel IDs",
)

patch(
    NEMO_ROOT / "collections" / "tts" / "models" / "magpietts_preference_optimization.py",
    "    inputs = inputs.to(device)",
    "    inputs = inputs.to(device=device, dtype=whisper_model.dtype)",
    "magpietts_preference_optimization.py: explicit dtype cast for whisper inputs",
)

patch(
    NEMO_ROOT / "core" / "classes" / "mixins" / "hf_io_mixin.py",
    """        model_filter = dict(
            author=None,
            library='nemo',
            language=None,
            model_name=None,
            task=None,
            tags=None,
            limit=None,
            full=None,
            cardData=False,
        )""",
    """        from packaging.version import Version as _V
        import huggingface_hub as _hfh
        if _V(_hfh.__version__) >= _V("0.24"):
            model_filter = dict(
                author=None,
                filter=['nemo'],
                model_name=None,
                limit=None,
                full=None,
                cardData=False,
            )
        else:
            model_filter = dict(
                author=None,
                library='nemo',
                language=None,
                model_name=None,
                task=None,
                tags=None,
                limit=None,
                full=None,
                cardData=False,
            )""",
    "hf_io_mixin.py: version-aware model filter API for old and new huggingface_hub",
)

patch(
    NEMO_ROOT / "core" / "classes" / "mixins" / "hf_io_mixin.py",
    """            # Make any modifications to the filter as necessary
            filt['language'] = [...]
            filt['task'] = ...
            filt['tags'] = [...]""",
    """            # Make any modifications to the filter as necessary
            if 'filter' in filt:
                filt['filter'].append('en')  # Add language filter
                filt['filter'].append('automatic-speech-recognition')  # Add task filter
            else:
                filt['language'] = [...]
                filt['task'] = ...
                filt['tags'] = [...]""",
    "hf_io_mixin.py: version-aware docstring example for filter API",
)


def apply_patches():
    applied = 0
    skipped = 0
    failed = 0

    for file_path, old, new, description in PATCHES:
        if not file_path.exists():
            print(f"  SKIP (file not found): {description}")
            print(f"        {file_path}")
            skipped += 1
            continue

        content = file_path.read_text(encoding="utf-8")

        if new in content:
            print(f"  SKIP (already patched): {description}")
            skipped += 1
            continue

        if old not in content:
            print(f"  FAIL (pattern not found): {description}")
            print(f"        {file_path}")
            failed += 1
            continue

        content = content.replace(old, new, 1)
        file_path.write_text(content, encoding="utf-8")
        print(f"  OK: {description}")
        applied += 1

    return applied, skipped, failed


if __name__ == "__main__":
    print(f"NeMo location: {NEMO_ROOT}")
    print(f"Patches to apply: {len(PATCHES)}")
    print()

    applied, skipped, failed = apply_patches()

    print()
    print(f"Results: {applied} applied, {skipped} skipped, {failed} failed")

    if failed > 0:
        print("\nSome patches failed — the installed NeMo version may differ from expected.")
        sys.exit(1)
    else:
        print("\nAll patches processed successfully.")
