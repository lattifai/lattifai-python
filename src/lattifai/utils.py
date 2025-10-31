"""Shared utility helpers for the LattifAI SDK."""

import os
from pathlib import Path
from typing import Any, Type

from lattifai.errors import ModelLoadError
from lattifai.tokenizer import LatticeTokenizer
from lattifai.workers import Lattice1AlphaWorker


def _resolve_model_path(model_name_or_path: str) -> str:
    """Resolve model path, downloading from Hugging Face when necessary."""
    if Path(model_name_or_path).exists():
        return model_name_or_path

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo_id=model_name_or_path, repo_type='model')
    except LocalEntryNotFoundError:
        try:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            return snapshot_download(repo_id=model_name_or_path, repo_type='model')
        except Exception as e:  # pragma: no cover - bubble up for caller context
            raise ModelLoadError(model_name_or_path, original_error=e)
    except Exception as e:  # pragma: no cover - unexpected download issue
        raise ModelLoadError(model_name_or_path, original_error=e)


def _select_device(device: str | None) -> str:
    """Select best available torch device when not explicitly provided."""
    if device:
        return device

    import torch

    detected = 'cpu'
    if torch.backends.mps.is_available():
        detected = 'mps'
    elif torch.cuda.is_available():
        detected = 'cuda'
    return detected


def _load_tokenizer(
    client_wrapper: Any,
    model_path: str,
    device: str,
    *,
    tokenizer_cls: Type[LatticeTokenizer] = LatticeTokenizer,
) -> LatticeTokenizer:
    """Instantiate tokenizer with consistent error handling."""
    try:
        return tokenizer_cls.from_pretrained(
            client_wrapper=client_wrapper,
            model_path=model_path,
            device=device,
        )
    except Exception as e:
        raise ModelLoadError(f'tokenizer from {model_path}', original_error=e)


def _load_worker(model_path: str, device: str) -> Lattice1AlphaWorker:
    """Instantiate lattice worker with consistent error handling."""
    try:
        return Lattice1AlphaWorker(model_path, device=device, num_threads=8)
    except Exception as e:
        raise ModelLoadError(f'worker from {model_path}', original_error=e)
