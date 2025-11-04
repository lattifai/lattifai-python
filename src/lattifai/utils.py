"""Shared utility helpers for the LattifAI SDK."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Type

from lattifai.errors import ModelLoadError
from lattifai.tokenizer import LatticeTokenizer
from lattifai.workers import Lattice1AlphaWorker


def _get_cache_marker_path(cache_dir: Path) -> Path:
    """Get the path for the cache marker file with current date."""
    today = datetime.now().strftime("%Y%m%d")
    return cache_dir / f".done{today}"


def _is_cache_valid(cache_dir: Path) -> bool:
    """Check if cached model is valid (exists and not older than 1 days)."""
    if not cache_dir.exists():
        return False

    # Find any .done* marker files
    marker_files = list(cache_dir.glob(".done*"))
    if not marker_files:
        return False

    # Get the most recent marker file
    latest_marker = max(marker_files, key=lambda p: p.stat().st_mtime)

    # Extract date from marker filename (format: .doneYYYYMMDD)
    try:
        date_str = latest_marker.name.replace(".done", "")
        marker_date = datetime.strptime(date_str, "%Y%m%d")
        # Check if marker is older than 1 days
        if datetime.now() - marker_date > timedelta(days=1):
            return False
        return True
    except (ValueError, IndexError):
        # Invalid marker file format, treat as invalid cache
        return False


def _create_cache_marker(cache_dir: Path) -> None:
    """Create a cache marker file with current date and clean old markers."""
    # Remove old marker files
    for old_marker in cache_dir.glob(".done*"):
        old_marker.unlink(missing_ok=True)

    # Create new marker file
    marker_path = _get_cache_marker_path(cache_dir)
    marker_path.touch()


def _resolve_model_path(model_name_or_path: str) -> str:
    """Resolve model path, downloading from Hugging Face when necessary."""
    if Path(model_name_or_path).exists():
        return model_name_or_path

    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.errors import LocalEntryNotFoundError

    # Determine cache directory for this model
    cache_dir = Path(HF_HUB_CACHE) / f'models--{model_name_or_path.replace("/", "--")}'

    # Check if we have a valid cached version
    if _is_cache_valid(cache_dir):
        # Return the snapshot path (latest version)
        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.exists():
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if snapshot_dirs:
                # Return the most recent snapshot
                latest_snapshot = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)
                return str(latest_snapshot)

    try:
        downloaded_path = snapshot_download(repo_id=model_name_or_path, repo_type="model")
        _create_cache_marker(cache_dir)
        return downloaded_path
    except LocalEntryNotFoundError:
        try:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            downloaded_path = snapshot_download(repo_id=model_name_or_path, repo_type="model")
            _create_cache_marker(cache_dir)
            return downloaded_path
        except Exception as e:  # pragma: no cover - bubble up for caller context
            raise ModelLoadError(model_name_or_path, original_error=e)
    except Exception as e:  # pragma: no cover - unexpected download issue
        raise ModelLoadError(model_name_or_path, original_error=e)


def _select_device(device: Optional[str]) -> str:
    """Select best available torch device when not explicitly provided."""
    if device:
        return device

    import torch

    detected = "cpu"
    if torch.backends.mps.is_available():
        detected = "mps"
    elif torch.cuda.is_available():
        detected = "cuda"
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
        raise ModelLoadError(f"tokenizer from {model_path}", original_error=e)


def _load_worker(model_path: str, device: str) -> Lattice1AlphaWorker:
    """Instantiate lattice worker with consistent error handling."""
    try:
        return Lattice1AlphaWorker(model_path, device=device, num_threads=8)
    except Exception as e:
        raise ModelLoadError(f"worker from {model_path}", original_error=e)
