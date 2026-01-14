"""Shared utility helpers for the LattifAI SDK."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from lattifai.errors import ModelLoadError


def safe_print(text: str, **kwargs) -> None:
    """
    Safely print text with Unicode characters, handling Windows encoding issues.

    On Windows, the default console encoding (cp1252) can't handle many Unicode
    characters like emojis. This function ensures text is printed correctly by
    using UTF-8 encoding when necessary.

    Args:
        text: The text to print, may contain Unicode/emoji characters
        **kwargs: Additional arguments passed to print()
    """
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        # On Windows, try to reconfigure stdout to use UTF-8
        if sys.platform == "win32":
            try:
                # Try to encode with UTF-8 and print
                if hasattr(sys.stdout, "buffer"):
                    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
                    sys.stdout.flush()
                else:
                    # Fallback: replace problematic characters
                    print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding), **kwargs)
            except Exception:
                # Last resort: remove emojis
                import re

                text_no_emoji = re.sub(r"[^\x00-\x7F\u4e00-\u9fff]+", "", text)
                print(text_no_emoji, **kwargs)
        else:
            # Non-Windows: this shouldn't happen, but fallback gracefully
            print(text.encode("utf-8", errors="replace").decode("utf-8"), **kwargs)


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
        if datetime.now() - marker_date > timedelta(days=7):
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


def _resolve_model_path(model_name_or_path: str, model_hub: str = "huggingface") -> str:
    """Resolve model path, downloading from the specified model hub when necessary.

    Args:
        model_name_or_path: Local path or remote model identifier.
        model_hub: Which hub to use for downloads. Supported: "huggingface", "modelscope".
    """
    if Path(model_name_or_path).expanduser().exists():
        return str(Path(model_name_or_path).expanduser())

    # Normalize hub name
    hub = (model_hub or "huggingface").lower()

    if hub not in ("huggingface", "modelscope"):
        raise ValueError(f"Unsupported model_hub: {model_hub}. Supported: 'huggingface', 'modelscope'.")

    # If local path exists, return it regardless of hub
    if Path(model_name_or_path).expanduser().exists():
        return str(Path(model_name_or_path).expanduser())

    if hub == "huggingface":
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE
        from huggingface_hub.errors import LocalEntryNotFoundError

        # Support repo_id@revision syntax
        hf_repo_id = model_name_or_path
        revision = None
        if "@" in model_name_or_path:
            hf_repo_id, revision = model_name_or_path.split("@", 1)

        # Determine cache directory for this model
        cache_dir = Path(HF_HUB_CACHE) / f'models--{hf_repo_id.replace("/", "--")}'

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

        # If no specific revision/commit is provided, try to fetch the real latest SHA
        # to bypass Hugging Face's model_info (metadata) sync lag.
        if not revision:
            try:
                api = HfApi()
                refs = api.list_repo_refs(repo_id=hf_repo_id, repo_type="model")
                # Look for the default branch (usually 'main')
                for branch in refs.branches:
                    if branch.name == "main":
                        revision = branch.target_commit
                        break
            except Exception:
                # Fallback to default behavior if API call fails
                revision = None

        try:
            downloaded_path = snapshot_download(repo_id=hf_repo_id, repo_type="model", revision=revision)
            _create_cache_marker(cache_dir)
            return downloaded_path
        except LocalEntryNotFoundError:
            # Fall back to modelscope if HF entry not found
            try:
                from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot

                downloaded_path = ms_snapshot(model_name_or_path)
                return downloaded_path
            except Exception as e:  # pragma: no cover - bubble up for caller context
                raise ModelLoadError(model_name_or_path, original_error=e)
        except Exception as e:  # pragma: no cover - unexpected download issue
            import colorful

            print(colorful.red | f"Error downloading from Hugging Face Hub: {e}. Trying ModelScope...")
            from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot

            downloaded_path = ms_snapshot(model_name_or_path)
            return downloaded_path

    # modelscope path
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot

    # Determine cache directory for ModelScope
    # ModelScope uses ~/.cache/modelscope/hub/models/{org}/{model} structure
    modelscope_cache = Path.home() / ".cache" / "modelscope" / "hub" / "models"
    cache_dir = modelscope_cache / model_name_or_path

    # Check if we have a valid cached version
    if _is_cache_valid(cache_dir):
        # Return the cached path directly
        if cache_dir.exists():
            return str(cache_dir)

    try:
        downloaded_path = ms_snapshot(model_name_or_path)
        # Create cache marker after successful download
        if downloaded_path:
            actual_cache_dir = Path(downloaded_path)
            _create_cache_marker(actual_cache_dir)
        return downloaded_path
    except Exception as e:  # pragma: no cover
        raise ModelLoadError(model_name_or_path, original_error=e)


def _select_device(device: Optional[str]) -> str:
    """Select best available torch device when not explicitly provided."""
    if device and device != "auto":
        return device

    import torch

    detected = "cpu"
    if torch.backends.mps.is_available():
        detected = "mps"
    elif torch.cuda.is_available():
        detected = "cuda"
    return detected
