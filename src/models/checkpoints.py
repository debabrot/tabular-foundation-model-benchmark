"""Resolve pinned model weights and preserve filenames for checkpoint loaders."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]


class Unavailable(RuntimeError):
    """A prerequisite prevents inference; this is not a successful evaluation."""


def checkpoint(key: str, download: bool, record: dict) -> Path:
    from httpx import HTTPError
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    spec = json.loads((ROOT / "checkpoints/manifest.json").read_text())[key]
    record["checkpoint"] = {**spec, "sha256": {}}
    paths = []
    for filename in spec["files"]:
        try:
            path = Path(hf_hub_download(
                repo_id=spec["repo_id"], revision=spec["revision"], filename=filename,
                cache_dir=ROOT / ".cache/huggingface", local_files_only=not download,
            ))
        except LocalEntryNotFoundError as exc:
            raise Unavailable("Checkpoint is not cached. Re-run with --download.") from exc
        except (HfHubHTTPError, HTTPError, OSError) as exc:
            # Do not copy HTTP headers, credentials, or full URLs into the report.
            raise Unavailable(
                f"Checkpoint download failed ({type(exc).__name__}). Check network "
                "access and the model's access requirements in docs/model-runtime.md."
            ) from exc
        with path.open("rb") as stream:
            record["checkpoint"]["sha256"][filename] = hashlib.file_digest(stream, "sha256").hexdigest()
        # TabPFN resolves symlinks before suffix-based format detection. Hub snapshot
        # symlinks resolve to extensionless blobs, which would select torch.load.
        # Materialize a regular file with the original name without altering weights.
        destination = ROOT / ".cache/checkpoints" / key / spec["revision"] / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            staged = Path(directory) / filename
            try:
                os.link(path.resolve(), staged)
            except OSError:
                shutil.copyfile(path, staged)
            staged.replace(destination)
        paths.append(destination)
    return paths[0] if len(paths) == 1 else paths[0].parent

