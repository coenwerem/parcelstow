"""Standard-library file-integrity helpers for retained experiment evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(chunk_size):
            digest.update(data)
    return digest.hexdigest()
