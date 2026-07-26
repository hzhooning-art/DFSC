"""Fetch and checksum the AnomDiffDB trajectory files used by dfsc."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "750nm_mesh_size.mat": (
        "https://raw.githubusercontent.com/AnomDiffDB/DB/master/750nm_mesh_size.mat",
        "fd11f82f60df8235f311f34940ac9fc9226348221ca17cf80a02c3436401d4cd",
    ),
    "brownian_beads.mat": (
        "https://raw.githubusercontent.com/AnomDiffDB/DB/master/brownian_beads.mat",
        "1bc05922899172053126df80b28e43a645411d8b6919b1575a9a70ef4aa9d8e6",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    directory = ROOT / "data" / "external" / "anomdiffdb"
    directory.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_sha256) in FILES.items():
        target = directory / filename
        if not target.exists() or sha256(target) != expected_sha256:
            temporary = target.with_suffix(".mat.download")
            urllib.request.urlretrieve(url, temporary)
            if sha256(temporary) != expected_sha256:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"downloaded {filename} failed the SHA-256 check")
            temporary.replace(target)
        print(f"verified {target} sha256={expected_sha256}")


if __name__ == "__main__":
    main()
