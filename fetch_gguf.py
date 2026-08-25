"""Resumable direct download of one GGUF file into the LM Studio models tree.

`lms get` only takes a repo and picks the variant itself, which is no use when a
specific quantisation is the whole point — on a 16 GB card the difference between
IQ4_XS and Q4_K_M is the difference between fitting and not.

Usage:
    python fetch_gguf.py <hf_repo> <filename> [dest_root]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

DEFAULT_ROOT = Path.home() / ".lmstudio" / "models"
CHUNK = 1 << 20


def fetch(repo: str, filename: str, root: Path = DEFAULT_ROOT) -> Path:
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
    dest = root / repo.replace("/", "\\") / Path(filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    have = part.stat().st_size if part.exists() else 0
    if dest.exists():
        print(f"already present: {dest} ({dest.stat().st_size / 2**30:.2f} GB)")
        return dest

    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120) as r:
        if r.status_code not in (200, 206):
            raise SystemExit(f"HTTP {r.status_code} for {url}")
        total = int(r.headers.get("content-length", 0)) + have
        mode = "ab" if have and r.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        started = time.monotonic()
        last = 0.0
        with part.open(mode) as fh:
            for block in r.iter_bytes(CHUNK):
                fh.write(block)
                have += len(block)
                now = time.monotonic()
                if now - last > 15:
                    last = now
                    rate = have / (now - started) / 2**20
                    pct = 100 * have / total if total else 0
                    print(f"  {have/2**30:6.2f} / {total/2**30:6.2f} GB  {pct:5.1f}%  {rate:5.1f} MB/s",
                          flush=True)

    part.rename(dest)
    print(f"done: {dest} ({dest.stat().st_size / 2**30:.2f} GB)")
    return dest


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    root = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_ROOT
    for attempt in range(1, 30):
        try:
            fetch(sys.argv[1], sys.argv[2], root)
            break
        except (httpx.HTTPError, OSError) as exc:
            print(f"attempt {attempt} failed ({type(exc).__name__}: {exc}); resuming", flush=True)
            time.sleep(5)
