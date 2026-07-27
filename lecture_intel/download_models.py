#!/usr/bin/env python3
"""
Download all Whisper models Recorder offers — into a local folder, via curl.

Why curl (not huggingface_hub)? On some networks (notably mainland China) the
huggingface_hub transfer layer (xet / LFS negotiation) stalls at 0 bytes, while
a plain HTTPS GET to the resolve URL streams fine. So we fetch each repo's files
directly with curl + resume, through the hf-mirror.com mirror by default.

Models land in:  <this dir>/models/whisper-<name>-mlx/
The app loads them from there automatically (fully offline, no HF call at runtime).

Usage:
    python download_models.py                 # all models, via hf-mirror.com
    python download_models.py --hf            # use huggingface.co instead
    python download_models.py small medium     # only these
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

# Shared install location so both the dev tree and the installed Recorder.app
# find the models (the transcriber checks here first).
MODELS_DIR = Path.home() / "Library" / "Application Support" / "Recorder" / "models"

# UI model name → HF repo
REPOS = {
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "medium":         "mlx-community/whisper-medium-mlx",
    "small":          "mlx-community/whisper-small-mlx",
}

SKIP = {".gitattributes", "README.md"}


def endpoint(use_hf: bool) -> str:
    return "https://huggingface.co" if use_hf else "https://hf-mirror.com"


def list_files(repo: str) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.load(r)
    return [f["rfilename"] for f in data.get("siblings", [])
            if f["rfilename"] not in SKIP]


def curl_download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -C - resumes a partial file; --retry handles transient drops
    cmd = ["curl", "-L", "--fail", "-C", "-", "--retry", "8",
           "--retry-delay", "5", "--retry-all-errors",
           "-o", str(dest), url]
    print(f"    → {url}", flush=True)
    return subprocess.run(cmd).returncode == 0


def fetch_model(name: str, use_hf: bool) -> bool:
    repo = REPOS[name]
    base = endpoint(use_hf)
    out_dir = MODELS_DIR / f"whisper-{name}-mlx"
    print(f"\n=== {name}  ({repo}) → {out_dir} ===", flush=True)
    try:
        files = list_files(repo)
    except Exception as e:
        print(f"  ! cannot list files ({e}); using defaults")
        files = ["config.json", "weights.npz"]   # mlx whisper fallback
    ok = True
    for fn in files:
        dest = out_dir / fn
        url = f"{base}/{repo}/resolve/main/{fn}"
        for attempt in range(1, 4):
            if curl_download(url, dest) and dest.exists() and dest.stat().st_size > 0:
                break
            print(f"    retry {attempt} for {fn}", flush=True)
        else:
            print(f"  ✗ failed: {fn}", flush=True)
            ok = False
    if ok:
        size = sum(f.stat().st_size for f in out_dir.glob("*")) / 1e6
        print(f"  ✓ {name} ready ({size:.0f} MB)", flush=True)
    return ok


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    use_hf = "--hf" in sys.argv
    names = args or list(REPOS)
    results = {n: fetch_model(n, use_hf) for n in names if n in REPOS}
    print("\n=== summary ===", flush=True)
    for n, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {n}", flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
