#!/usr/bin/env python3
"""
Download Whisper models for MLX or faster-whisper into separate local caches.

Usage:
    python download_models.py                     # MLX on Apple Silicon, CPU elsewhere
    python download_models.py --engine cpu small  # portable CTranslate2 model
    python download_models.py --engine mlx --hf   # Apple Silicon, Hugging Face directly
    python download_models.py --engine all small  # both formats for one model
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from core.model_cache import (faster_whisper_cache_dir, is_faster_whisper_model,
                              mlx_cache_dir, whisper_cpp_model_available,
                              whisper_cpp_model_dir)

# MLX and faster-whisper use different model formats and separate cache folders.
MODELS_DIR = mlx_cache_dir()

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


def _set_hf_endpoint(use_hf: bool) -> None:
    if use_hf:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def fetch_cpu_model(name: str, use_hf: bool = False) -> bool:
    """Pre-download a CTranslate2 model into faster-whisper's isolated cache."""
    _set_hf_endpoint(use_hf)
    from faster_whisper.utils import download_model

    print(f"\n=== {name} (faster-whisper / CPU) ===", flush=True)
    try:
        path = Path(download_model(name, cache_dir=str(faster_whisper_cache_dir())))
    except Exception as exc:
        print(f"  ✗ download failed: {exc}", flush=True)
        return False
    if not is_faster_whisper_model(path):
        print(f"  ✗ incomplete model files: {path}", flush=True)
        return False
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 1e6
    print(f"  ✓ {name} ready ({size:.0f} MB) at {path}", flush=True)
    return True


def fetch_whisper_cpp_model(name: str, backend: str, use_hf: bool = False) -> bool:
    """Fetch ggml model files for Vulkan or the Intel OpenVINO encoder."""
    _set_hf_endpoint(use_hf)
    from huggingface_hub import hf_hub_download

    out_dir = whisper_cpp_model_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if backend == "vulkan":
            hf_hub_download(
                repo_id="ggerganov/whisper.cpp",
                filename=f"ggml-{name}.bin",
                local_dir=str(out_dir),
            )
        elif backend == "openvino":
            archive = hf_hub_download(
                repo_id="Intel/whisper.cpp-openvino-models",
                filename=f"ggml-{name}-models.zip",
                local_dir=str(out_dir),
            )
            wanted = {
                f"ggml-{name}.bin",
                f"ggml-{name}-encoder-openvino.xml",
                f"ggml-{name}-encoder-openvino.bin",
            }
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if Path(member.filename).name in wanted and not member.is_dir():
                        destination = out_dir / Path(member.filename).name
                        with bundle.open(member) as source, destination.open("wb") as target:
                            shutil.copyfileobj(source, target)
        else:
            raise ValueError(f"Unknown whisper.cpp backend: {backend}")
    except Exception as exc:
        print(f"  ✗ download failed: {exc}", flush=True)
        return False

    if not whisper_cpp_model_available(name, backend):
        print(f"  ✗ incomplete {backend} model files in {out_dir}", flush=True)
        return False
    print(f"  ✓ {name} ready for whisper.cpp {backend} at {out_dir}", flush=True)
    return True


def _default_engine() -> str:
    if (platform.system() == "Darwin"
            and platform.machine().lower() in ("arm64", "aarch64")):
        return "mlx"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Recorder Whisper models")
    parser.add_argument("--engine", choices=("auto", "mlx", "cpu", "cpp-vulkan", "cpp-openvino", "all"),
                        default="auto", help="model format/backend (default: platform-appropriate)")
    parser.add_argument("--hf", action="store_true",
                        help="use huggingface.co instead of hf-mirror.com for model downloads")
    parser.add_argument("models", nargs="*", help="model names (default: small for CPU, all for MLX)")
    args = parser.parse_args(argv)

    if args.engine == "auto":
        engines = ("mlx", "cpu") if _default_engine() == "mlx" else ("cpu",)
    elif args.engine == "all":
        engines = ("mlx", "cpu")
    else:
        engines = (args.engine,)
    names_by_engine = {
        backend: (args.models or (list(REPOS) if backend == "mlx" else ["small"]))
        for backend in engines
    }
    unknown = sorted({name for names in names_by_engine.values() for name in names}
                     - set(REPOS))
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")

    results: dict[str, bool] = {}
    for backend in engines:
        if backend == "mlx" and not _default_engine() == "mlx":
            print("MLX models require Apple Silicon macOS; skipping MLX download.", flush=True)
            continue
        for name in names_by_engine[backend]:
            if backend == "mlx":
                ok = fetch_model(name, args.hf)
            elif backend == "cpu":
                ok = fetch_cpu_model(name, args.hf)
            else:
                cpp_backend = backend.removeprefix("cpp-")
                ok = fetch_whisper_cpp_model(name, cpp_backend, args.hf)
            results[f"{backend}:{name}"] = ok

    print("\n=== summary ===", flush=True)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}", flush=True)
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
