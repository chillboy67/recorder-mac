"""
Provenance — an auditable trail of every transformation applied to the audio.

The fidelity rule ("never rewrite the speaker's words") is only credible if we
can *prove* which audio a transcript came from. Every pipeline step therefore
leaves a trace in `<output_dir>/meta.json`:

    {
      "original": {"file": "original.wav", "sha256": "…",
                    "sample_rate": 48000, "channels": 1, "duration_sec": 12.3},
      "steps": [
        {"name": "normalize", "params": {...},
         "input_sha256": "…", "output_sha256": "…", "timestamp": "…"},
        {"name": "denoise", "noise_floor_db": -38.2, "applied": true, ...},
        ...
      ]
    }

Pure stdlib (+ soundfile, already a dependency) — no new requirements.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

_CHUNK = 1 << 20   # 1 MiB streaming reads


def sha256_file(path: str | Path) -> str:
    """Streamed SHA-256 of a file without loading it into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def probe_wav(path: str | Path) -> dict:
    """Sample rate / channels / duration via soundfile (best effort)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_sec": round(float(info.duration), 3),
        }
    except Exception:
        return {}


def _meta_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "meta.json"


def _load(output_dir: str | Path) -> dict:
    p = _meta_path(output_dir)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"steps": []}


def _write(output_dir: str | Path, data: dict) -> None:
    p = _meta_path(output_dir)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)


def append_meta(output_dir: str | Path, step: dict) -> dict:
    """Append one processing step to meta.json; return the full meta dict.

    ``output_dir`` must already exist — this does not create it (unlike
    ``archive_input``). The engine relies on that ordering: archive_input runs
    as step 0, so every later append_meta call finds the directory in place.
    A corrupt or non-dict meta.json is discarded rather than propagated.
    """
    data = _load(output_dir)
    step = dict(step)
    step.setdefault("timestamp",
                    datetime.now(timezone.utc).isoformat(timespec="seconds"))
    data.setdefault("steps", []).append(step)
    _write(output_dir, data)
    return data


def archive_input(input_path: str | Path, output_dir: str | Path) -> Path:
    """Copy the input into the output dir as ``original.wav`` and record it.

    The archived copy is what the transcript is verifiably derived from: the
    caller's file (often a GUI temp) may be deleted once processing succeeds,
    so the output directory stays self-contained evidence. The original is
    never modified — every downstream step works on derived copies.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "original.wav"
    shutil.copy2(input_path, dest)
    data = _load(output_dir)
    data["original"] = {"file": dest.name, "sha256": sha256_file(dest),
                        **probe_wav(dest)}
    _write(output_dir, data)
    return dest
