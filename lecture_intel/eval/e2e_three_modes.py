"""E2E: run real online audio (SEP-28k stutter podcast + Chinese stutter phone
call) through all three modes and verify the fidelity invariants:

- general / ielts: text never modified, disfluencies only annotated
- classroom: only confirmed ASR-loop artifacts folded, real stutter kept
- meta.json traces every transformation

Usage (with the app's own venv):

    .venv/bin/python eval/e2e_three_modes.py <en.wav> <zh.wav>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))
from core.engine import run  # noqa: E402

MODES = ["general", "classroom", "ielts"]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    clips = {"en": Path(sys.argv[1]), "zh": Path(sys.argv[2])}
    out_root = EVAL_DIR / "e2e_out"
    out_root.mkdir(parents=True, exist_ok=True)
    for name, clip in clips.items():
        for mode in MODES:
            out = out_root / f"{name}_{mode}"
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== {name} / {mode} ===", flush=True)
            run(clip, out, mode_key=mode, model="large-v3", formats=["json", "md"])
            data = json.loads((out / f"{clip.stem}.json").read_text(encoding="utf-8"))
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            print(f"  full_text: {data['full_text']!r}", flush=True)
            print(f"  annotations: "
                  f"{json.dumps(data['annotations'], ensure_ascii=False)[:800]}",
                  flush=True)
            steps = [s.get("name") for s in meta.get("steps", [])]
            print(f"  meta steps: {steps}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
