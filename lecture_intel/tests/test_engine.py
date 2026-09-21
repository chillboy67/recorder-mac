"""
Engine — the orchestrator.

`run()` is the spine of every mode, so it gets two kinds of coverage here: the
pure helpers it exposes, and a light integration run whose ML collaborators are
stubbed. The integration run still writes real files, so it pins the things that
actually broke before — the provenance ordering, the meta.json steps, and the
fidelity trail reaching both the summary and the exported transcript.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import ASRResult, ASRSegment, ASRWord  # noqa: E402
from core import engine, export as E  # noqa: E402
from core import repeat_arbitration as ra  # noqa: E402


# ── _lecture_context ────────────────────────────────────────────────

@pytest.mark.parametrize("language,expected", [
    ("zh", "一节课的课堂录音"),
    ("mixed", "一节课的课堂录音"),      # code-switching keeps the Chinese hint
    ("", "一节课的课堂录音"),
    (None, "一节课的课堂录音"),
    ("en", "a classroom lecture recording"),
])
def test_lecture_context_follows_the_transcript_language(language, expected):
    assert engine._lecture_context(language) == expected


# ── _ielts_summary ──────────────────────────────────────────────────

def test_ielts_summary_flattens_the_report():
    report = SimpleNamespace(words_per_minute=142.0, filler_count=7,
                            long_pause_count=3, pron_issues=["a", "b"],
                            grammar_issues=["c"], naturalness=["d", "e", "f"],
                            markdown="# report")
    assert engine._ielts_summary(report) == {
        "wpm": 142.0, "filler_count": 7, "long_pause_count": 3,
        "pron_issue_count": 2, "grammar_issue_count": 1, "naturalness_count": 3,
        "markdown": "# report",
    }


# ── _fidelity_summary ───────────────────────────────────────────────

def _annotated(verdicts: list[tuple[str, bool]]) -> ASRResult:
    asr = ASRResult(segments=[], full_text="", language="zh", model_used="t")
    for verdict, folded in verdicts:
        asr.annotations.append({
            "type": "repeat_arbitration", "segment_id": 0, "start": 1.0,
            "end": 2.0, "original": "打打打打", "verdict": verdict,
            "folded": folded, "evidence": {"level": 1, "span_sec": 0.1,
                                           "expected_sec": 2.0}})
    return asr


def test_fidelity_summary_counts_each_verdict_and_the_folds():
    asr = _annotated([("asr_loop", True), ("asr_loop", False),
                      ("real_speech", False), ("uncertain", False)])
    asr.annotations.append({"type": "adjacent_duplicate_segment",
                            "segment_id": 1, "start": 3.0, "end": 4.0,
                            "text": "谢谢", "duplicate_of": 0})
    dropped = [{"segment_id": 9, "start": 5.0, "end": 6.0, "text": "好的",
                "duplicate_of": 8}]

    summary = engine._fidelity_summary(asr, dropped)

    assert summary["asr_loop"] == 2
    assert summary["real_speech"] == 1
    assert summary["uncertain"] == 1
    assert summary["adjacent_duplicates"] == 1
    assert summary["dropped_count"] == 1
    assert summary["folded_count"] == 2          # 1 folded annotation + 1 dropped
    assert summary["total"] == 6                 # 5 annotations + 1 dropped
    assert len(summary["lines"]) == 6            # one readable line per item


def test_fidelity_summary_is_empty_for_a_clean_transcript():
    summary = engine._fidelity_summary(_annotated([]), [])
    assert summary["total"] == 0 and summary["lines"] == []
    assert summary["folded_count"] == 0 and summary["dropped_count"] == 0


def test_fidelity_lines_match_the_exported_tail():
    """The GUI tab and the .md/.txt tail must not drift apart."""
    asr = _annotated([("asr_loop", True)])
    assert engine._fidelity_summary(asr, [])["lines"] == E.fidelity_lines(asr, [])


# ── run(): a light integration pass ─────────────────────────────────

def _seg(i, text, start=0.0, lang="zh"):
    words = [ASRWord(w, start + j * 0.2, start + j * 0.2 + 0.2, -0.1)
             for j, w in enumerate(text)]
    return ASRSegment(id=i, start=start, end=start + len(text) * 0.2, text=text,
                      language=lang, confidence=-0.1, words=words)


def test_run_writes_provenance_outputs_and_a_fidelity_summary(tmp_path, monkeypatch):
    """General mode, no ML: real files in, real outputs out."""
    src = tmp_path / "take.wav"
    src.write_bytes(b"RIFF" + b"\0" * 128)
    out_dir = tmp_path / "out"

    class FakeAudio:
        def __init__(self, path):
            self.path = path
            self.duration_sec = 42.0

    class FakeLoader:
        def __init__(self, config):
            assert config == {}                     # engine passes no YAML anymore

        def process(self, path):
            wav = out_dir / "normalized.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"RIFF" + b"\0" * 64)
            return FakeAudio(wav)

    class FakeTranscriber:
        def __init__(self, model=None, engine=None):
            self.model = model

        def transcribe(self, path, **kwargs):
            kwargs["progress"](0.5, "halfway")      # exercise the progress bridge
            return ASRResult(segments=[_seg(0, "今天讲算法", 0.0),
                                       _seg(1, "接着说", 3.0)],
                             full_text="今天讲算法 接着说", language="zh",
                             model_used="fake", audio_duration_sec=42.0)

    monkeypatch.setattr(engine, "AudioLoader", FakeLoader)
    monkeypatch.setattr(engine, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(ra, "arbitrate", lambda *a, **kw: [])   # no audio probes

    steps: list[str] = []
    summary = engine.run(str(src), str(out_dir), mode_key="general",
                         model="fake", progress=lambda i: steps.append(i["step"]))

    # provenance: the archived original plus a step for every transformation
    assert (out_dir / "original.wav").read_bytes() == src.read_bytes()
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    names = [s["name"] for s in meta["steps"]]
    assert names == ["normalize", "transcribe", "annotations", "export"]
    assert meta["original"]["file"] == "original.wav"

    # summary shape the GUI depends on
    assert summary["mode"] == "general"
    assert summary["language"] == "zh" and summary["segment_count"] == 2
    assert summary["duration_sec"] == 42.0
    assert summary["classroom"] is None and summary["ielts"] is None
    assert summary["fidelity"]["total"] == 0        # nothing flagged here

    # the report reached the progress callback
    assert {"load", "asr", "export"} <= set(steps)

    # exports exist and keep the transcript verbatim
    txt = Path(summary["output_files"]["txt"]).read_text(encoding="utf-8")
    assert "今天讲算法" in txt and "接着说" in txt
    assert "忠实度标注" not in txt                  # no annotations → no tail


def test_run_reports_annotations_through_to_the_export(tmp_path, monkeypatch):
    """The batch-② chain end to end: verdict → annotation → summary → file."""
    src = tmp_path / "take.wav"
    src.write_bytes(b"RIFF" + b"\0" * 128)
    out_dir = tmp_path / "out"

    class FakeAudio:
        def __init__(self, path):
            self.path = path
            self.duration_sec = 10.0

    class FakeLoader:
        def __init__(self, config):
            pass

        def process(self, path):
            wav = out_dir / "normalized.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"RIFF" + b"\0" * 64)
            return FakeAudio(wav)

    class FakeTranscriber:
        def __init__(self, model=None, engine=None):
            pass

        def transcribe(self, path, **kwargs):
            return ASRResult(segments=[_seg(0, "今天讲算法打打打打")],
                             full_text="今天讲算法打打打打", language="zh",
                             model_used="fake", audio_duration_sec=10.0)

    monkeypatch.setattr(engine, "AudioLoader", FakeLoader)
    monkeypatch.setattr(engine, "Transcriber", FakeTranscriber)
    # one confirmed artifact, as L3 would report it against the original
    monkeypatch.setattr(ra, "arbitrate", lambda *a, **kw: [
        ra.Verdict(0, 0.0, 1.0, "打打打打", "asr_loop",
                   {"level": 3, "seeds": [0, 0, 0], "k": 4,
                    "oracle": "original"}, run=None)])

    summary = engine.run(str(src), str(out_dir), mode_key="general", model="fake")

    fid = summary["fidelity"]
    assert fid["asr_loop"] == 1
    assert fid["total"] == 1
    assert fid["lines"] and "打打打打" in fid["lines"][0]
    assert summary["annotations"][0]["verdict"] == "asr_loop"

    # general mode never folds, and the file says so — plus the body is untouched
    txt = Path(summary["output_files"]["txt"]).read_text(encoding="utf-8")
    body, _, tail = txt.partition("忠实度标注")
    assert "今天讲算法打打打打" in body
    assert "正文原样保留" in tail
