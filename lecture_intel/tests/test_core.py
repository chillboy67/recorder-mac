"""
Core-logic regression tests that don't need a Whisper model.

They exercise the IELTS analysis, speaker-label handling, and exporters on a
synthetic ASRResult, so they run in milliseconds and catch wiring bugs.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import ASRResult, ASRSegment, ASRWord  # noqa: E402
from core import ielts as I  # noqa: E402
from core import export as E  # noqa: E402
from core.diarize import CANDIDATE, EXAMINER  # noqa: E402
from core.modes import get_mode, MODES  # noqa: E402


def _sample_asr() -> ASRResult:
    segs = [
        ASRSegment(0, 0.0, 3.0, "Can you describe a folk tale you know?", "en", -0.2,
                   [ASRWord("Can", 0, 0.3, 0.99), ASRWord("describe", 0.5, 1.0, 0.95)]),
        ASRSegment(1, 3.2, 12.0,
                   "Well, I would talk about Bluebeard, I heard it at promary school.", "en", -0.3,
                   [ASRWord("Bluebeard", 5.0, 5.6, 0.88),
                    ASRWord("promary", 9.0, 9.6, 0.21),     # low-confidence → flagged
                    ASRWord("school", 9.6, 10.0, 0.97)]),
        ASRSegment(2, 12.5, 14.0, "What happens next?", "en", -0.2,
                   [ASRWord("What", 12.5, 12.8, 0.99)]),
        ASRSegment(3, 15.5, 22.0, "He have many wives and it is a very good story.", "en", -0.25,
                   [ASRWord("have", 14.4, 14.7, 0.6), ASRWord("wives", 15.0, 15.4, 0.4)]),
    ]
    return ASRResult(segments=segs, full_text=" ".join(s.text for s in segs),
                     language="en", model_used="test", audio_duration_sec=22.0)


@pytest.fixture
def labels():
    return {0: EXAMINER, 1: CANDIDATE, 2: EXAMINER, 3: CANDIDATE}


def test_pronunciation_flags_low_confidence(labels):
    asr = _sample_asr()
    # no LanguageTool server in CI → offline rules
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    flagged = {p.word.lower() for p in report.pron_issues}
    assert "promary" in flagged          # the mispronounced word is caught
    assert all(p.confidence < I.PRON_CONFIDENCE_THRESHOLD for p in report.pron_issues)


def test_transcript_is_verbatim(labels):
    asr = _sample_asr()
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    # candidate text must preserve the original errors exactly
    assert "promary" in report.transcript_candidate
    assert "He have" in report.transcript_candidate


def test_offline_grammar_and_naturalness(labels):
    asr = _sample_asr()
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    msgs = " ".join(g.message for g in report.grammar_issues)
    assert "has" in msgs                 # "He have" → has
    assert any("very good" in n for n in report.naturalness)


def test_export_all_formats(tmp_path, labels):
    asr = _sample_asr()
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    out = E.export_all(asr, tmp_path, "s", ["txt", "md", "doc", "docx"],
                       labels=labels, extra_markdown=report.markdown,
                       extra_markdown_suffix="ielts")
    for k in ("txt", "md", "docx", "report"):
        assert out[k].exists() and out[k].stat().st_size > 0
    # .doc relies on macOS textutil; assert when available
    import shutil
    if shutil.which("textutil"):
        assert out["doc"].exists() and out["doc"].stat().st_size > 0
    txt = out["txt"].read_text(encoding="utf-8")
    assert "考生" in txt and "教官" in txt          # speaker labels rendered
    assert "promary" in txt                          # verbatim in output


def test_docx_contains_report_text(tmp_path, labels):
    asr = _sample_asr()
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    out = E.export_all(asr, tmp_path, "s", ["docx"], labels=labels,
                       extra_markdown=report.markdown)
    from docx import Document
    text = "\n".join(p.text for p in Document(str(out["docx"])).paragraphs)
    assert "promary" in text          # verbatim transcript inside the Word doc


def test_repetition_collapse():
    from core.transcriber import _collapse_repeats
    # Whisper hallucination loops collapse to one copy
    assert _collapse_repeats("about this " * 8).strip() == "about this"
    assert len(_collapse_repeats("no no no no no no").split()) <= 2   # loop gone
    # Chinese (no spaces): a runaway char/phrase loop collapses
    assert _collapse_repeats("時" * 200) == "時"
    assert "時" * 10 not in _collapse_repeats("算法" + "時" * 150 + "分析")
    # genuine emphasis (3x) and normal text are left alone
    assert _collapse_repeats("well no no no I disagree") == "well no no no I disagree"
    assert _collapse_repeats("I really like it a lot") == "I really like it a lot"


# ── repeat arbitration: stutter vs ASR loop ─────────────────────────

def _wseg(tokens, language="en", seg_id=0):
    """Build a segment from (word, start, end) tuples, Whisper-style: the text
    is the concatenation of the tokens (English tokens carry their own leading
    space), and the words carry the timeline."""
    ws = [ASRWord(w, s, e, 0.9) for (w, s, e) in tokens]
    return ASRSegment(id=seg_id, start=tokens[0][1], end=tokens[-1][2],
                      text="".join(w.word for w in ws), language=language,
                      confidence=-0.2, words=ws)


def _loop_tokens(word, k, dur=0.28, step=None):
    """k copies of `word`, each advancing dur seconds (real-stutter shape)."""
    if step is None:
        step = dur
    return [((" " + word) if i else word, i * step, i * step + dur)
            for i in range(k)]


def test_repeat_run_detected_en_and_zh():
    from core.repeat_arbitration import find_repeat_runs
    en = _wseg(_loop_tokens("no", 4))
    (run,) = find_repeat_runs(en)
    assert run.k == 4 and run.n == 1 and run.gram == ("no",)
    zh = _wseg(_loop_tokens("我", 5, dur=0.2, step=0.25), language="zh")
    (run,) = find_repeat_runs(zh)
    assert run.k == 5 and run.n == 1


def test_repeat_run_ignores_k3():
    from core.repeat_arbitration import find_repeat_runs
    assert find_repeat_runs(_wseg(_loop_tokens("no", 3))) == []
    # genuine emphasis "no, no, no" plus more speech: nothing folds
    toks = _loop_tokens("no", 3) + [(" I disagree", 1.0, 1.3)]
    assert find_repeat_runs(_wseg(toks)) == []


def test_l1_frozen_timeline_is_asr_loop():
    from core.repeat_arbitration import classify_run, find_repeat_runs
    # word clocks do not advance across copies → decoder froze → asr_loop,
    # decided at level 1 without ever touching audio
    toks = [("no", 1.0, 1.01)] * 4
    (run,) = find_repeat_runs(_wseg(toks))
    v = classify_run(run, None, None, "en", median_dur=0.25)
    assert v.verdict == "asr_loop" and v.evidence["level"] == 1


def test_l1_advancing_timeline_proceeds_to_l2():
    from core.repeat_arbitration import classify_run, find_repeat_runs
    import core.repeat_arbitration as ra
    toks = _loop_tokens("no", 4, dur=0.25, step=0.35)
    (run,) = find_repeat_runs(_wseg(toks))
    # stub L2: 4 voiced bursts → real_speech without needing real audio
    monkey = lambda *a, **k: 4
    orig = ra._count_voiced_bursts
    ra._count_voiced_bursts = monkey
    try:
        v = classify_run(run, None, None, "en", median_dur=0.25)
    finally:
        ra._count_voiced_bursts = orig
    assert v.verdict == "real_speech" and v.evidence["level"] == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_l2_silencedetect_on_synthetic_audio(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from core.repeat_arbitration import classify_run, find_repeat_runs
    import core.repeat_arbitration as ra
    sr = 16000

    def wav_of(kind):
        if kind == "pulses":        # 4 voiced bursts aligned with the run's
            w = np.zeros(sr * 3)    # word clock ≈ a real stutter
            for i in range(4):
                n = int(0.25 * sr)
                t0 = int((i * 0.35 + 0.02) * sr)
                w[t0:t0 + n] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)
        elif kind == "tone":        # one continuous tone ≈ a frozen loop
            n = sr * 3
            w = 0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)
        else:                       # silence ≈ a loop over nothing
            w = np.zeros(sr * 3)
        p = tmp_path / f"{kind}.wav"
        sf.write(p, w, sr)
        return p

    class Mock:
        """L3 re-decode: the greedy model hears exactly one copy."""
        def transcribe(self, path, **kw):
            seg = _wseg(_loop_tokens("no", 1))
            return ASRResult(segments=[seg], full_text=seg.text,
                             language="en", model_used="mock")

    toks = _loop_tokens("no", 4, dur=0.25, step=0.35)
    (run,) = find_repeat_runs(_wseg(toks))
    # strong voicing evidence is conclusive for real speech at L2
    v = classify_run(run, wav_of("pulses"), None, "en", median_dur=0.25)
    assert v.verdict == "real_speech" and v.evidence["level"] == 2
    # low burst count is NOT trusted as asr_loop — it must escalate to L3
    for kind in ("tone", "silence"):
        v = classify_run(run, wav_of(kind), Mock(), "en", median_dur=0.25)
        assert v.verdict == "asr_loop", (kind, v.evidence)
        assert v.evidence["level"] == 3, (kind, v.evidence)


def test_l1_long_rapid_real_run_escapes_frozen_clock_test(tmp_path, monkeypatch):
    """Real rapid stutter: 8 copies in 1.2 s (per-copy ≪ median word). The
    relative timeline test would flag it (span < 0.5×expected) — the absolute
    FROZEN_SPAN_MAX cap must save it, and L2's low burst count must escalate
    to L3 instead of concluding asr_loop directly."""
    import core.repeat_arbitration as ra
    monkeypatch.setattr(ra, "_extract_clip", lambda *a, **k: str(tmp_path / "clip.wav"))
    toks = _loop_tokens("no", 8, dur=0.15, step=0.15)
    (run,) = ra.find_repeat_runs(_wseg(toks))
    monkeypatch.setattr(ra, "_count_voiced_bursts", lambda *a, **k: 2)

    class Mock:
        def transcribe(self, path, **kw):
            seg = _wseg(_loop_tokens("no", 8))
            return ASRResult(segments=[seg], full_text=seg.text,
                             language="en", model_used="mock")

    v = ra.classify_run(run, "x.wav", Mock(), "en", median_dur=0.25)
    assert v.verdict == "real_speech"
    assert v.evidence["level"] == 3 and v.evidence["redecoded_copies"] == 8


def test_l3_redecode_with_mock_transcriber(tmp_path, monkeypatch):
    import core.repeat_arbitration as ra
    monkeypatch.setattr(ra, "_extract_clip", lambda *a, **k: str(tmp_path / "clip.wav"))
    # 3 bursts for k=4 sits between the L2 thresholds (2.4 / 3.2) → inconclusive
    monkeypatch.setattr(ra, "_count_voiced_bursts", lambda *a, **k: 3)
    toks = _loop_tokens("no", 4, dur=0.25, step=0.35)
    (run,) = ra.find_repeat_runs(_wseg(toks))

    class Mock:
        def __init__(self, k):
            self.k = k

        def transcribe(self, path, **kw):
            assert kw["condition_on_previous"] is False
            assert kw["temperature"] in ((0.0,), (0.2,), (0.8,))
            seg = _wseg(_loop_tokens("no", self.k))
            return ASRResult(segments=[seg], full_text=seg.text,
                             language="en", model_used="mock")

    v = ra.classify_run(run, "x.wav", Mock(1), "en", median_dur=0.25)
    assert v.verdict == "asr_loop" and v.evidence["level"] == 3
    v = ra.classify_run(run, "x.wav", Mock(4), "en", median_dur=0.25)
    assert v.verdict == "real_speech"
    v = ra.classify_run(run, "x.wav", Mock(2), "en", median_dur=0.25)
    assert v.verdict == "uncertain"


def test_l3_empty_or_misheard_redecode_is_uncertain_not_asr_loop(tmp_path, monkeypatch):
    """A greedy re-decode that returns NOTHING, or hears different words than
    the n-gram (e.g. a shouted window on phone audio), must not count as
    'model heard one copy' — folding on that would destroy real stutter."""
    import core.repeat_arbitration as ra
    monkeypatch.setattr(ra, "_extract_clip", lambda *a, **k: str(tmp_path / "clip.wav"))
    monkeypatch.setattr(ra, "_count_voiced_bursts", lambda *a, **k: 1)
    toks = _loop_tokens("谢谢", 4, dur=0.3, step=0.35)
    (run,) = ra.find_repeat_runs(_wseg(toks, language="zh"))

    class Empty:
        def transcribe(self, path, **kw):
            return ASRResult(segments=[], full_text="", language="zh",
                             model_used="mock")

    class Mishear:
        def transcribe(self, path, **kw):   # hears something, but not 谢谢
            seg = _wseg(_loop_tokens("是", 2, dur=0.3, step=0.35), language="zh")
            return ASRResult(segments=[seg], full_text=seg.text,
                             language="zh", model_used="mock")

    for mock in (Empty(), Mishear()):
        v = ra.classify_run(run, "x.wav", mock, "zh", median_dur=0.3)
        assert v.verdict == "uncertain"
        assert v.evidence["level"] == 3


def test_l3_sampling_seed_rescues_real_stutter_greedy_heard_nothing(
        tmp_path, monkeypatch):
    """Greedy (seed 0.0) hears nothing on a degraded window, but a sampling
    seed reproduces the run → real_speech: the n-gram IS in the audio. This is
    the 14.3 'shouted phone-call window' correction, now recovered by seeds."""
    import core.repeat_arbitration as ra
    monkeypatch.setattr(ra, "_extract_clip",
                        lambda *a, **k: str(tmp_path / "clip.wav"))
    monkeypatch.setattr(ra, "_count_voiced_bursts", lambda *a, **k: 1)
    toks = _loop_tokens("打", 4, dur=0.3, step=0.35)
    (run,) = ra.find_repeat_runs(_wseg(toks, language="zh"))

    class Mock:
        def __init__(self, per_seed):
            self.per_seed = iter(per_seed)

        def transcribe(self, path, **kw):
            word, k = next(self.per_seed)
            seg = _wseg(_loop_tokens(word, k, dur=0.3, step=0.35),
                        language="zh")
            return ASRResult(segments=[seg], full_text=seg.text,
                             language="zh", model_used="mock")

    v = ra.classify_run(run, "x.wav", Mock([("是", 1), ("打", 4), ("打", 2)]),
                        "zh", median_dur=0.3)
    assert v.verdict == "real_speech"
    assert v.evidence["seeds"] == [0, 4, 2]
    assert v.evidence["oracle"] == "pipeline"


def test_l3_redecodes_against_oracle_original(tmp_path, monkeypatch):
    """The engine passes the pre-denoise original as the L3 oracle; the
    re-decode clip must be cut from that file, not the denoised pipeline wav.
    A single surviving copy on the original confirms the collapse → asr_loop."""
    import core.repeat_arbitration as ra
    seen: list[str] = []
    monkeypatch.setattr(
        ra, "_extract_clip",
        lambda wav, a, b: (seen.append(str(wav)) or str(tmp_path / "clip.wav")))
    monkeypatch.setattr(ra, "_count_voiced_bursts", lambda *a, **k: 1)
    toks = _loop_tokens("no", 4, dur=0.25, step=0.35)
    (run,) = ra.find_repeat_runs(_wseg(toks))

    class Mock:
        def transcribe(self, path, **kw):
            seg = _wseg(_loop_tokens("no", 1))
            return ASRResult(segments=[seg], full_text=seg.text,
                             language="en", model_used="mock")

    v = ra.classify_run(run, "/pip/denoised.wav", Mock(), "en",
                        median_dur=0.25, oracle_path="/pip/original.wav")
    assert v.verdict == "asr_loop"
    assert v.evidence["oracle"] == "original"
    assert seen and all(p == "/pip/original.wav" for p in seen)


def test_l3_all_seeds_zero_on_original_is_asr_loop_but_pipeline_stays_uncertain(
        tmp_path, monkeypatch):
    """Zero copies on every seed over the PRE-DENOISE original → the repetition
    is absent from the recording → asr_loop (foldable in classroom, original
    kept in the annotation). The SAME re-decode on the pipeline audio stays
    uncertain — 14.3's '0 copies = ambiguous' correction still holds there."""
    import core.repeat_arbitration as ra
    monkeypatch.setattr(ra, "_extract_clip",
                        lambda *a, **k: str(tmp_path / "clip.wav"))
    monkeypatch.setattr(ra, "_count_voiced_bursts", lambda *a, **k: 1)
    toks = _loop_tokens("打", 4, dur=0.3, step=0.35)
    (run,) = ra.find_repeat_runs(_wseg(toks, language="zh"))

    class Mishear:
        def transcribe(self, path, **kw):
            seg = _wseg(_loop_tokens("是", 2, dur=0.3, step=0.35),
                        language="zh")
            return ASRResult(segments=[seg], full_text=seg.text,
                             language="zh", model_used="mock")

    v = ra.classify_run(run, "x.wav", Mishear(), "zh", median_dur=0.3,
                        oracle_path="/pip/original.wav")
    assert v.verdict == "asr_loop"
    assert v.evidence["oracle"] == "original"
    assert v.evidence["seeds"] == [0, 0, 0]

    v = ra.classify_run(run, "x.wav", Mishear(), "zh", median_dur=0.3)
    assert v.verdict == "uncertain"
    assert v.evidence["oracle"] == "pipeline"


def test_apply_verdicts_general_annotates_only():
    from core.repeat_arbitration import Verdict, apply_verdicts
    asr = ASRResult(segments=[_wseg(_loop_tokens("no", 4))],
                    full_text="no no no no", language="en", model_used="t")
    assert asr.annotations == []            # default is empty
    v = Verdict(0, 0.0, 1.0, "no no no no", "asr_loop", {"level": 1})
    apply_verdicts(asr, [v], "general")
    assert asr.full_text == "no no no no"   # general mode never folds
    (ann,) = asr.annotations
    assert ann["type"] == "repeat_arbitration"
    assert ann["original"] == "no no no no" and ann["verdict"] == "asr_loop"


def test_apply_verdicts_classroom_folds_asr_loop():
    from core.repeat_arbitration import apply_verdicts, arbitrate
    # frozen word clock → L1 asr_loop, no audio needed
    toks = [("no" if i == 0 else " no", 1.0, 1.01) for i in range(4)]
    toks += [(" I disagree", 1.6, 1.9)]
    seg = _wseg(toks)
    asr = ASRResult(segments=[seg], full_text=seg.text, language="en",
                    model_used="t")
    verdicts = arbitrate(asr, None, None)
    assert verdicts and all(v.verdict == "asr_loop" for v in verdicts)
    apply_verdicts(asr, verdicts, "classroom")
    assert asr.segments[0].text == "no I disagree"   # folded to one copy
    assert asr.annotations[0]["original"] == "no no no no"   # original kept


def test_text_fallback_run_is_never_folded():
    from core.repeat_arbitration import classify_run, find_repeat_runs
    seg = ASRSegment(id=0, start=0.0, end=4.0, text="時" * 30,
                     language="zh", confidence=-0.2)   # no words
    (run,) = find_repeat_runs(seg)
    assert run.verified is False
    v = classify_run(run, None, None, "zh", median_dur=0.25)
    assert v.verdict == "uncertain"


def test_adjacent_duplicate_segments_reported():
    from core.repeat_arbitration import adjacent_duplicate_segments
    a = ASRSegment(id=0, start=0.0, end=1.0, text="about this", language="en",
                   confidence=-0.2)
    b = ASRSegment(id=1, start=1.0, end=2.0, text="about this", language="en",
                   confidence=-0.2)
    c = ASRSegment(id=2, start=2.0, end=3.0, text="something else entirely now",
                   language="en", confidence=-0.2)
    asr = ASRResult(segments=[a, b, c], full_text="x", language="en",
                    model_used="t")
    (dup,) = adjacent_duplicate_segments(asr)
    assert dup["segment_id"] == 1 and dup["duplicate_of"] == 0


def test_modes_present():
    assert set(MODES) == {"general", "classroom", "ielts"}
    assert get_mode("ielts").analyze_ielts is True
    assert get_mode("classroom").denoise is True
    assert get_mode("general").diarize is False


# ── language-based examiner / candidate split ────────────────────────

def _seg(text: str, language: str = "en") -> ASRSegment:
    return ASRSegment(id=0, start=0.0, end=1.0, text=text,
                      language=language, confidence=-0.2)


def test_coach_split_matches_old_chinese_test_on_zh_en():
    """This split is what makes examiner/candidate separation usable, so the
    zh/en case must be bit-for-bit what the old hard-coded `_has_chinese` test
    produced. These segments carry the candidate's language, so the script check
    decides — exactly as it did before."""
    import re
    from core.diarize import _is_coach_segment

    old_has_chinese = lambda t: len(re.findall(r"[一-鿿]", t)) >= 2
    for text in [
        "Can you describe a folk tale you know?",
        "Well, I would talk about Bluebeard, I heard it at promary school.",
        "这里要注意时态，你用了过去式。",
        "这个 very good",
        "我 said something",          # one stray char is not a turn
        "What happens next?",
        "",
    ]:
        assert _is_coach_segment(_seg(text)) == old_has_chinese(text), text


@pytest.mark.parametrize("text", [
    "すみません、時制が間違っています",        # Japanese coach
    "시제를 잘못 쓰셨어요",                    # Korean coach
    "Обратите внимание на время глагола",      # Russian coach
    "คุณใช้ไวยากรณ์ผิดตรงนี้",                 # Thai coach
])
def test_coach_split_recognises_any_coach_language(text):
    """The coach no longer has to be Chinese-speaking."""
    from core.diarize import _is_coach_segment
    assert _is_coach_segment(_seg(text))


def test_latin_script_coach_needs_the_segment_language():
    """A Latin-script coach writes the same letters as the English candidate, so
    no script test can separate them — the per-chunk language the chunked ASR
    path attaches to each segment is the signal that does."""
    from core.diarize import _is_coach_segment

    french = "Attention, vous avez utilisé le passé composé."
    assert _is_coach_segment(_seg(french, "fr"))
    assert _is_coach_segment(_seg(french, "de"))
    # Without a per-segment language it is indistinguishable from the candidate.
    assert not _is_coach_segment(_seg(french, "en"))


def test_script_fallback_still_covers_what_language_cannot():
    """Short turns, and the single-pass / faster-whisper paths, carry no usable
    per-segment language — the script test has to keep doing the work there."""
    from core.diarize import _is_coach_segment
    assert _is_coach_segment(_seg("这里要注意时态"))              # language defaults to en
    assert not _is_coach_segment(_seg("a perfectly normal English answer"))


def test_mixed_segments_defer_to_the_script_check():
    """A zh/en code-switched turn is labelled "mixed"; it must not be swept to
    the coach's side merely for not being "en"."""
    from core.diarize import _is_coach_segment
    assert _is_coach_segment(_seg("这个 very good", "mixed"))
    assert not _is_coach_segment(_seg("we switched to English here", "mixed"))


def test_foreign_script_count_respects_the_reference_language():
    from core.languages import foreign_script_count
    assert foreign_script_count("日本語です", "en") >= 2       # foreign to English
    assert foreign_script_count("日本語です", "ja") == 0       # native to Japanese
    assert foreign_script_count("한국어입니다", "ja") >= 2     # Hangul is not Japanese
    assert foreign_script_count("hello there", "en") == 0


# ── _candidate_score regression tests (#10) ─────────────────────────

def _scored_seg(text: str, language: str = "en") -> ASRSegment:
    return ASRSegment(id=0, start=0.0, end=5.0, text=text,
                      language=language, confidence=-0.2)


def test_candidate_score_prefers_english_cluster_zh_en():
    """Regression: the Chinese coach cluster must score lower than the English
    candidate cluster — same behaviour as the old latin/cjk ratio."""
    from core.diarize import _candidate_score
    coach = [_scored_seg("这里要注意时态，你用了过去式。", "zh"),
             _scored_seg("这个 very good", "zh")]
    candidate = [_scored_seg("Well, I would talk about Bluebeard.", "en"),
                 _scored_seg("He have many wives and it is a very good story.", "en")]
    assert _candidate_score(coach) < _candidate_score(candidate)


def test_candidate_score_prefers_english_cluster_fr_en():
    """#10 regression: a French examiner cluster must score lower than an
    English candidate cluster.  The old latin/cjk ratio gave both ≈1.0 and
    degenerated to word count, flipping the assignment."""
    from core.diarize import _candidate_score
    examiner = [_scored_seg("Attention, vous avez utilisé le passé composé.", "fr"),
                _scored_seg("N'oubliez pas l'accord du participe passé.", "fr")]
    candidate = [_scored_seg("Well, I would talk about Bluebeard.", "en"),
                 _scored_seg("He have many wives and it is a very good story.", "en")]
    assert _candidate_score(examiner) < _candidate_score(candidate)


# ── output location ────────────────────────────────────────────────────

def _fake_paths_at(tmp_path, monkeypatch, where):
    """Point core.paths at a fake location, as if the code lived in `where`."""
    from core import paths

    fake = where / "core" / "paths.py"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("")
    monkeypatch.setattr(paths, "__file__", str(fake))
    return paths


def test_output_goes_into_the_folder_git_cloned(tmp_path, monkeypatch):
    """Another user's output must land in *their* checkout, wherever they put
    it — the path may never be baked in on the author's machine."""
    clone = tmp_path / "Downloads" / "recorder"
    (clone / ".git").mkdir(parents=True)
    paths = _fake_paths_at(tmp_path, monkeypatch, clone / "lecture_intel")

    assert paths.clone_root() == clone
    out = paths.default_output_root()
    assert out == clone / "output"
    assert out.is_dir()                       # created on demand


def test_installed_app_never_defaults_to_a_system_folder(tmp_path, monkeypatch):
    """make_app.sh flattens lecture_intel/ into ~/Library/Application Support,
    where there is no checkout. Output must not follow the code there."""
    install = tmp_path / "Application Support" / "Recorder"
    paths = _fake_paths_at(tmp_path, monkeypatch, install)
    monkeypatch.setattr(paths, "data_root",
                        lambda: tmp_path / "Documents" / "Recorder")

    assert paths.clone_root() is None
    out = paths.default_output_root()
    assert out == tmp_path / "Documents" / "Recorder"
    assert "Application Support" not in str(out)
    assert "/Applications" not in str(out)


def test_repo_output_root_is_inside_the_checkout():
    from core import paths
    out = paths.default_output_root()
    assert out == paths.clone_root() / "output"
    assert "Application Support" not in str(out)
