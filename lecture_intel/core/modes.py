"""
Processing modes.

A *mode* bundles all the knobs that change between the three use cases the
user cares about:

  - general   : highest-accuracy faithful transcription, nothing fancy.
  - classroom : far-field / echoey room, possible background chatter →
                 light denoise + keep only the main speaker, never rewrite words.
  - ielts     : a two-person speaking practice. Respect the original text
                 *exactly* (errors are diagnostic gold), separate examiner vs
                 candidate, flag likely pronunciation / grammar / phrasing issues.

The cardinal rule across every mode: **never silently rewrite the speaker's
words.** Whisper already transcribes faithfully; we do not run any LLM
"correction" that paraphrases. Terminology fixes (classroom only) touch
spelling of technical terms, not sentence structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Mode:
    key: str
    label: str
    description: str

    # --- ASR ---
    # None = let Whisper auto-detect (handles zh/en code-switching).
    language: Optional[str] = None
    # A short, neutral prompt nudges spelling without biasing content.
    # Keep it EMPTY unless the domain genuinely needs it — a wrong prompt
    # hurts accuracy more than it helps.
    initial_prompt: str = ""
    # condition_on_previous_text carries context across 30s windows. Great for
    # clean speech, but on noisy audio it can trigger repetition loops, so
    # classroom turns it off.
    condition_on_previous: bool = True

    # --- audio pre-processing ---
    denoise: bool = False          # ffmpeg afftdn + highpass (classroom)

    # --- speaker handling ---
    diarize: bool = False          # split speakers (ielts: examiner vs candidate)
    expected_speakers: int = 0     # 0 = auto; ielts = 2
    keep_main_speaker_only: bool = False  # classroom: drop background chatter

    # --- analysis ---
    analyze_ielts: bool = False    # pronunciation / grammar / phrasing report

    # --- terminology ---
    fix_terminology: bool = False  # classroom only; spelling of domain terms

    formats: list[str] = field(default_factory=lambda: ["txt", "md", "srt", "json"])


GENERAL = Mode(
    key="general",
    label="通用转写",
    description="最高精度、忠实原文的语音转文字。中英混合自动识别。",
    language=None,
    initial_prompt="",
    condition_on_previous=True,
    formats=["txt", "md", "docx"],
)

CLASSROOM = Mode(
    key="classroom",
    label="课堂录音",
    description="空旷/有回声的教室。轻度降噪，聚焦主讲人，排除旁人闲聊。术语拼写纠正。不改句子。",
    language=None,
    initial_prompt="",
    condition_on_previous=False,   # noisy → avoid repetition loops
    denoise=True,
    keep_main_speaker_only=True,
    fix_terminology=True,
    formats=["txt", "md", "docx"],
)

IELTS = Mode(
    key="ielts",
    label="雅思口语教官",
    description="教官与考生对话练习。尊重原文（绝不纠正），区分教官/考生，标注疑似读音、语法、表达问题，生成反馈。",
    language=None,
    initial_prompt="",
    condition_on_previous=True,
    diarize=True,
    expected_speakers=2,
    analyze_ielts=True,
    formats=["txt", "md", "docx"],
)

MODES: dict[str, Mode] = {m.key: m for m in (GENERAL, CLASSROOM, IELTS)}


def get_mode(key: str) -> Mode:
    return MODES.get(key, GENERAL)
