from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal
from typing import Optional

SpeakingPart = Literal["part1", "part2", "part3"]


class GrammarIssue(BaseModel):
    message: str
    context: str
    replacement: Optional[str] = None
    category: str = "grammar"
    severity: Literal["low", "medium", "high"] = "medium"


class PronunciationIssue(BaseModel):
    word: str
    message: str
    start: Optional[float] = None
    end: Optional[float] = None
    severity: Literal["low", "medium", "high"] = "medium"


class FluencyMetric(BaseModel):
    words_per_minute: Optional[float] = None
    estimated_pause_count: int = 0
    note: str


class AnalysisResult(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "failed"]
    source: Literal["audio", "text"]
    speaking_part: SpeakingPart = "part2"
    transcript: str = ""
    grammar_issues: list[GrammarIssue] = Field(default_factory=list)
    pronunciation_issues: list[PronunciationIssue] = Field(default_factory=list)
    natural_suggestions: list[str] = Field(default_factory=list)
    upgraded_answer: str = ""
    fluency: Optional[FluencyMetric] = None
    report_markdown: str = ""
    assistant_notes: str = ""
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class TextAnalyzeRequest(BaseModel):
    text: str
    assistant_notes: Optional[str] = None
    template: Optional[str] = None
    speaking_part: SpeakingPart = "part2"
