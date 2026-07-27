from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional
import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.models import AnalysisResult, SpeakingPart, TextAnalyzeRequest
from app.services.analyzer import analyze_audio, analyze_text


app = FastAPI(title="Offline IELTS Speaking Coach")
jobs: dict[str, AnalysisResult] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/api/health")
async def health() -> dict:
    languagetool_available = False
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.post(settings.languagetool_url, data={"text": "This are a test.", "language": "en-US"})
            languagetool_available = response.status_code < 500
    except Exception:
        languagetool_available = False

    mfa_cli_path = shutil.which(settings.mfa_cli)
    return {
        "ok": True,
        "offline": True,
        "tools": {
            "whisper": {
                "available": bool(shutil.which(settings.whisper_cli)),
                "cli": settings.whisper_cli,
                "model": settings.whisper_model,
            },
            "ffmpeg": {
                "available": bool(shutil.which("ffmpeg")),
            },
            "languagetool": {
                "available": languagetool_available,
                "url": settings.languagetool_url,
            },
            "mfa": {
                "enabled": settings.mfa_enabled,
                "available": bool(mfa_cli_path),
                "cli": settings.mfa_cli,
                "dictionary": settings.mfa_dictionary_path,
                "acoustic_model": settings.mfa_acoustic_model,
                "configured": bool(settings.mfa_enabled and mfa_cli_path and settings.mfa_dictionary_path and settings.mfa_acoustic_model),
            },
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/analyze/text", response_model=AnalysisResult)
async def analyze_text_endpoint(payload: TextAnalyzeRequest) -> AnalysisResult:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    return await analyze_text(payload.text, payload.template, assistant_notes=payload.assistant_notes, speaking_part=payload.speaking_part)


@app.post("/api/analyze/audio", response_model=AnalysisResult)
async def analyze_audio_endpoint(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    transcript: str = Form(""),
    assistant_notes: str = Form(""),
    template: str = Form(""),
    speaking_part: SpeakingPart = Form("part2"),
) -> AnalysisResult:
    job_id = str(uuid.uuid4())
    work_dir = settings.data_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    audio_path = work_dir / f"source{suffix}"
    with audio_path.open("wb") as handle:
        shutil.copyfileobj(audio.file, handle)

    queued = AnalysisResult(job_id=job_id, status="queued", source="audio", speaking_part=speaking_part)
    jobs[job_id] = queued
    background_tasks.add_task(_run_audio_job, job_id, audio_path, transcript, work_dir, template or None, assistant_notes, speaking_part)
    return queued


@app.get("/api/jobs/{job_id}", response_model=AnalysisResult)
def get_job(job_id: str) -> AnalysisResult:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs[job_id]


async def _run_audio_job(job_id: str, audio_path: Path, transcript: str, work_dir: Path, template: Optional[str], assistant_notes: str, speaking_part: SpeakingPart) -> None:
    jobs[job_id] = AnalysisResult(job_id=job_id, status="running", source="audio", speaking_part=speaking_part)
    try:
        jobs[job_id] = await analyze_audio(audio_path, transcript, work_dir, template, job_id, assistant_notes, speaking_part)
    except Exception as exc:
        jobs[job_id] = AnalysisResult(job_id=job_id, status="failed", source="audio", speaking_part=speaking_part, error=str(exc))
