import pytest
from pathlib import Path
from app.services.analyzer import analyze_text, extract_student_speech
from app.services.grammar import _fallback_rules
from app.services.report import render_report
from app.services.transcription import transcribe_audio
from app.services.pronunciation import _mfa_environment, _prepare_mfa_audio
from app.models import AnalysisResult


def test_fallback_rules_detect_common_ielts_errors():
    issues = _fallback_rules("He have many informations and discuss about school.")
    messages = " ".join(issue.message for issue in issues)
    assert "has" in messages
    assert "Information" in messages
    assert "discuss" in messages


@pytest.mark.asyncio
async def test_analyze_text_generates_report():
    result = await analyze_text("I think it is very important because he have a lot of advices.")
    assert result.status == "complete"
    assert result.transcript
    assert result.grammar_issues
    assert result.natural_suggestions
    assert "口语反馈" in result.report_markdown
    assert "继续加油" in result.report_markdown


def test_custom_template_renders_known_fields():
    result = AnalysisResult(job_id="x", status="complete", source="text", transcript="Hello world.")
    report = render_report(result, "Transcript: {transcript}\nPlan: {practice_plan}")
    assert "Hello world." in report
    assert "P2先复练纠错句" in report


def test_default_report_for_part2_omits_part1_and_part3():
    result = AnalysisResult(job_id="x", status="complete", source="audio", transcript="I visited a museum last year.")
    report = render_report(result)
    assert "part2" in report
    assert "part1" not in report
    assert "part3" not in report


def test_default_report_for_part1_only_outputs_part1():
    result = AnalysisResult(job_id="x", status="complete", source="text", speaking_part="part1", transcript="I like reading.")
    report = render_report(result)
    assert "part1" in report
    assert "part2" not in report
    assert "part3" not in report


def test_default_report_for_part3_only_outputs_part3():
    result = AnalysisResult(job_id="x", status="complete", source="text", speaking_part="part3", transcript="Technology changes how people study.")
    report = render_report(result)
    assert "part3" in report
    assert "part1" not in report
    assert "part2" not in report


def test_mfa_environment_includes_mfa_bin(monkeypatch):
    monkeypatch.setattr("app.services.pronunciation.settings.mfa_cli", "/tmp/mfa-env/bin/mfa")
    env = _mfa_environment()
    assert env["PATH"].split(":")[0] == str(Path("/tmp/mfa-env/bin").resolve())


def test_prepare_mfa_audio_converts_to_wav(monkeypatch, tmp_path):
    audio_path = tmp_path / "source.webm"
    audio_path.write_bytes(b"fake audio")
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (corpus_dir / "sample.wav").write_bytes(b"fake wav")
        return _completed(command)

    monkeypatch.setattr("app.services.pronunciation.shutil.which", lambda command, path=None: "/usr/bin/ffmpeg")
    monkeypatch.setattr("app.services.pronunciation.subprocess.run", fake_run)
    wav_path = _prepare_mfa_audio(audio_path, corpus_dir, "sample")
    assert wav_path == corpus_dir / "sample.wav"
    assert commands[0][-1] == str(corpus_dir / "sample.wav")
    assert "-ac" in commands[0]
    assert "-ar" in commands[0]



def test_extract_student_speech_ignores_tutor_turns():
    mixed = """助教：What do you usually do on weekends?
学生：I has many advices from my friend.
助教：注意三单，I has 要改。"""
    student = extract_student_speech(mixed)
    assert "I has many advices" in student
    assert "What do you" not in student
    assert "注意三单" not in student


def test_transcribe_audio_warns_when_whisper_missing(monkeypatch, tmp_path):
    def fake_which(command):
        return None if command == "whisper" else f"/usr/bin/{command}"

    monkeypatch.setattr("app.services.transcription.shutil.which", fake_which)
    transcript, warnings = transcribe_audio(tmp_path / "source.wav", tmp_path)
    assert transcript == ""
    assert warnings == ["Whisper CLI is not installed; pronunciation analysis will use the optional transcript only."]


def test_transcribe_audio_warns_when_ffmpeg_missing(monkeypatch, tmp_path):
    calls = []

    def fake_which(command):
        return "/usr/local/bin/whisper" if command == "whisper" else None

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Whisper should not run without ffmpeg.")

    monkeypatch.setattr("app.services.transcription.shutil.which", fake_which)
    monkeypatch.setattr("app.services.transcription.subprocess.run", fake_run)
    transcript, warnings = transcribe_audio(tmp_path / "source.wav", tmp_path)
    assert transcript == ""
    assert warnings == ["ffmpeg is not installed or not on PATH; install ffmpeg and retry audio transcription."]
    assert calls == []


def test_transcribe_audio_reads_expected_json(monkeypatch, tmp_path):
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"fake audio")

    def fake_run(command, **kwargs):
        output_dir = tmp_path / "whisper"
        (output_dir / "source.json").write_text('{"text": " Hello IELTS. "}', encoding="utf-8")
        (output_dir / "other.json").write_text('{"text": "Wrong file."}', encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr("app.services.transcription.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr("app.services.transcription.subprocess.run", fake_run)
    transcript, warnings = transcribe_audio(audio_path, tmp_path)
    assert transcript == "Hello IELTS."
    assert warnings == []


def test_transcribe_audio_warns_when_json_missing(monkeypatch, tmp_path):
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"fake audio")

    def fake_run(command, **kwargs):
        return _completed(command, stdout="done without files")

    monkeypatch.setattr("app.services.transcription.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr("app.services.transcription.subprocess.run", fake_run)
    transcript, warnings = transcribe_audio(audio_path, tmp_path)
    assert transcript == ""
    assert len(warnings) == 1
    assert "Whisper finished but did not produce a JSON transcript" in warnings[0]
    assert str(tmp_path / "whisper") in warnings[0]
    assert "done without files" in warnings[0]


def test_transcribe_audio_warns_when_json_invalid(monkeypatch, tmp_path):
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"fake audio")

    def fake_run(command, **kwargs):
        (tmp_path / "whisper" / "source.json").write_text("{not json", encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr("app.services.transcription.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr("app.services.transcription.subprocess.run", fake_run)
    transcript, warnings = transcribe_audio(audio_path, tmp_path)
    assert transcript == ""
    assert len(warnings) == 1
    assert "Whisper produced invalid JSON transcript source.json" in warnings[0]


def _completed(command, stdout="", stderr=""):
    import subprocess

    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=stderr)
