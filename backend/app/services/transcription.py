from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import warnings as py_warnings

from app.config import settings

# ==================== 第一阶段：使用 faster-whisper ====================

def transcribe_audio(audio_path: Path, work_dir: Path) -> Tuple[str, List[str]]:
    """
    使用 faster-whisper 进行语音转录（第一阶段推荐方案）
    """
    warnings: List[str] = []

    if not settings.use_faster_whisper:
        # 兼容旧版 CLI 方式
        return _transcribe_with_cli(audio_path, work_dir, warnings)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        warnings.append("faster-whisper 未安装，请运行: pip install faster-whisper")
        return "", warnings

    # 选择设备和计算精度
    device = "cpu"
    compute_type = "int8"  # Mac 上推荐 int8，速度快且占用低

    try:
        model = WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=compute_type
        )
    except Exception as e:
        warnings.append(f"加载 Whisper 模型失败: {str(e)}")
        return "", warnings

    # 转录参数
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,                    # 启用 VAD 过滤，提高质量
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    transcript_parts = []
    for segment in segments:
        transcript_parts.append(segment.text.strip())

    transcript = " ".join(transcript_parts).strip()

    if not transcript:
        warnings.append("faster-whisper 未生成有效文本")

    return transcript, warnings


def _transcribe_with_cli(audio_path: Path, work_dir: Path, warnings: List[str]) -> Tuple[str, List[str]]:
    """保留原有 CLI 方式作为降级方案"""
    import shutil
    import subprocess
    import json

    if not shutil.which(settings.whisper_cli):
        warnings.append("Whisper CLI 未安装")
        return "", warnings

    output_dir = work_dir / "whisper"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        settings.whisper_cli,
        str(audio_path),
        "--model", settings.whisper_model,
        "--language", "en",
        "--output_format", "json",
        "--output_dir", str(output_dir),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        warnings.append(f"Whisper CLI 失败: {completed.stderr.strip()}")
        return "", warnings

    json_file = output_dir / f"{audio_path.stem}.json"
    if not json_file.exists():
        warnings.append("未找到 Whisper JSON 输出")
        return "", warnings

    try:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        transcript = payload.get("text", "").strip()
        return transcript, warnings
    except Exception as e:
        warnings.append(f"解析 Whisper 输出失败: {str(e)}")
        return "", warnings
