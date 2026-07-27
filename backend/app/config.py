from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import os
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MFA_CLI = "/opt/homebrew/Caskroom/miniforge/base/envs/mfa/bin/mfa"
DETECTED_MFA_CLI = shutil.which("mfa") or (DEFAULT_MFA_CLI if Path(DEFAULT_MFA_CLI).exists() else "mfa")
MFA_INSTALLED = bool(shutil.which(DETECTED_MFA_CLI) or Path(DETECTED_MFA_CLI).exists())


class Settings(BaseModel):
    data_dir: Path = Path(os.getenv("IELTS_COACH_DATA_DIR", PROJECT_ROOT / "data")).resolve()

    # ==================== 第一阶段修改 ====================
    # Whisper 配置
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")          # 默认从 base 改为 small
    use_faster_whisper: bool = os.getenv("USE_FASTER_WHISPER", "true").lower() == "true"

    # 保留原有 CLI 配置（兼容旧方式）
    whisper_cli: str = os.getenv("WHISPER_CLI", "whisper")

    # LanguageTool & MFA 配置保持不变
    languagetool_url: str = os.getenv("LANGUAGETOOL_URL", "http://127.0.0.1:8010/v2/check")
    mfa_enabled: bool = os.getenv("MFA_ENABLED", "true" if MFA_INSTALLED else "false").lower() == "true"
    mfa_cli: str = os.getenv("MFA_CLI", DETECTED_MFA_CLI)
    mfa_dictionary_path: Optional[str] = os.getenv("MFA_DICTIONARY_PATH", "english_us_arpa" if MFA_INSTALLED else None)
    mfa_acoustic_model: Optional[str] = os.getenv("MFA_ACOUSTIC_MODEL", "english_us_arpa" if MFA_INSTALLED else None)


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)