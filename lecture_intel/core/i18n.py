"""
Interface localization: Chinese (zh) ↔ English (en).

This module is deliberately Qt-free and dependency-free so BOTH the GUI and the
pipeline subprocess can share it. The GUI imports ``t``/``set_language`` and
re-renders its widgets when the language changes; ``core.runner`` sets the
language once at subprocess start (from the ``ui_lang`` setting) so the progress
messages it emits come out in the same language the user picked.

Design notes:
  * Every user-facing string lives in ``_CATALOG`` as ``key -> {"zh":…, "en":…}``.
    Call sites use ``t("key")`` (never a literal), which keeps the two languages
    side by side and makes missing translations obvious.
  * ``t`` supports ``str.format`` placeholders: ``t("status_done", seconds=12)``.
  * The English text is written by hand for this app — not machine translated —
    so it reads naturally rather than mirroring the Chinese word order.
  * Language *names* in the switcher stay in their own script ("中文" / "English")
    regardless of the active UI language, matching the app's existing convention
    for the audio-language picker (see ``core.languages.PICKER_LANGUAGES``).
"""
from __future__ import annotations

import os
from typing import Optional

# ── supported UI languages ────────────────────────────────────────────────
ZH = "zh"
EN = "en"
SUPPORTED: tuple[str, ...] = (ZH, EN)
DEFAULT = ZH

# Native names, shown in the switcher in every language.
_UI_LANGUAGE_NAMES: dict[str, str] = {ZH: "中文", EN: "English"}

_state: dict[str, str] = {"lang": DEFAULT}


def ui_language_choices() -> tuple[tuple[str, str], ...]:
    """(code, native-name) pairs for the language switcher."""
    return tuple((code, _UI_LANGUAGE_NAMES[code]) for code in SUPPORTED)


def detect_system_language() -> str:
    """Best-effort UI language from the OS locale: Chinese → zh, else en."""
    candidates = []
    try:
        import locale
        candidates.append(locale.getlocale()[0])
    except Exception:
        pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        candidates.append(os.environ.get(var))
    for tag in candidates:
        if tag:
            return ZH if str(tag).lower().startswith("zh") else EN
    return EN


def set_language(code: Optional[str]) -> str:
    """Set the active UI language; unknown/empty values fall back to DEFAULT.

    Returns the language actually applied so callers can persist it.
    """
    lang = str(code).strip().lower() if code else ""
    _state["lang"] = lang if lang in SUPPORTED else DEFAULT
    return _state["lang"]


def current_language() -> str:
    return _state["lang"]


def t(key: str, **kwargs) -> str:
    """Translate ``key`` into the active language, then apply ``kwargs``.

    A missing key returns the key itself, so gaps are visible instead of
    silently blank.
    """
    entry = _CATALOG.get(key)
    if entry is None:
        text = key
    else:
        text = entry.get(_state["lang"]) or entry.get(DEFAULT) or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    return text


# ── catalog ───────────────────────────────────────────────────────────────
# Grouped by where the string appears. zh first, en second, on every entry.
_CATALOG: dict[str, dict[str, str]] = {
    # window / branding
    "app_title": {"zh": "Recorder · 录音转文字", "en": "Recorder · Speech to Text"},
    "wordmark": {"zh": "R E C O R D E R  —  全程离线",
                 "en": "R E C O R D E R  —  Fully Offline"},

    # menu bar
    "menu_file": {"zh": "文件", "en": "File"},
    "menu_file_open": {"zh": "打开录音…", "en": "Open Recording…"},
    "menu_file_reveal": {"zh": "显示输出文件夹", "en": "Reveal Output Folder"},
    "menu_view": {"zh": "视图", "en": "View"},
    "menu_view_reset": {"zh": "重置", "en": "Reset"},
    "menu_appearance": {"zh": "外观", "en": "Appearance"},
    "appearance_auto": {"zh": "跟随系统", "en": "Follow System"},
    "appearance_light": {"zh": "浅色", "en": "Light"},
    "appearance_dark": {"zh": "深色", "en": "Dark"},
    "menu_language": {"zh": "语言", "en": "Language"},
    "menu_help": {"zh": "帮助", "en": "Help"},
    "menu_help_docs": {"zh": "打开说明", "en": "Open Documentation"},

    # top strip
    "btn_output_folder": {"zh": "输出文件夹", "en": "Output Folder"},
    "theme_tooltip": {"zh": "点击切换：跟随系统 → 深色 → 浅色",
                      "en": "Click to cycle: Follow System → Dark → Light"},
    "appearance_btn_auto": {"zh": "◐ 外观 · 跟随系统", "en": "◐ Appearance · System"},
    "appearance_btn_dark": {"zh": "● 外观 · 深色", "en": "● Appearance · Dark"},
    "appearance_btn_light": {"zh": "○ 外观 · 浅色", "en": "○ Appearance · Light"},
    "lang_btn": {"zh": "语言 · 中文", "en": "Language · English"},
    "lang_btn_tooltip": {"zh": "点击切换界面语言：中文 ↔ English",
                         "en": "Click to switch the interface language: 中文 ↔ English"},

    # center state label
    "state_processing": {"zh": "处理中", "en": "PROCESSING"},
    "state_done": {"zh": "✓ 转写完成 · 用时 {seconds}s",
                   "en": "✓ Done · took {seconds}s"},
    "state_error": {"zh": "发生错误", "en": "Error"},
    "state_cancelled": {"zh": "已取消", "en": "Cancelled"},

    # status bar
    "status_ready": {"zh": "就绪 · 全程离线", "en": "Ready · Fully offline"},
    "status_recording": {"zh": "录音中…", "en": "Recording…"},
    "status_recorded": {"zh": "已录制：{name}", "en": "Recorded: {name}"},
    "status_processing": {"zh": "处理中…", "en": "Processing…"},
    "status_done": {"zh": "✓ 完成，用时 {seconds}s · 输出：{dir}",
                    "en": "✓ Done in {seconds}s · Output: {dir}"},
    "status_error": {"zh": "发生错误", "en": "An error occurred"},
    "status_cancelled": {"zh": "已取消", "en": "Cancelled"},

    # dialogs / notifications / startup
    "error_box_title": {"zh": "处理出错", "en": "Processing Error"},
    "notify_done": {"zh": "转写完成 — {name}", "en": "Transcription complete — {name}"},
    "worker_crash_code": {"zh": "（退出码 {code}）", "en": " (exit code {code})"},
    "worker_crash": {
        "zh": "处理进程意外退出{code}，很可能是内存不足或模型过大。\n\n"
              "你的录音文件已安全保存，没有丢失。\n"
              "建议在「识别模型」里选「均衡」或「最快」后重试。",
        "en": "The processing process quit unexpectedly{code} — most likely out "
              "of memory, or a model that is too large.\n\n"
              "Your recording is safely on disk and was not lost.\n"
              "Try picking “Balanced” or “Fastest” under Model, then run it again.",
    },
    "deps_title": {"zh": "缺少依赖", "en": "Missing Dependencies"},
    "deps_body": {
        "zh": "无法启动，缺少以下依赖：\n\n{missing}\n\n"
              "请在项目目录运行：\n  uv pip install -r requirements.txt",
        "en": "Can’t start — the following dependencies are missing:\n\n{missing}\n\n"
              "In the project folder, run:\n  uv pip install -r requirements.txt",
    },
    "deps_whisper": {"zh": "mlx-whisper 或 faster-whisper（二者至少装一个）",
                     "en": "mlx-whisper or faster-whisper (install at least one)"},

    # left step rail
    "rail_input": {"zh": "输入", "en": "Input"},
    "rail_mode": {"zh": "模式", "en": "Mode"},
    "rail_transcribe": {"zh": "转写", "en": "Transcribe"},
    "rail_results": {"zh": "结果", "en": "Results"},

    # home screen
    "home_drop_title": {"zh": "把音频拖到这里", "en": "Drop audio here"},
    "home_browse": {"zh": "选择文件…", "en": "Choose File…"},
    "home_record": {"zh": "●  实时录音", "en": "●  Record Live"},
    "home_start": {"zh": "开 始 转 写", "en": "Start Transcription"},
    "home_remove": {"zh": "移除", "en": "Remove"},
    "home_model": {"zh": "识别模型", "en": "Model"},
    "home_language": {"zh": "语言", "en": "Language"},
    "home_lang_tooltip": {
        "zh": "默认自动检测：中英等混说按静音分块逐块识别。指定语言可纠正识别"
              "不稳的音频，但会关闭逐块语种切换。",
        "en": "Auto-detect by default: mixed speech (say, Chinese and English) is "
              "split at silences and recognized chunk by chunk. Pinning a language "
              "helps with unstable audio, but turns off per-chunk language switching.",
    },
    "home_export": {"zh": "导出", "en": "Export"},
    "home_llm": {"zh": "本地大模型增强", "en": "Local LLM enhancement"},
    "home_llm_tooltip": {
        "zh": "中文用中文模型、英文用英文模型（自动判断）。全程本地离线，"
              "未安装 Ollama 时自动跳过。",
        "en": "Uses a Chinese model for Chinese and an English model for English "
              "(decided automatically). Runs fully offline; skipped when Ollama "
              "isn’t installed.",
    },
    "home_browse_title": {"zh": "选择音频文件", "en": "Choose an audio file"},

    # source segmented control
    "source_mic": {"zh": "麦克风", "en": "Microphone"},
    "source_system": {"zh": "电脑声音", "en": "System Audio"},
    "source_both": {"zh": "麦克风＋电脑声音", "en": "Mic + System Audio"},

    # model picker
    "model_large": {"zh": "最准 large-v3", "en": "Most accurate · large-v3"},
    "model_turbo": {"zh": "均衡 large-v3-turbo", "en": "Balanced · large-v3-turbo"},
    "model_small": {"zh": "最快 small", "en": "Fastest · small"},

    # modes (titles reused on the results rail)
    "mode_general_title": {"zh": "通用转写", "en": "General"},
    "mode_classroom_title": {"zh": "课堂录音", "en": "Classroom"},
    "mode_ielts_title": {"zh": "雅思口语教官", "en": "IELTS Coach"},
    "mode_general_desc": {
        "zh": "最高精度、忠实原文，中英混合自动识别",
        "en": "Highest accuracy, faithful to the original; auto-detects mixed "
              "Chinese and English",
    },
    "mode_classroom_desc": {
        "zh": "降噪 · 聚焦主讲人 · 自动提取重点",
        "en": "Noise reduction · Focus on the main speaker · Auto-extract key points",
    },
    "mode_ielts_desc": {
        "zh": "区分教官/考生 · 标注读音与语法疑点",
        "en": "Separates examiner and candidate · Flags pronunciation and grammar "
              "issues",
    },

    # recording screen
    "rec_status_recording": {"zh": "录音中 · {source}", "en": "Recording · {source}"},
    "rec_status_paused": {"zh": "已暂停 · {source}", "en": "Paused · {source}"},
    "rec_pause": {"zh": "❚❚  暂停", "en": "❚❚  Pause"},
    "rec_resume": {"zh": "▶  继续录音", "en": "▶  Resume"},
    "rec_stop": {"zh": "■  停止录音", "en": "■  Stop Recording"},
    "rec_hint_idle": {"zh": "停止后可直接转写", "en": "Transcribe right after you stop"},
    "rec_hint_written": {"zh": "已写入 {mb} MB · 停止后可直接转写",
                         "en": "{mb} MB written · Transcribe right after you stop"},
    "rec_hint_paused": {"zh": "已暂停 · 已写入 {mb} MB", "en": "Paused · {mb} MB written"},
    "rec_dialog_title": {"zh": "录音", "en": "Recording"},
    "rec_error": {"zh": "错误：{err}", "en": "Error: {err}"},
    "rec_sys_missing": {
        "zh": "系统音频组件缺失，请重新运行 make_app.sh 编译后再试。",
        "en": "The system-audio component is missing. Re-run make_app.sh to "
              "compile it, then try again.",
    },
    "rec_sys_perm": {
        "zh": "无法录制电脑声音：需要「屏幕录制」权限。\n\n"
              "请到 系统设置 → 隐私与安全性 → 屏幕录制，勾选 Recorder，"
              "然后重试。\n（首次使用系统会弹出授权请求。）",
        "en": "Can’t capture system audio: Screen Recording permission is required.\n\n"
              "Go to System Settings → Privacy & Security → Screen Recording, "
              "enable Recorder, then try again.\n(The first time, macOS will ask "
              "you for permission.)",
    },
    "rec_sys_fail": {"zh": "录制电脑声音失败：\n{err}",
                     "en": "Failed to capture system audio:\n{err}"},
    "rec_no_audio": {"zh": "没有录到声音。请检查来源或权限后重试。",
                     "en": "No audio was captured. Check the source or permissions "
                           "and try again."},
    "rec_save_title": {"zh": "保存录音", "en": "Save Recording"},
    "rec_save_text": {"zh": "要保存这段录音吗？", "en": "Save this recording?"},
    "rec_save_info": {
        "zh": "保存后会存到 “Recorder/record” 文件夹，方便以后再用。",
        "en": "It will be kept in the “Recorder/record” folder so you can use it "
              "again later.",
    },
    "rec_save_yes": {"zh": "保存", "en": "Save"},
    "rec_save_no": {"zh": "不保存", "en": "Don’t Save"},
    "rec_file_prefix": {"zh": "录音", "en": "recording"},

    # processing screen
    "step_load": {"zh": "加载音频", "en": "Load audio"},
    "step_denoise": {"zh": "降噪处理", "en": "Denoise"},
    "step_asr": {"zh": "语音转写", "en": "Transcribe"},
    "step_diarize": {"zh": "区分说话人", "en": "Separate speakers"},
    "step_analyze": {"zh": "分析 / AI 增强", "en": "Analyze / AI enhance"},
    "step_export": {"zh": "导出结果", "en": "Export results"},
    "proc_preparing": {"zh": "准备中…", "en": "Preparing…"},
    "proc_cancel": {"zh": "取消", "en": "Cancel"},
    "proc_in_progress": {"zh": "进行中", "en": "In progress"},

    # results screen
    "res_transcript": {"zh": "逐字转写", "en": "Transcript"},
    "res_transcript_ielts": {"zh": "逐字转写 · 教官 / 考生",
                             "en": "Transcript · Examiner / Candidate"},
    "res_open_folder": {"zh": "打开输出文件夹", "en": "Open Output Folder"},
    "res_copy": {"zh": "复制全文", "en": "Copy All"},
    "res_new": {"zh": "新的转写", "en": "New Transcription"},
    "res_info": {"zh": "转写信息", "en": "Details"},
    "res_feedback": {"zh": "反馈报告", "en": "Feedback Report"},
    "res_tab_ielts": {"zh": "雅思反馈", "en": "IELTS Feedback"},
    "res_tab_summary": {"zh": "重点总结", "en": "Key Points"},
    "res_tab_summary_ai": {"zh": "重点总结（AI）", "en": "Key Points (AI)"},
    "res_tab_tidy": {"zh": "AI 校对版", "en": "AI Proofread"},
    "res_stat_pron": {"zh": "发音疑点", "en": "Pronunciation"},
    "res_stat_grammar": {"zh": "语法 / 用词", "en": "Grammar / Word choice"},
    "res_stat_wpm": {"zh": "WPM 语速", "en": "WPM"},
    "res_key_mode": {"zh": "模式", "en": "Mode"},
    "res_key_language": {"zh": "语言", "en": "Language"},
    "res_key_duration": {"zh": "时长", "en": "Duration"},
    "res_key_elapsed": {"zh": "转写用时", "en": "Time taken"},
    "res_key_emphasis": {"zh": "提取重点", "en": "Key points"},
    "res_key_definitions": {"zh": "术语定义", "en": "Definitions"},
    "res_section_files": {"zh": "导出文件", "en": "Exported files"},
    "res_output": {"zh": "输出：{dir}", "en": "Output: {dir}"},
    "res_read_error": {"zh": "[无法读取 {path}]", "en": "[Cannot read {path}]"},
    "lang_unknown": {"zh": "未知", "en": "Unknown"},

    # pipeline progress messages (emitted by the subprocess)
    "eng_load_start": {"zh": "加载音频…", "en": "Loading audio…"},
    "eng_load_done": {"zh": "音频已加载", "en": "Audio loaded"},
    "eng_denoise_start": {"zh": "降噪处理…", "en": "Denoising…"},
    "eng_denoise_done": {"zh": "降噪完成", "en": "Denoising complete"},
    "eng_asr_start": {"zh": "转写中（首次会下载模型，请耐心等待）…",
                      "en": "Transcribing (the model downloads on first run — "
                            "please wait)…"},
    "eng_asr_done": {"zh": "转写完成（{count} 段）",
                     "en": "Transcription complete ({count} segments)"},
    "eng_llm_unavailable": {
        "zh": "已勾选本地大模型，但 Ollama 未运行或模型未安装，已回退离线处理。",
        "en": "Local LLM was enabled, but Ollama isn’t running or the model isn’t "
              "installed — falling back to offline processing.",
    },
    "eng_diarize_start": {"zh": "区分说话人…", "en": "Separating speakers…"},
    "eng_diarize_done": {"zh": "识别到 {count} 个说话人", "en": "Found {count} speakers"},
    "eng_main_start": {"zh": "聚焦主讲人，排除旁人…",
                       "en": "Focusing on the main speaker, dropping background voices…"},
    "eng_main_done": {"zh": "已聚焦主讲人", "en": "Focused on the main speaker"},
    "eng_analyze_start": {"zh": "分析发音 / 语法 / 表达…",
                          "en": "Analyzing pronunciation, grammar and phrasing…"},
    "eng_llm_feedback": {"zh": "大模型点评中…", "en": "Generating LLM feedback…"},
    "eng_analyze_done": {"zh": "分析完成", "en": "Analysis complete"},
    "eng_correct_start": {"zh": "AI 根据上课内容校对原文…",
                          "en": "AI is proofreading the transcript against the "
                                "lecture context…"},
    "eng_extract_start": {"zh": "AI 提炼重点…", "en": "AI is extracting key points…"},
    "eng_extract_done": {"zh": "重点提取完成", "en": "Key points extracted"},
    "eng_general_correct": {"zh": "AI 校对全文（提高准确性）…",
                            "en": "AI is proofreading the full transcript for "
                                  "accuracy…"},
    "eng_general_done": {"zh": "整理完成", "en": "Proofreading complete"},
    "eng_export_start": {"zh": "导出结果…", "en": "Exporting results…"},
    "eng_export_done": {"zh": "完成", "en": "Done"},

    # transcriber progress messages
    "tr_load_model": {"zh": "加载 {model} 模型 ({engine})…",
                      "en": "Loading the {model} model ({engine})…"},
    "tr_mlx_fallback": {"zh": "MLX 失败，改用 CPU 引擎…",
                        "en": "MLX failed — switching to the CPU engine…"},
    "tr_done": {"zh": "转写完成", "en": "Transcription complete"},
    "tr_mlx": {"zh": "转写中（MLX 加速）…", "en": "Transcribing (MLX accelerated)…"},
    "tr_gpu_chunk": {"zh": "转写中（GPU 分块）…", "en": "Transcribing (GPU, chunked)…"},
    "tr_cpu": {"zh": "转写中（CPU）…", "en": "Transcribing (CPU)…"},
    "tr_warn_fallback": {"zh": "mlx-whisper 失败，已回退 faster-whisper：{exc}",
                         "en": "mlx-whisper failed; fell back to faster-whisper: {exc}"},

    # local-LLM progress messages
    "llm_proofreading": {"zh": "校对中 {i}/{n}", "en": "Proofreading {i}/{n}"},
    "llm_correcting": {"zh": "AI 校对 {i}/{n}", "en": "AI proofreading {i}/{n}"},
    "llm_extracting": {"zh": "提炼要点 {i}/{n}", "en": "Extracting key points {i}/{n}"},
    "llm_merging": {"zh": "整合总结…", "en": "Assembling the summary…"},
    "llm_tidying": {"zh": "整理中 {i}/{n}", "en": "Tidying up {i}/{n}"},

    # core component errors (surface in dialogs / tracebacks)
    "sys_component_missing": {
        "zh": "系统音频录制组件缺失，请重新运行 make_app.sh 编译。",
        "en": "The system-audio recording component is missing. Re-run "
              "make_app.sh to compile it.",
    },
    "export_textutil_missing": {
        "zh": "textutil 不可用（仅 macOS 支持 .doc 导出）",
        "en": "textutil is unavailable (.doc export is supported on macOS only).",
    },
    "export_textutil_failed": {
        "zh": "textutil 转换失败: {err}",
        "en": "textutil conversion failed: {err}",
    },
}
