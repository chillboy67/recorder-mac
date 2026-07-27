"""
InputPanel: Tab-switched input — "Record" for live microphone capture,
"File" for drag-and-drop or browse.

Dependencies:
    PySide6.QtMultimedia (bundled with PySide6, no extra install)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import (
    QAudioDevice,
    QAudioInput,
    QMediaCaptureSession,
    QMediaDevices,
    QMediaFormat,
    QMediaRecorder,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

SUPPORTED_EXTENSIONS: set[str] = {
    ".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".webm",
}


# ═══════════════════════════════════════════════════════════════
# Tab 1 — Live Recording
# ═══════════════════════════════════════════════════════════════

class RecordTab(QWidget):
    """Live microphone recording to a temporary .wav file.

    Emits ``recording_ready(path)`` when recording stops successfully.
    """

    recording_ready = Signal(str)  # absolute path to temp .wav

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_recording = False
        self._elapsed_s = 0
        self._temp_path: str | None = None
        # system-audio (loopback) capture via the native ScreenCaptureKit helper
        from core.sysaudio import SystemAudioRecorder
        self._sys_rec = SystemAudioRecorder()
        self._mic_path: str | None = None
        self._sys_path: str | None = None
        self._pending_mix = False

        # -- Qt Multimedia pipeline --
        self._session = QMediaCaptureSession()
        self._audio_in = QAudioInput()
        self._recorder = QMediaRecorder()
        self._session.setAudioInput(self._audio_in)
        self._session.setRecorder(self._recorder)

        # WAV · 16 kHz · mono (matches pipeline expectations)
        fmt = QMediaFormat()
        fmt.setFileFormat(QMediaFormat.FileFormat.Wave)
        self._recorder.setMediaFormat(fmt)
        self._recorder.setAudioSampleRate(16000)
        self._recorder.setAudioChannelCount(1)
        self._recorder.setAudioBitRate(256000)

        self._recorder.recorderStateChanged.connect(self._on_state_change)
        self._recorder.errorOccurred.connect(self._on_error)

        # -- One-second timer for the on-screen clock --
        self._timer = QTimer()
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._refresh_devices()

    # ── UI ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Source picker: mic / system audio / both
        src_row = QHBoxLayout()
        src_label = QLabel("录音来源：")
        src_label.setFixedWidth(72)
        self._source_combo = QComboBox()
        self._source_combo.addItem("麦克风（外界声音）", userData="mic")
        self._source_combo.addItem("电脑声音（内部播放）", userData="system")
        self._source_combo.addItem("麦克风 + 电脑声音", userData="both")
        self._source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._source_combo.setMinimumContentsLength(6)
        self._source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_row.addWidget(src_label)
        src_row.addWidget(self._source_combo)
        layout.addLayout(src_row)

        # Microphone picker (hidden when source is "system")
        self._mic_widget = QWidget()
        mic_row = QHBoxLayout(self._mic_widget)
        mic_row.setContentsMargins(0, 0, 0, 0)
        mic_label = QLabel("麦克风：")
        mic_label.setFixedWidth(72)
        self._mic_combo = QComboBox()
        self._mic_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Don't let a long device name force the whole panel wide — cap the
        # width it asks for; it still expands to fill the available space.
        self._mic_combo.setMinimumContentsLength(6)
        self._mic_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        mic_row.addWidget(mic_label)
        mic_row.addWidget(self._mic_combo)
        layout.addWidget(self._mic_widget)

        # Clock display
        self._time_label = QLabel("00:00:00")
        self._time_label.setAlignment(Qt.AlignCenter)
        self._time_label.setStyleSheet(
            "font-size: 36px; font-weight: 200; color: #1C1C1E;"
            " letter-spacing: 4px;"
        )
        layout.addWidget(self._time_label)

        # Status text
        self._status_label = QLabel("准备就绪")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        layout.addWidget(self._status_label)

        # Record / Stop button
        self._rec_btn = QPushButton("⏺  开始录音")
        self._rec_btn.setFixedHeight(48)
        self._rec_btn.setStyleSheet(_STYLE_IDLE)
        self._rec_btn.clicked.connect(self._toggle_recording)
        layout.addWidget(self._rec_btn)

        # Saved-file info (hidden until recording stops)
        self._file_label = QLabel("")
        self._file_label.setAlignment(Qt.AlignCenter)
        self._file_label.setStyleSheet("color: #34C759; font-size: 11px;")
        self._file_label.setVisible(False)
        layout.addWidget(self._file_label)

        layout.addStretch()

    # ── Device discovery ────────────────────────────────────

    def _refresh_devices(self) -> None:
        self._mic_combo.blockSignals(True)
        self._mic_combo.clear()
        devices = QMediaDevices.audioInputs()
        default = QMediaDevices.defaultAudioInput()
        default_idx = 0
        for i, dev in enumerate(devices):
            self._mic_combo.addItem(dev.description(), userData=dev)
            if dev.id() == default.id():
                default_idx = i
        self._mic_combo.setCurrentIndex(default_idx)
        self._mic_combo.blockSignals(False)
        self._on_mic_changed(default_idx)

    def _on_mic_changed(self, idx: int) -> None:
        dev: QAudioDevice = self._mic_combo.itemData(idx)
        if dev:
            self._audio_in.setDevice(dev)

    def _current_source(self) -> str:
        return self._source_combo.currentData() or "mic"

    def _on_source_changed(self, *_) -> None:
        # The mic picker is only relevant when the mic is involved.
        self._mic_widget.setVisible(self._current_source() in ("mic", "both"))

    # ── Recording control ───────────────────────────────────

    def _toggle_recording(self) -> None:
        if not self._is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        src = self._current_source()
        self._mic_path = self._sys_path = None
        self._pending_mix = (src == "both")

        # System audio (loopback) via the native helper
        if src in ("system", "both"):
            from core import sysaudio
            if not sysaudio.available():
                self._show_error("系统音频组件缺失，请重新运行 make_app.sh 编译后再试。")
                return
            try:
                self._sys_path = self._sys_rec.start()
            except Exception as exc:
                self._show_error(str(exc))
                return
            # The helper may be denied Screen Recording permission — it exits
            # almost immediately in that case. Check shortly after starting.
            QTimer.singleShot(900, self._check_system_capture)

        # Microphone via Qt (drives the UI via recorderStateChanged)
        if src in ("mic", "both"):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="lecture_mic_", delete=False)
            self._mic_path = tmp.name
            tmp.close()
            self._recorder.setOutputLocation(QUrl.fromLocalFile(self._mic_path))
            self._recorder.record()
        else:
            self._enter_recording_ui()   # system-only: no Qt recorder event

        self._source_combo.setEnabled(False)

    def _check_system_capture(self) -> None:
        if not self._is_recording:
            return
        if not self._sys_rec.is_running:
            # helper died — almost always a missing Screen Recording permission
            err = self._sys_rec.error_text()
            self._abort_recording()
            if "TCC" in err or "拒絕" in err or "denied" in err.lower() or not err:
                self._show_error(
                    "无法录制电脑声音：需要「屏幕录制」权限。\n\n"
                    "请到 系统设置 → 隐私与安全性 → 屏幕录制，勾选 Recorder，然后重试。\n"
                    "（首次使用系统会弹出授权请求。）")
            else:
                self._show_error("录制电脑声音失败：\n" + err[:200])

    def _stop_recording(self) -> None:
        src = self._current_source()
        if src == "system":
            self._exit_recording_ui()
            self._source_combo.setEnabled(True)
            self._finalize(self._sys_rec.stop())
        else:
            # mic or both: stopping the Qt recorder triggers _on_state_change,
            # which finalizes (and mixes in the system track for "both").
            if src == "both":
                self._sys_path = self._sys_rec.stop()
            self._recorder.stop()

    def _enter_recording_ui(self) -> None:
        self._is_recording = True
        self._elapsed_s = 0
        self._timer.start()
        self._rec_btn.setText("⏹  停止录音")
        self._rec_btn.setStyleSheet(_STYLE_RECORDING)
        self._status_label.setText("录音中…")
        self._status_label.setStyleSheet(
            "color: #FF3B30; font-size: 12px; font-weight: 500;")
        self._file_label.setVisible(False)

    def _exit_recording_ui(self) -> None:
        self._is_recording = False
        self._timer.stop()
        self._rec_btn.setText("⏺  开始录音")
        self._rec_btn.setStyleSheet(_STYLE_IDLE)

    def _abort_recording(self) -> None:
        try:
            if self._recorder.recorderState() != QMediaRecorder.RecorderState.StoppedState:
                self._recorder.stop()
        except Exception:
            pass
        self._sys_rec.stop()
        self._exit_recording_ui()
        self._source_combo.setEnabled(True)

    def _on_state_change(self, state: QMediaRecorder.RecorderState) -> None:
        if state == QMediaRecorder.RecorderState.RecordingState:
            self._enter_recording_ui()
        elif state == QMediaRecorder.RecorderState.StoppedState:
            if not self._is_recording:
                return
            self._exit_recording_ui()
            self._source_combo.setEnabled(True)
            if self._pending_mix:
                self._finalize(self._mix(self._mic_path, self._sys_path))
            else:
                self._finalize(self._mic_path)

    def _mix(self, mic: str | None, sysp: str | None) -> str | None:
        """Mix mic + system tracks into one wav via ffmpeg amix; fall back to
        whichever single track exists."""
        import shutil
        import subprocess
        have = [p for p in (mic, sysp) if p and Path(p).exists() and Path(p).stat().st_size > 1024]
        if not have:
            return None
        if len(have) == 1:
            return have[0]
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        out = tempfile.NamedTemporaryFile(suffix=".wav", prefix="lecture_mix_", delete=False)
        out.close()
        cmd = [ffmpeg, "-y", "-i", have[0], "-i", have[1],
               "-filter_complex", "amix=inputs=2:duration=longest:normalize=0",
               "-ac", "1", "-ar", "16000", out.name]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and Path(out.name).stat().st_size > 1024:
                return out.name
        except Exception:
            pass
        return have[0]   # mixing failed → at least return the mic track

    def _finalize(self, path: str | None) -> None:
        if not path or not Path(path).exists() or Path(path).stat().st_size <= 1024:
            self._show_error("没有录到声音。请检查来源或权限后重试。")
            return
        self._temp_path = self._maybe_save_recording(path)
        size_mb = Path(self._temp_path).stat().st_size / 1_048_576
        dur_str = _format_time(self._elapsed_s)
        self._status_label.setText("录音已就绪")
        self._status_label.setStyleSheet("color: #34C759; font-size: 12px;")
        self._file_label.setText(f"{dur_str}  ·  {size_mb:.1f} MB")
        self._file_label.setVisible(True)
        self.recording_ready.emit(self._temp_path)

    def _show_error(self, msg: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        self._exit_recording_ui()
        self._source_combo.setEnabled(True)
        self._status_label.setText("未开始")
        self._status_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        QMessageBox.warning(self, "录音", msg)

    def _maybe_save_recording(self, temp_path: str) -> str:
        """Ask whether to keep this recording; if yes, copy it to ~/Recorder/record/.

        Returns the path to use afterwards (the saved copy, or the temp file)."""
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("保存录音")
        box.setText("要保存这段录音吗？")
        box.setInformativeText("保存后会存到 “Recorder/record” 文件夹，方便以后再用。")
        save_btn = box.addButton("保存", QMessageBox.AcceptRole)
        box.addButton("不保存", QMessageBox.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        if box.clickedButton() is not save_btn:
            return temp_path

        import shutil
        from datetime import datetime
        dest_dir = Path.home() / "Recorder" / "record"
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = f"录音_{datetime.now():%Y%m%d_%H%M%S}.wav"
        dest = dest_dir / name
        try:
            shutil.copy2(temp_path, dest)
            return str(dest)
        except Exception:
            return temp_path

    def _on_error(self, error: QMediaRecorder.Error, error_string: str) -> None:
        self._status_label.setText(f"错误：{error_string}")
        self._status_label.setStyleSheet("color: #FF3B30; font-size: 12px;")
        self._is_recording = False
        self._timer.stop()
        self._rec_btn.setText("⏺  开始录音")
        self._rec_btn.setStyleSheet(_STYLE_IDLE)

    # ── Clock ───────────────────────────────────────────────

    def _tick(self) -> None:
        self._elapsed_s += 1
        self._time_label.setText(_format_time(self._elapsed_s))

    # ── Public properties ───────────────────────────────────

    @property
    def selected_path(self) -> str | None:
        if self._temp_path and Path(self._temp_path).exists():
            return self._temp_path
        return None

    def reset(self) -> None:
        if self._is_recording:
            self._abort_recording()
        self._temp_path = None
        self._elapsed_s = 0
        self._time_label.setText("00:00:00")
        self._status_label.setText("准备就绪")
        self._status_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        self._file_label.setVisible(False)
        self._rec_btn.setText("⏺  开始录音")
        self._rec_btn.setStyleSheet(_STYLE_IDLE)
        self._source_combo.setEnabled(True)


# ═══════════════════════════════════════════════════════════════
# Tab 2 — File selection
# ═══════════════════════════════════════════════════════════════

class FileTab(QWidget):
    """Drag-and-drop area + independent Browse button for audio files."""

    file_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._selected_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Drop zone (display only — no click handling)
        self._drop_area = QLabel(
            "🎵\n\n把音频拖到这里\n\n"
            "m4a · mp3 · wav · webm · flac · aac"
        )
        self._drop_area.setAlignment(Qt.AlignCenter)
        self._drop_area.setMinimumHeight(120)
        self._reset_drop_style()
        layout.addWidget(self._drop_area)

        # Independent Browse button (avoids mousePressEvent pitfalls)
        self._browse_btn = QPushButton("选择文件…")
        self._browse_btn.setFixedHeight(40)
        self._browse_btn.setCursor(Qt.PointingHandCursor)
        self._browse_btn.setStyleSheet("""
            QPushButton {
                background: #EAF3FF; color: #0A84FF;
                border: 1px solid #CFE4FF; border-radius: 10px;
                font-size: 13px; font-weight: 600;
            }
            QPushButton:hover   { background: #DCEBFF; }
            QPushButton:pressed { background: #CFE4FF; }
        """)
        self._browse_btn.clicked.connect(self._browse)
        layout.addWidget(self._browse_btn)

        # Selected file info
        self._file_label = QLabel("")
        self._file_label.setAlignment(Qt.AlignCenter)
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: #0A84FF; font-size: 12px; font-weight: 600;")
        self._file_label.setVisible(False)
        layout.addWidget(self._file_label)

        layout.addStretch()

    # ── Browse ──────────────────────────────────────────────

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio Recording",
            str(Path.home() / "Desktop"),
            "Audio Files (*.m4a *.mp3 *.wav *.flac *.aac *.ogg *.opus *.webm);;"
            "All Files (*)",
        )
        if path:
            self._set_file(path)

    # ── Drag & drop ─────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    self._drop_area.setStyleSheet("""
                        QLabel {
                            border: 2px dashed #0A84FF;
                            border-radius: 14px;
                            background: #EAF3FF;
                            color: #0A84FF; font-size: 13px;
                        }
                    """)
                    return
        event.ignore()

    def dragLeaveEvent(self, event: object) -> None:
        self._reset_drop_style()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
                self._set_file(path)
                break
        self._reset_drop_style()

    def _reset_drop_style(self) -> None:
        selected = self._selected_path is not None
        color = "#34C759" if selected else "#D2D3D9"
        bg = "#F0FBF3" if selected else "#FAFBFC"
        text = "#34C759" if selected else "#9A9AA2"
        self._drop_area.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {color};
                border-radius: 14px;
                background: {bg};
                color: {text}; font-size: 13px;
            }}
        """)

    def _set_file(self, path: str) -> None:
        self._selected_path = path
        p = Path(path)
        size_mb = p.stat().st_size / 1_048_576
        self._file_label.setText(f"✓  {p.name}\n{size_mb:.1f} MB")
        self._file_label.setVisible(True)
        self._drop_area.setText(f"✓ 已选择\n\n{p.name}")
        self._reset_drop_style()
        self.file_selected.emit(path)

    @property
    def selected_path(self) -> str | None:
        return self._selected_path

    def reset(self) -> None:
        self._selected_path = None
        self._file_label.setVisible(False)
        self._drop_area.setText(
            "🎵\n\n把音频拖到这里\n\n"
            "m4a · mp3 · wav · webm · flac · aac"
        )
        self._reset_drop_style()


# ═══════════════════════════════════════════════════════════════
# Combined InputPanel — public API for main_window
# ═══════════════════════════════════════════════════════════════

class InputPanel(QWidget):
    """Tabbed input: Record (mic) | File (drag/browse).

    Public API (drop-in replacement for DropZone):
        .selected_path   → str | None
        .file_ready      → Signal(str)   emitted when a file is available
        .reset()         → clear both tabs
    """

    file_ready = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._record_tab = RecordTab()
        self._file_tab = FileTab()

        self._tabs.addTab(self._record_tab, "🎙 录音")
        self._tabs.addTab(self._file_tab, "📁 文件")

        self._record_tab.recording_ready.connect(self.file_ready)
        self._file_tab.file_selected.connect(self.file_ready)

        layout.addWidget(self._tabs)

    @property
    def selected_path(self) -> str | None:
        if self._tabs.currentIndex() == 0:
            return self._record_tab.selected_path
        return self._file_tab.selected_path

    def reset(self) -> None:
        self._record_tab.reset()
        self._file_tab.reset()


# ═══════════════════════════════════════════════════════════════
# Shared helpers & styles
# ═══════════════════════════════════════════════════════════════

def _format_time(s: int) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


_STYLE_IDLE = """
    QPushButton {
        background: #0A84FF; color: white; border: none;
        border-radius: 12px; font-size: 15px; font-weight: 600;
    }
    QPushButton:hover   { background: #0066D6; }
    QPushButton:pressed { background: #0059BE; }
"""

_STYLE_RECORDING = """
    QPushButton {
        background: #FF3B30; color: white; border: none;
        border-radius: 12px; font-size: 15px; font-weight: 600;
    }
    QPushButton:hover   { background: #E0271D; }
    QPushButton:pressed { background: #C71F16; }
"""
