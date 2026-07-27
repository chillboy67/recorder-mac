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

        # Microphone picker
        mic_row = QHBoxLayout()
        mic_label = QLabel("麦克风：")
        mic_label.setFixedWidth(90)
        self._mic_combo = QComboBox()
        self._mic_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        mic_row.addWidget(mic_label)
        mic_row.addWidget(self._mic_combo)
        layout.addLayout(mic_row)

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

    # ── Recording control ───────────────────────────────────

    def _toggle_recording(self) -> None:
        if not self._is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="lecture_", delete=False,
        )
        self._temp_path = tmp.name
        tmp.close()

        self._recorder.setOutputLocation(QUrl.fromLocalFile(self._temp_path))
        self._recorder.record()

    def _stop_recording(self) -> None:
        self._recorder.stop()

    def _on_state_change(self, state: QMediaRecorder.RecorderState) -> None:
        if state == QMediaRecorder.RecorderState.RecordingState:
            self._is_recording = True
            self._elapsed_s = 0
            self._timer.start()
            self._rec_btn.setText("⏹  停止录音")
            self._rec_btn.setStyleSheet(_STYLE_RECORDING)
            self._status_label.setText("录音中…")
            self._status_label.setStyleSheet(
                "color: #FF3B30; font-size: 12px; font-weight: 500;"
            )
            self._file_label.setVisible(False)

        elif state == QMediaRecorder.RecorderState.StoppedState:
            self._is_recording = False
            self._timer.stop()
            self._rec_btn.setText("⏺  开始录音")
            self._rec_btn.setStyleSheet(_STYLE_IDLE)

            if self._temp_path and Path(self._temp_path).exists():
                size_mb = Path(self._temp_path).stat().st_size / 1_048_576
                dur_str = _format_time(self._elapsed_s)
                self._status_label.setText("录音已保存")
                self._status_label.setStyleSheet(
                    "color: #34C759; font-size: 12px;"
                )
                self._file_label.setText(
                    f"已保存：{dur_str}  ·  {size_mb:.1f} MB"
                )
                self._file_label.setVisible(True)
                self.recording_ready.emit(self._temp_path)

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
            self._stop_recording()
        self._temp_path = None
        self._elapsed_s = 0
        self._time_label.setText("00:00:00")
        self._status_label.setText("准备就绪")
        self._status_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        self._file_label.setVisible(False)
        self._rec_btn.setText("⏺  开始录音")
        self._rec_btn.setStyleSheet(_STYLE_IDLE)


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
            "🎵\n\nDrag & drop your recording here\n\n"
            "m4a · mp3 · wav · flac · aac"
        )
        self._drop_area.setAlignment(Qt.AlignCenter)
        self._drop_area.setMinimumHeight(120)
        self._reset_drop_style()
        layout.addWidget(self._drop_area)

        # Independent Browse button (avoids mousePressEvent pitfalls)
        self._browse_btn = QPushButton("Browse Files...")
        self._browse_btn.setFixedHeight(36)
        self._browse_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #007AFF;
                border: 1.5px solid #007AFF;
                border-radius: 8px;
                font-size: 13px;
            }
            QPushButton:hover   { background: rgba(0,122,255,0.08); }
            QPushButton:pressed { background: rgba(0,122,255,0.15); }
        """)
        self._browse_btn.clicked.connect(self._browse)
        layout.addWidget(self._browse_btn)

        # Selected file info
        self._file_label = QLabel("")
        self._file_label.setAlignment(Qt.AlignCenter)
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: #007AFF; font-size: 12px;")
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
                            border: 2px dashed #007AFF;
                            border-radius: 12px;
                            background: rgba(0,122,255,0.05);
                            color: #007AFF; font-size: 13px;
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
        color = "#34C759" if selected else "#C7C7CC"
        bg = "rgba(52,199,89,0.04)" if selected else "rgba(0,0,0,0.02)"
        self._drop_area.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {color};
                border-radius: 12px;
                background: {bg};
                color: #8E8E93; font-size: 13px;
            }}
        """)

    def _set_file(self, path: str) -> None:
        self._selected_path = path
        p = Path(path)
        size_mb = p.stat().st_size / 1_048_576
        self._file_label.setText(f"✓  {p.name}\n{size_mb:.1f} MB")
        self._file_label.setVisible(True)
        self._drop_area.setText(f"✓ File ready\n\n{p.name}")
        self._reset_drop_style()
        self.file_selected.emit(path)

    @property
    def selected_path(self) -> str | None:
        return self._selected_path

    def reset(self) -> None:
        self._selected_path = None
        self._file_label.setVisible(False)
        self._drop_area.setText(
            "🎵\n\nDrag & drop your recording here\n\n"
            "m4a · mp3 · wav · flac · aac"
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

        self._tabs.addTab(self._record_tab, "🎙 Record")
        self._tabs.addTab(self._file_tab, "📁 File")

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
        background: #007AFF; color: white;
        border-radius: 10px; font-size: 15px; font-weight: 500;
    }
    QPushButton:hover   { background: #0066DD; }
    QPushButton:pressed { background: #0055BB; }
"""

_STYLE_RECORDING = """
    QPushButton {
        background: #FF3B30; color: white;
        border-radius: 10px; font-size: 15px; font-weight: 500;
    }
    QPushButton:hover   { background: #E02020; }
    QPushButton:pressed { background: #CC1111; }
"""
