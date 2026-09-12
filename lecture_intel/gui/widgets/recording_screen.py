"""
RecordingScreen — live capture stage with the AURA glow ring.

Owns the capture pipeline (ported from the old RecordTab): Qt Multimedia for
the mic, the native ScreenCaptureKit helper for system audio, ffmpeg amix for
"both". Emits recording_ready(path, meta) when a usable file exists, or
recording_aborted() if nothing was captured.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtMultimedia import (
    QAudioDevice,
    QAudioInput,
    QMediaCaptureSession,
    QMediaFormat,
    QMediaRecorder,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.i18n import t
from gui import theme
from gui.widgets.visuals import GlowRing, PulseDot, WaveBars

_SOURCE_LABELS = {"mic": "source_mic", "system": "source_system",
                  "both": "source_both"}


def _format_time(s: int) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


class RecordingScreen(QWidget):
    recording_ready = Signal(str, str)   # path, meta text
    recording_aborted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_recording = False
        self._paused = False
        self._elapsed_s = 0
        self._source = "mic"
        self._mic_path: str | None = None
        self._sys_path: str | None = None
        self._pending_mix = False

        from core.sysaudio import SystemAudioRecorder
        self._sys_rec = SystemAudioRecorder()

        # Qt Multimedia pipeline — WAV · 16 kHz · mono
        self._session = QMediaCaptureSession()
        self._audio_in = QAudioInput()
        self._recorder = QMediaRecorder()
        self._session.setAudioInput(self._audio_in)
        self._session.setRecorder(self._recorder)
        fmt = QMediaFormat()
        fmt.setFileFormat(QMediaFormat.FileFormat.Wave)
        self._recorder.setMediaFormat(fmt)
        self._recorder.setAudioSampleRate(16000)
        self._recorder.setAudioChannelCount(1)
        self._recorder.setAudioBitRate(256000)
        self._recorder.recorderStateChanged.connect(self._on_state_change)
        self._recorder.errorOccurred.connect(self._on_error)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 12, 48, 24)
        root.setSpacing(26)
        root.setAlignment(Qt.AlignCenter)

        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        status_row.setAlignment(Qt.AlignHCenter)
        self._dot = PulseDot("danger")
        self._status = QLabel(t("rec_status_recording", source=t("source_mic")))
        theme.set_tone(self._status, "danger")
        status_row.addWidget(self._dot)
        status_row.addWidget(self._status)
        root.addLayout(status_row)

        self._ring = GlowRing()
        root.addWidget(self._ring, alignment=Qt.AlignHCenter)

        self._bars = WaveBars()
        root.addWidget(self._bars, alignment=Qt.AlignHCenter)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignHCenter)
        self._pause_btn = QPushButton(t("rec_pause"))
        self._pause_btn.setObjectName("ghostPill")
        self._pause_btn.setCursor(Qt.PointingHandCursor)
        self._pause_btn.clicked.connect(self._toggle_pause)
        self._stop_btn = QPushButton(t("rec_stop"))
        self._stop_btn.setObjectName("dangerPill")
        self._stop_btn.setCursor(Qt.PointingHandCursor)
        self._stop_btn.clicked.connect(self.stop)
        btn_row.addWidget(self._pause_btn)
        btn_row.addWidget(self._stop_btn)
        root.addLayout(btn_row)

        self._hint = QLabel(t("rec_hint_idle"))
        self._hint.setProperty("mono", True)
        self._hint.setAlignment(Qt.AlignCenter)
        root.addWidget(self._hint, alignment=Qt.AlignHCenter)

    # ── control ─────────────────────────────────────────────

    def start_capture(self, source: str, device: QAudioDevice | None) -> None:
        self._source = source
        self._status.setText(
            t("rec_status_recording", source=t(_SOURCE_LABELS.get(source, source))))
        theme.set_tone(self._status, "danger")
        self._mic_path = self._sys_path = None
        self._pending_mix = (source == "both")

        # Pausing needs every active capture path to support it. The mic
        # (QMediaRecorder) always does; system audio only if the compiled
        # native helper handles the pause signals (old builds would die on
        # SIGUSR1 instead).
        can_pause = source == "mic" or self._sys_rec.can_pause()
        self._pause_btn.setVisible(can_pause)

        if device is not None:
            self._audio_in.setDevice(device)

        if source in ("system", "both"):
            from core import sysaudio
            if not sysaudio.available():
                self._fail(t("rec_sys_missing"))
                return
            try:
                self._sys_path = self._sys_rec.start()
            except Exception as exc:
                self._fail(str(exc))
                return
            QTimer.singleShot(900, self._check_system_capture)

        if source in ("mic", "both"):
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix="lecture_mic_", delete=False)
            self._mic_path = tmp.name
            tmp.close()
            self._recorder.setOutputLocation(QUrl.fromLocalFile(self._mic_path))
            self._recorder.record()
        else:
            self._enter_recording_ui()

    def _toggle_pause(self) -> None:
        if not self._is_recording:
            return
        label = t(_SOURCE_LABELS.get(self._source, self._source))
        if not self._paused:
            self._paused = True
            if self._source in ("mic", "both"):
                self._recorder.pause()
            if self._source in ("system", "both") and hasattr(self._sys_rec, "pause"):
                self._sys_rec.pause()
            self._timer.stop()
            self._dot.stop()
            self._bars.stop()
            self._pause_btn.setText(t("rec_resume"))
            self._status.setText(t("rec_status_paused", source=label))
            theme.set_tone(self._status, "hint")
            self._hint.setText(
                t("rec_hint_paused", mb=f"{self._elapsed_s * 0.031:.1f}"))
        else:
            self._paused = False
            if self._source in ("mic", "both"):
                self._recorder.record()
            if self._source in ("system", "both") and hasattr(self._sys_rec, "resume"):
                self._sys_rec.resume()
            self._timer.start()
            self._dot.start()
            self._bars.start()
            self._pause_btn.setText(t("rec_pause"))
            self._status.setText(t("rec_status_recording", source=label))
            theme.set_tone(self._status, "danger")

    def stop(self) -> None:
        if not self._is_recording:
            return
        if self._source == "system":
            self._exit_recording_ui()
            self._finalize(self._sys_rec.stop())
        else:
            if self._source == "both":
                self._sys_path = self._sys_rec.stop()
            self._recorder.stop()

    # ── internals (ported from RecordTab) ───────────────────

    def _check_system_capture(self) -> None:
        if not self._is_recording:
            return
        if not self._sys_rec.is_running:
            err = self._sys_rec.error_text()
            self._abort()
            if "TCC" in err or "拒絕" in err or "denied" in err.lower() or not err:
                self._fail(t("rec_sys_perm"))
            else:
                self._fail(t("rec_sys_fail", err=err[:200]))

    def _on_state_change(self, state: QMediaRecorder.RecorderState) -> None:
        if state == QMediaRecorder.RecorderState.RecordingState:
            if not self._is_recording:      # resume from pause re-enters this state
                self._enter_recording_ui()
        elif state == QMediaRecorder.RecorderState.StoppedState:
            if not self._is_recording:
                return
            self._exit_recording_ui()
            if self._pending_mix:
                self._finalize(self._mix(self._mic_path, self._sys_path))
            else:
                self._finalize(self._mic_path)

    def _mix(self, mic: str | None, sysp: str | None) -> str | None:
        import shutil
        import subprocess
        have = [p for p in (mic, sysp)
                if p and Path(p).exists() and Path(p).stat().st_size > 1024]
        if not have:
            return None
        if len(have) == 1:
            return have[0]
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        out = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="lecture_mix_", delete=False)
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
        return have[0]

    def _finalize(self, path: str | None) -> None:
        if not path or not Path(path).exists() or Path(path).stat().st_size <= 1024:
            self._fail(t("rec_no_audio"))
            return
        path = self._maybe_save_recording(path)
        size_mb = Path(path).stat().st_size / 1_048_576
        meta = f"{_format_time(self._elapsed_s)} · {size_mb:.1f} MB · 16 kHz WAV"
        self.recording_ready.emit(path, meta)

    def _maybe_save_recording(self, temp_path: str) -> str:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(t("rec_save_title"))
        box.setText(t("rec_save_text"))
        box.setInformativeText(t("rec_save_info"))
        save_btn = box.addButton(t("rec_save_yes"), QMessageBox.AcceptRole)
        box.addButton(t("rec_save_no"), QMessageBox.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        if box.clickedButton() is not save_btn:
            return temp_path
        import shutil
        from datetime import datetime
        from core.paths import record_dir
        dest = record_dir() / (
            f"{t('rec_file_prefix')}_{datetime.now():%Y%m%d_%H%M%S}.wav")
        try:
            shutil.copy2(temp_path, dest)
            return str(dest)
        except Exception:
            return temp_path

    def _enter_recording_ui(self) -> None:
        self._is_recording = True
        self._paused = False
        self._pause_btn.setText(t("rec_pause"))
        self._elapsed_s = 0
        self._ring.set_time("00:00:00")
        self._hint.setText(t("rec_hint_idle"))
        self._timer.start()
        self._dot.start()
        self._bars.start()

    def _exit_recording_ui(self) -> None:
        self._is_recording = False
        self._timer.stop()
        self._dot.stop()
        self._bars.stop()

    def _abort(self) -> None:
        try:
            if self._recorder.recorderState() != QMediaRecorder.RecorderState.StoppedState:
                self._recorder.stop()
        except Exception:
            pass
        self._sys_rec.stop()
        self._exit_recording_ui()

    def abort(self) -> None:
        """External cancel (e.g. window reset)."""
        if self._is_recording:
            self._abort()
        self.recording_aborted.emit()

    def _fail(self, msg: str) -> None:
        self._abort()
        QMessageBox.warning(self, t("rec_dialog_title"), msg)
        self.recording_aborted.emit()

    def _on_error(self, _error, error_string: str) -> None:
        self._exit_recording_ui()
        QMessageBox.warning(self, t("rec_dialog_title"),
                            t("rec_error", err=error_string))
        self.recording_aborted.emit()

    def _tick(self) -> None:
        self._elapsed_s += 1
        self._ring.set_time(_format_time(self._elapsed_s))
        mb = self._elapsed_s * 0.031
        self._hint.setText(t("rec_hint_written", mb=f"{mb:.1f}"))

    # ── live language switch ─────────────────────────

    def _source_text(self) -> str:
        return t(_SOURCE_LABELS.get(self._source, self._source))

    def _render_status(self) -> None:
        if self._paused:
            self._status.setText(t("rec_status_paused", source=self._source_text()))
            theme.set_tone(self._status, "hint")
        else:
            self._status.setText(
                t("rec_status_recording", source=self._source_text()))
            theme.set_tone(self._status, "danger")

    def _render_hint(self) -> None:
        mb = f"{self._elapsed_s * 0.031:.1f}"
        if self._paused:
            self._hint.setText(t("rec_hint_paused", mb=mb))
        elif self._is_recording and self._elapsed_s > 0:
            self._hint.setText(t("rec_hint_written", mb=mb))
        else:
            self._hint.setText(t("rec_hint_idle"))

    def retranslate(self) -> None:
        self._stop_btn.setText(t("rec_stop"))
        self._pause_btn.setText(t("rec_resume") if self._paused else t("rec_pause"))
        self._render_status()
        self._render_hint()
