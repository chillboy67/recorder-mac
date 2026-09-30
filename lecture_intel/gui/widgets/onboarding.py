"""
First-run strip on the home screen: the three steps, and a microphone check.

The check listens to the selected microphone for a few seconds and says
whether any sound arrived. It keeps nothing and loads no model, so a new user
can confirm the mic (and the macOS permission) before the first transcription
starts a multi-gigabyte model download. "Got it" hides the strip for good;
Help → Show Getting Started brings it back.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.i18n import t
from gui import theme

CHECK_SECONDS = 3.0
# Peak below this (about −50 dBFS) is treated as "nothing arrived": a working
# mic in a quiet room still picks up breathing and room tone above it, while a
# muted, denied or wrong device delivers digital silence.
SILENT_PEAK = 0.003


def peak_level(data: bytes, sample_format) -> float:
    """Peak amplitude (0‥1) of a raw PCM buffer."""
    import numpy as np

    formats = {
        QAudioFormat.SampleFormat.Int16: (np.int16, 32768.0, 0.0),
        QAudioFormat.SampleFormat.Int32: (np.int32, 2147483648.0, 0.0),
        QAudioFormat.SampleFormat.Float: (np.float32, 1.0, 0.0),
        QAudioFormat.SampleFormat.UInt8: (np.uint8, 128.0, 128.0),
    }
    dtype, scale, offset = formats.get(sample_format, (np.int16, 32768.0, 0.0))
    usable = len(data) - len(data) % np.dtype(dtype).itemsize
    if usable <= 0:
        return 0.0
    samples = np.frombuffer(data[:usable], dtype=dtype).astype(np.float64)
    return float(min(1.0, np.max(np.abs(samples - offset)) / scale))


def to_dbfs(peak: float) -> float:
    return 20 * math.log10(peak) if peak > 0 else -math.inf


class MicCapture(QObject):
    """Listen to one input device for a few seconds; nothing is written."""

    level = Signal(float)      # running peak, 0‥1
    finished = Signal(float)   # overall peak
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._source: Optional[QAudioSource] = None
        self._io = None
        self._peak = 0.0
        self._format = None

    def start(self, device, seconds: float = CHECK_SECONDS) -> None:
        device = device if device is not None and not device.isNull() \
            else QMediaDevices.defaultAudioInput()
        if device.isNull():
            self.failed.emit(t("onb_mic_none"))
            return
        fmt = device.preferredFormat()
        self._format = fmt.sampleFormat()
        self._peak = 0.0
        self._source = QAudioSource(device, fmt, self)
        self._io = self._source.start()
        if self._io is None:
            self.failed.emit(t("onb_mic_failed"))
            return
        self._io.readyRead.connect(self._read)
        QTimer.singleShot(int(seconds * 1000), self._stop)

    def _read(self) -> None:
        if self._io is None:
            return
        chunk = bytes(self._io.readAll())
        self._peak = max(self._peak, peak_level(chunk, self._format))
        self.level.emit(self._peak)

    def _stop(self) -> None:
        if self._source is None:
            return
        self._read()
        self._source.stop()
        self._source = None
        self._io = None
        self.finished.emit(self._peak)


class LevelMeter(QWidget):
    """A thin bar showing the loudest level heard so far (theme colours)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self.setFixedSize(90, 6)

    def set_level(self, level: float) -> None:
        # dB scale so speech at a normal distance fills a visible share
        db = to_dbfs(level)
        self._level = 0.0 if db == -math.inf else max(0.0, min(1.0, (db + 60) / 60))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        from PySide6.QtGui import QColor
        c = theme.current_scheme()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["line"]))
        r = QRectF(0, 0, self.width(), self.height())
        p.drawRoundedRect(r, 3, 3)
        if self._level > 0:
            p.setBrush(QColor(c["accent"]))
            p.drawRoundedRect(QRectF(0, 0, self.width() * self._level, self.height()), 3, 3)
        p.end()


class OnboardingStrip(QFrame):
    """The first-run strip. ``device_provider`` returns the selected mic."""

    dismissed = Signal()

    def __init__(self, device_provider: Callable[[], object],
                 capture_factory: Callable[[QObject], MicCapture] = MicCapture,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("glassCard")
        self.setFixedWidth(640)
        self._device_provider = device_provider
        self._capture_factory = capture_factory
        self._capture: Optional[MicCapture] = None

        row = QHBoxLayout(self)
        row.setContentsMargins(20, 10, 14, 10)
        row.setSpacing(12)
        text = QVBoxLayout()
        text.setSpacing(2)
        self._title = QLabel()
        self._title.setStyleSheet("font-weight: 600;")
        self._steps = QLabel()
        self._steps.setWordWrap(True)
        theme.set_tone(self._steps, "hint")
        text.addWidget(self._title)
        text.addWidget(self._steps)
        row.addLayout(text, 1)

        self._meter = LevelMeter()
        self._meter.setVisible(False)
        row.addWidget(self._meter, alignment=Qt.AlignVCenter)
        self._check_btn = QPushButton()
        self._check_btn.setObjectName("ghostPill")
        self._check_btn.setCursor(Qt.PointingHandCursor)
        self._check_btn.clicked.connect(self.start_check)
        self._close_btn = QPushButton()
        self._close_btn.setObjectName("outlinePill")
        self._close_btn.setCursor(Qt.PointingHandCursor)
        self._close_btn.clicked.connect(self.dismissed.emit)
        row.addWidget(self._check_btn, alignment=Qt.AlignVCenter)
        row.addWidget(self._close_btn, alignment=Qt.AlignVCenter)
        self._result: Optional[tuple[str, dict, str]] = None   # (key, args, tone)
        self.retranslate()

    # ── mic check ──────────────────────────────────────────

    def start_check(self) -> None:
        if self._capture is not None:
            return
        self._capture = self._capture_factory(self)
        self._capture.level.connect(self._meter.set_level)
        self._capture.finished.connect(self._on_finished)
        self._capture.failed.connect(self._on_failed)
        self._meter.set_level(0.0)
        self._meter.setVisible(True)
        self._check_btn.setEnabled(False)
        self._set_result("onb_mic_listening", {}, "accent")
        self._capture.start(self._device_provider())

    def _on_finished(self, peak: float) -> None:
        if peak < SILENT_PEAK:
            self._set_result("onb_mic_silent", {}, "danger")
        else:
            self._set_result("onb_mic_ok", {"db": f"{to_dbfs(peak):.0f}"}, "ok")
        self._end_check()

    def _on_failed(self, message: str) -> None:
        self._set_result("onb_mic_error", {"error": message}, "danger")
        self._end_check()

    def _end_check(self) -> None:
        if self._capture is not None:
            self._capture.deleteLater()
            self._capture = None
        self._check_btn.setEnabled(True)
        self._check_btn.setText(t("onb_check_again"))

    def _set_result(self, key: str, args: dict, tone: str) -> None:
        self._result = (key, args, tone)
        self._steps.setText(t(key, **args))
        theme.set_tone(self._steps, tone)

    # ── texts ──────────────────────────────────────────────

    def retranslate(self) -> None:
        self._title.setText(t("onb_title"))
        self._close_btn.setText(t("onb_dismiss"))
        if self._result is None:
            self._steps.setText(t("onb_steps"))
            self._check_btn.setText(t("onb_check"))
        else:
            key, args, _tone = self._result
            self._steps.setText(t(key, **args))
            if self._capture is None:
                self._check_btn.setText(t("onb_check_again"))
        self._steps.setToolTip(t("onb_steps_tip"))
