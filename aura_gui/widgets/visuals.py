"""
Custom-painted AURA visuals: animated wave bars, the recording glow ring,
and the processing progress ring. All colors come from theme.current_scheme()
at paint time, so appearance switches restyle them via update().
"""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from gui import theme


def _c(key: str) -> QColor:
    return QColor(theme.current_scheme()[key])


def _alpha(key: str, a: int) -> QColor:
    col = _c(key)
    col.setAlpha(a)
    return col


# ── animated wave bars (recording) ──────────────────────────────────────

class WaveBars(QWidget):
    """A row of rounded bars pulsing with independent phases."""

    def __init__(self, bars: int = 17, height: int = 40,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n = bars
        self._phases = [i * 1.7 for i in range(bars)]
        self._speeds = [0.10 + 0.05 * ((i * 7) % 5) for i in range(bars)]
        self._t = 0.0
        self.setFixedSize(bars * 8, height)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._t += 1.0
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(_c("accent"))
        h = self.height()
        for i in range(self._n):
            frac = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(
                self._phases[i] + self._t * self._speeds[i]))
            bh = h * frac
            x = i * 8 + 2.5
            p.drawRoundedRect(QRectF(x, (h - bh) / 2, 3, bh), 1.5, 1.5)
        p.end()


# ── static wave glyph (drop card) ───────────────────────────────────────

class WaveGlyph(QWidget):
    HEIGHTS = (10, 20, 30, 16, 24, 10)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(len(self.HEIGHTS) * 8, 36)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(_c("accent"))
        h = self.height()
        for i, bh in enumerate(self.HEIGHTS):
            p.drawRoundedRect(QRectF(i * 8 + 2.5, (h - bh) / 2, 3, bh), 1.5, 1.5)
        p.end()


# ── recording glow ring (no progress implication) ───────────────────────

class GlowRing(QWidget):
    """Uniform soft ice-blue ring with the elapsed clock centered inside."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(260, 260)
        self._clock = "00:00:00"
        self._sub = "−14 dB · 16 kHz"

    def set_time(self, text: str) -> None:
        self._clock = text
        self.update()

    def set_sub(self, text: str) -> None:
        self._sub = text
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2

        # outer glow
        glow = QRadialGradient(cx, cy, cx)
        glow.setColorAt(0.80, _alpha("accent", 0))
        glow.setColorAt(0.90, _alpha("accent", 46))
        glow.setColorAt(1.00, _alpha("accent", 0))
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QRectF(0, 0, self.width(), self.height()))

        # ring band
        p.setBrush(_alpha("accent", 26))
        p.drawEllipse(QRectF(10, 10, self.width() - 20, self.height() - 20))
        p.setBrush(_c("bg2"))
        p.drawEllipse(QRectF(16, 16, self.width() - 32, self.height() - 32))

        # clock
        p.setPen(QPen(_c("ink")))
        p.setFont(theme.display_font(34, QFont.Weight.Light))
        p.drawText(QRectF(0, 0, self.width(), self.height() - 26),
                   Qt.AlignCenter, self._clock)
        p.setPen(QPen(_c("ink3")))
        p.setFont(theme.display_font(9, QFont.Weight.Normal))
        p.drawText(QRectF(0, 60, self.width(), self.height() - 60),
                   Qt.AlignCenter, self._sub)
        p.end()


# ── processing progress ring ────────────────────────────────────────────

class ProgressRing(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(220, 220)
        self._percent = 0
        self._label = ""

    def set_progress(self, percent: int, label: str) -> None:
        self._percent = max(0, min(100, percent))
        self._label = label
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(8, 8, self.width() - 16, self.height() - 16)

        track = QPen(_c("line2"), 13, Qt.SolidLine, Qt.RoundCap)
        p.setPen(track)
        p.drawArc(rect, 0, 360 * 16)

        arc = QPen(_c("accent"), 13, Qt.SolidLine, Qt.RoundCap)
        p.setPen(arc)
        span = int(-360 * 16 * self._percent / 100)
        p.drawArc(rect, 90 * 16, span)

        p.setPen(QPen(_c("ink")))
        p.setFont(theme.display_font(36, QFont.Weight.Light))
        p.drawText(QRectF(0, 0, self.width(), self.height() - 30),
                   Qt.AlignCenter, f"{self._percent}%")
        p.setPen(QPen(_c("accent")))
        f = QFont(); f.setPointSize(11)
        p.setFont(f)
        p.drawText(QRectF(0, 54, self.width(), self.height() - 54),
                   Qt.AlignCenter, self._label)
        p.end()


# ── pulsing record dot ──────────────────────────────────────────────────

class PulseDot(QWidget):
    def __init__(self, color_key: str = "danger",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._key = color_key
        self._on = True
        self.setFixedSize(10, 10)
        self._timer = QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self._flip)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._on = True
        self.update()

    def _flip(self) -> None:
        self._on = not self._on
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        col = _c(self._key)
        col.setAlpha(255 if self._on else 90)
        p.setBrush(col)
        p.drawEllipse(QRectF(0.5, 0.5, 9, 9))
        p.end()


# ── window background with radial glow ──────────────────────────────────

class GlowBackground(QWidget):
    """Central-widget base that paints the AURA radial glow."""

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), _c("bg"))
        g = QRadialGradient(self.width() * 0.60, -self.height() * 0.10,
                            max(self.width() * 0.8, 600))
        is_dark = theme.is_dark()
        g.setColorAt(0.0, _alpha("accent", 34 if is_dark else 48))
        g.setColorAt(0.65, _alpha("accent", 0))
        p.fillRect(self.rect(), g)
        p.end()
