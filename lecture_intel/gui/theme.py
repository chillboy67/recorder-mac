"""
Visual theme for Recorder — light + dark, follow-system.

Fusion base (fully stylable) + a hand-tuned palette and stylesheet, modeled on
native macOS: flat surfaces, hairline borders, the system blue accent, no
gradients and no drop shadows (shadow effects clip inside scroll areas and
look muddy — contrast comes from background layers instead).

All widget colors live HERE. Widgets opt in via objectName / dynamic properties
(e.g. setProperty("tone", "ok") + theme.repolish(w)) so that switching the
appearance restyles everything without per-widget code.

Public API:
    theme.apply(app, mode)          mode ∈ {"auto","light","dark"}
    theme.set_tone(label, tone)     tone ∈ {"hint","ok","danger","accent"}
    theme.repolish(widget)          re-evaluate QSS after a property change
    theme.wave_pixmap(h, color)     waveform glyph (drop zone decoration)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

# ── color schemes (zinc neutrals + indigo accent, Linear/Raycast-like) ──
LIGHT = {
    "canvas": "#F7F7F8", "canvas2": "#F1F1F3",
    "card": "#FFFFFF", "border": "#E8E8EC", "divider": "#F0F0F1",
    "ink": "#18181B", "ink2": "#51525C", "ink3": "#9C9CA5",
    "accent": "#6366F1", "accent_dk": "#4F46E5", "accent_soft": "#EEF0FE",
    "accent_border": "#DCE0FB",
    "field": "#FFFFFF", "field_border": "#E1E1E6",
    "danger": "#E5484D", "danger_dk": "#D33036", "danger_soft": "#FDEDED",
    "ok": "#16A34A", "ok_soft": "#EBF7EF",
    "tab_bg": "#ECECEF",
}
DARK = {
    "canvas": "#111113", "canvas2": "#0C0C0E",
    "card": "#19191C", "border": "#29292E", "divider": "#232327",
    "ink": "#EEEEF0", "ink2": "#A0A0A8", "ink3": "#64646C",
    "accent": "#818CF8", "accent_dk": "#6A76F5", "accent_soft": "#232441",
    "accent_border": "#363868",
    "field": "#202024", "field_border": "#333338",
    "danger": "#F2555A", "danger_dk": "#E5484D", "danger_soft": "#3A2224",
    "ok": "#3DD68C", "ok_soft": "#1B3125",
    "tab_bg": "#232327",
}

_state = {"mode": "auto", "app": None, "scheme": LIGHT, "is_dark": False}
_CHECK_PNG = ""
_DOT_PNG = ""
_CHEVRON_PNG = ""


def is_dark() -> bool:
    return _state["is_dark"]


def current_scheme() -> dict:
    return _state["scheme"]


def _system_is_dark(app: QApplication) -> bool:
    # colorScheme() is the right API but returns Unknown very early in startup;
    # fall back to the (native) palette's window lightness, which reflects the OS.
    try:
        cs = app.styleHints().colorScheme()
        if cs == Qt.ColorScheme.Dark:
            return True
        if cs == Qt.ColorScheme.Light:
            return False
    except Exception:
        pass
    return app.palette().color(QPalette.Window).lightness() < 128


def apply(app: QApplication, mode: str | None = None) -> None:
    if mode is not None:
        _state["mode"] = mode
    _state["app"] = app

    # Decide light/dark BEFORE switching to Fusion — setStyle resets the palette
    # to Fusion's (light) default, destroying the native dark-mode signal.
    effective_dark = (_system_is_dark(app) if _state["mode"] == "auto"
                      else _state["mode"] == "dark")

    app.setStyle("Fusion")
    c = DARK if effective_dark else LIGHT
    _state["scheme"] = c
    _state["is_dark"] = effective_dark

    _make_indicator_pngs(c)
    app.setPalette(_palette(c))
    app.setStyleSheet(_qss(c))

    # react to system appearance changes when following the system
    try:
        sh = app.styleHints()
        sh.colorSchemeChanged.disconnect(_on_system_scheme_changed)
    except Exception:
        pass
    if _state["mode"] == "auto":
        try:
            app.styleHints().colorSchemeChanged.connect(_on_system_scheme_changed)
        except Exception:
            pass


def _on_system_scheme_changed(*_) -> None:
    app = _state["app"]
    if app is not None and _state["mode"] == "auto":
        apply(app, "auto")


# ── widget helpers ──────────────────────────────────────────────────────

def repolish(widget) -> None:
    """Re-evaluate the stylesheet after a dynamic property change."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def set_tone(label, tone: str) -> None:
    """Color a status QLabel via the [tone=…] QSS rules (theme-aware)."""
    label.setProperty("tone", tone)
    repolish(label)


# ── palette + qss ───────────────────────────────────────────────────────

def _palette(c: dict) -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(c["canvas"]))
    p.setColor(QPalette.WindowText, QColor(c["ink"]))
    p.setColor(QPalette.Base, QColor(c["card"]))
    p.setColor(QPalette.AlternateBase, QColor(c["canvas"]))
    p.setColor(QPalette.Text, QColor(c["ink"]))
    p.setColor(QPalette.Button, QColor(c["card"]))
    p.setColor(QPalette.ButtonText, QColor(c["ink"]))
    p.setColor(QPalette.Highlight, QColor(c["accent"]))
    p.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    p.setColor(QPalette.ToolTipBase, QColor(c["ink"]))
    p.setColor(QPalette.ToolTipText, QColor(c["card"]))
    p.setColor(QPalette.PlaceholderText, QColor(c["ink3"]))
    for grp in (QPalette.Disabled,):
        p.setColor(grp, QPalette.Text, QColor(c["ink3"]))
        p.setColor(grp, QPalette.ButtonText, QColor(c["ink3"]))
        p.setColor(grp, QPalette.WindowText, QColor(c["ink3"]))
    return p


def _qss(c: dict) -> str:
    return f"""
    * {{
        font-family: -apple-system, "SF Pro Text", "PingFang SC", "Helvetica Neue";
        font-size: 13px; color: {c['ink']}; outline: 0;
    }}
    QMainWindow {{ background: {c['canvas']}; }}
    QStatusBar {{ background: transparent; color: {c['ink3']}; font-size: 12px; }}
    QToolTip {{ background: {c['ink']}; color: {c['card']}; border: none;
               padding: 6px 9px; border-radius: 6px; }}

    /* ── cards ─────────────────────────────────────────────── */
    QGroupBox {{
        background: {c['card']}; border: 1px solid {c['border']};
        border-radius: 14px; margin-top: 22px; padding: 16px;
        font-size: 12px; font-weight: 600; color: {c['ink2']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 4px; top: 2px; padding: 0 2px;
        color: {c['ink3']}; font-size: 11px; font-weight: 700; letter-spacing: 3px;
    }}

    /* ── status label tones (theme-aware, see theme.set_tone) ─ */
    QLabel[tone="hint"]   {{ color: {c['ink3']};   font-size: 12px; }}
    QLabel[tone="ok"]     {{ color: {c['ok']};     font-size: 12px; font-weight: 600; }}
    QLabel[tone="danger"] {{ color: {c['danger']}; font-size: 12px; font-weight: 600; }}
    QLabel[tone="accent"] {{ color: {c['accent']}; font-size: 12px; font-weight: 600; }}
    QLabel[hint="true"]   {{ color: {c['ink3']};   font-size: 11px; }}

    /* ── the recording clock ──────────────────────────────── */
    QLabel#clock {{
        font-size: 44px; font-weight: 100; letter-spacing: 1px;
        font-family: "SF Pro Display", -apple-system, "Helvetica Neue";
        color: {c['ink']};
    }}

    /* ── inputs ────────────────────────────────────────────── */
    QComboBox {{
        background: {c['field']}; border: 1px solid {c['field_border']};
        border-radius: 9px; padding: 7px 11px; min-height: 18px; color: {c['ink']};
    }}
    QComboBox:hover {{ border-color: {c['ink3']}; }}
    QComboBox:focus {{ border-color: {c['accent']}; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox::down-arrow {{ image: url({_CHEVRON_PNG}); width: 11px; height: 11px; }}
    QComboBox QAbstractItemView {{
        background: {c['card']}; border: 1px solid {c['border']}; border-radius: 10px;
        padding: 5px; selection-background-color: {c['accent']}; selection-color: #fff;
        outline: 0;
    }}

    QRadioButton, QCheckBox {{ spacing: 8px; color: {c['ink2']};
                               background: transparent; font-size: 13px; font-weight: 500; }}
    QRadioButton:checked, QCheckBox:checked {{ color: {c['ink']}; font-weight: 600; }}
    QRadioButton::indicator, QCheckBox::indicator {{ width: 17px; height: 17px; }}
    QRadioButton::indicator {{ border: 1px solid {c['field_border']};
                               border-radius: 8px; background: {c['field']}; }}
    QRadioButton::indicator:checked {{ border: 1px solid {c['accent']};
        border-radius: 8px; background: {c['accent']}; image: url({_DOT_PNG}); }}
    QRadioButton::indicator:hover, QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
    QCheckBox::indicator {{ border: 1px solid {c['field_border']}; border-radius: 5px;
                            background: {c['field']}; }}
    QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']};
                                    image: url({_CHECK_PNG}); }}

    /* ── buttons ───────────────────────────────────────────── */
    QPushButton {{
        background: {c['card']}; border: 1px solid {c['field_border']};
        border-radius: 9px; padding: 7px 14px; color: {c['ink']}; font-weight: 500;
    }}
    QPushButton:hover {{ background: {c['canvas']}; border-color: {c['ink3']}; }}
    QPushButton:pressed {{ background: {c['tab_bg']}; }}
    QPushButton:disabled {{ color: {c['ink3']}; border-color: {c['border']};
                            background: {c['card']}; }}

    QPushButton#primary {{
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {c['accent']}, stop:1 {c['accent_dk']});
        color: #fff; border: 1px solid {c['accent_dk']}; border-radius: 10px;
        padding: 12px 16px; font-size: 14px; font-weight: 600; letter-spacing: 4px;
    }}
    QPushButton#primary:hover {{ background: {c['accent_dk']}; color: #fff; }}
    QPushButton#primary:pressed {{ background: {c['accent_dk']}; }}
    QPushButton#primary:disabled {{ background: {c['tab_bg']}; color: {c['ink3']};
                                    border-color: {c['border']}; }}

    QPushButton#danger {{
        background: {c['danger']}; color: #fff; border: none;
        border-radius: 10px; padding: 10px 16px; font-weight: 600; letter-spacing: 2px;
    }}
    QPushButton#danger:hover {{ background: {c['danger_dk']}; color: #fff; }}

    /* record button — accent when idle, red while recording */
    QPushButton#record {{
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {c['accent']}, stop:1 {c['accent_dk']});
        color: #fff; border: 1px solid {c['accent_dk']}; border-radius: 10px;
        font-size: 14px; font-weight: 600; letter-spacing: 2px;
    }}
    QPushButton#record:hover {{ background: {c['accent_dk']}; color: #fff; }}
    QPushButton#record[recording="true"] {{
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {c['danger']}, stop:1 {c['danger_dk']});
        border-color: {c['danger_dk']};
    }}
    QPushButton#record[recording="true"]:hover {{ background: {c['danger_dk']}; }}

    /* ghost button — quiet accent action (选择文件…) */
    QPushButton#ghost {{
        background: transparent; color: {c['accent']};
        border: 1px solid {c['accent_border']}; border-radius: 9px;
        font-size: 13px; font-weight: 600; padding: 9px 14px;
    }}
    QPushButton#ghost:hover {{ background: {c['accent_soft']};
                               border-color: {c['accent']}; }}

    /* ── drop zone ─────────────────────────────────────────── */
    QFrame#dropArea {{
        border: 1.5px dashed {c['field_border']};
        border-radius: 12px;
        background: {c['canvas']};
    }}
    QFrame#dropArea QLabel {{ background: transparent; border: none; }}
    QFrame#dropArea QLabel#dropTitle {{ color: {c['ink2']}; font-size: 13px; font-weight: 600; }}
    QFrame#dropArea QLabel#dropSub   {{ color: {c['ink3']}; font-size: 11px; }}
    QFrame#dropArea[state="hover"] {{
        border: 1.5px dashed {c['accent']}; background: {c['accent_soft']};
    }}
    QFrame#dropArea[state="hover"] QLabel#dropTitle {{ color: {c['accent']}; }}
    QFrame#dropArea[state="selected"] {{
        border: 1.5px solid {c['ok']}; background: {c['ok_soft']};
    }}
    QFrame#dropArea[state="selected"] QLabel#dropTitle {{ color: {c['ok']}; }}

    /* ── tabs (editorial underline style) ──────────────────── */
    QTabWidget::pane {{ border: none; top: 6px; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{
        background: transparent; color: {c['ink3']};
        border: none; border-bottom: 2px solid transparent; border-radius: 0;
        padding: 7px 2px 9px 2px; margin-right: 22px;
        font-weight: 600; min-width: 40px;
    }}
    QTabBar::tab:selected {{ color: {c['ink']};
                             border-bottom: 2px solid {c['accent']}; }}
    QTabBar::tab:hover:!selected {{ color: {c['ink2']}; }}

    /* ── progress ──────────────────────────────────────────── */
    QProgressBar {{ background: {c['tab_bg']}; border: none; border-radius: 3px; }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}

    /* ── text areas ────────────────────────────────────────── */
    QPlainTextEdit, QTextEdit {{
        background: {c['card']}; border: 1px solid {c['border']}; border-radius: 12px;
        padding: 13px; selection-background-color: {c['accent_soft']}; selection-color: {c['ink']};
    }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 3px; }}
    QScrollBar::handle:vertical {{ background: {c['field_border']}; border-radius: 4px;
                                   min-height: 32px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['ink3']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ height: 0px; }}

    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {c['border']}; }}
    QMenuBar {{ background: transparent; }}
    QMenuBar::item:selected {{ background: {c['tab_bg']}; border-radius: 5px; }}
    QMenu {{ background: {c['card']}; border: 1px solid {c['border']};
             border-radius: 10px; padding: 5px; }}
    QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {c['accent']}; color: #fff; }}
    QMessageBox, QFileDialog {{ background: {c['canvas']}; }}
    """


# ── drawn glyphs (crisper than emoji, theme-aware) ──────────────────────

# waveform bar heights as fractions of the glyph height
_WAVE_BARS = (0.42, 0.72, 1.0, 0.58, 0.82, 0.36)


def wave_pixmap(height: int = 30, color: str | None = None) -> QPixmap:
    """A rounded-bar waveform glyph in the accent color (for the drop zone)."""
    c = _state["scheme"]
    col = QColor(color or c["accent"])
    dpr = 2.0
    w = int(height * 1.5)
    pm = QPixmap(int(w * dpr), int(height * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(col))
    n = len(_WAVE_BARS)
    bar_w = w / 11
    gap = (w - n * bar_w) / (n - 1)
    x = 0.0
    for frac in _WAVE_BARS:
        h = height * frac
        y = (height - h) / 2
        p.drawRoundedRect(QRectF(x, y, bar_w, h), bar_w / 2, bar_w / 2)
        x += bar_w + gap
    p.end()
    return pm


# ── indicator images ────────────────────────────────────────────────────

def _make_indicator_pngs(c: dict) -> None:
    global _CHECK_PNG, _DOT_PNG, _CHEVRON_PNG
    tmp = Path(tempfile.gettempdir())

    pm = QPixmap(28, 28); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor("#FFFFFF")); pen.setWidth(3)
    pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin); p.setPen(pen)
    p.drawPolyline([QPointF(7, 14), QPointF(12, 19), QPointF(21, 9)]); p.end()
    cpath = tmp / "recorder_check.png"; pm.save(str(cpath), "PNG")
    _CHECK_PNG = str(cpath).replace("\\", "/")

    pm2 = QPixmap(28, 28); pm2.fill(Qt.transparent)
    p2 = QPainter(pm2); p2.setRenderHint(QPainter.Antialiasing)
    p2.setPen(Qt.NoPen); p2.setBrush(QBrush(QColor("#FFFFFF")))
    p2.drawEllipse(QPointF(14, 14), 5, 5); p2.end()
    dpath = tmp / "recorder_dot.png"; pm2.save(str(dpath), "PNG")
    _DOT_PNG = str(dpath).replace("\\", "/")

    # chevron for combo boxes — ink3 so it reads in both schemes
    pm3 = QPixmap(24, 24); pm3.fill(Qt.transparent)
    p3 = QPainter(pm3); p3.setRenderHint(QPainter.Antialiasing)
    pen3 = QPen(QColor(c["ink3"])); pen3.setWidth(3)
    pen3.setCapStyle(Qt.RoundCap); pen3.setJoinStyle(Qt.RoundJoin); p3.setPen(pen3)
    p3.drawPolyline([QPointF(6, 10), QPointF(12, 16), QPointF(18, 10)]); p3.end()
    # name varies per scheme so Qt's pixmap cache doesn't serve a stale arrow
    vpath = tmp / f"recorder_chevron_{'d' if c is DARK else 'l'}.png"
    pm3.save(str(vpath), "PNG")
    _CHEVRON_PNG = str(vpath).replace("\\", "/")
