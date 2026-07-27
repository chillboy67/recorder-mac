"""
AURA visual theme for Recorder — deep blue-black + ice-blue accent.

Fusion base + hand-tuned palette/QSS. Same public API as the old theme so the
rest of the app keeps working:

    theme.apply(app, mode)          mode ∈ {"auto","light","dark"}
    theme.set_tone(label, tone)     tone ∈ {"hint","ok","danger","accent"}
    theme.repolish(widget)
    theme.current_scheme() / is_dark()
    theme.display_font(pt, weight)  Space Grotesk-style numerals (fallback safe)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

# ── AURA color schemes ──────────────────────────────────────────────────
DARK = {
    "bg": "#0A0D14", "bg2": "#0C1018",
    "card": "#12161F", "card2": "#10141C",
    "line": "#232A37", "line2": "#1A202C",
    "ink": "#E9EFF6", "ink2": "#8C99AD", "ink3": "#566076",
    "accent": "#7FC6EE", "accent_ink": "#0A0D14",
    "accent_soft": "#17293B", "accent_border": "#2F4C63",
    "danger": "#F27E72", "danger_dk": "#E2685C", "danger_soft": "#321F20",
    "ok": "#6ED8A8", "ok_soft": "#152B22",
    "warn": "#F0BE6E", "violet": "#B0A6F2", "violet_soft": "#26243C",
}
LIGHT = {
    "bg": "#EFF3F8", "bg2": "#E7EDF5",
    "card": "#FFFFFF", "card2": "#F7FAFD",
    "line": "#D6DFE9", "line2": "#E4EAF1",
    "ink": "#1A2433", "ink2": "#5D6B7E", "ink3": "#8A97AA",
    "accent": "#2E7FB8", "accent_ink": "#FFFFFF",
    "accent_soft": "#E2EEF7", "accent_border": "#A9CCE3",
    "danger": "#D95C4E", "danger_dk": "#C74B3E", "danger_soft": "#FBE9E6",
    "ok": "#2F9D6E", "ok_soft": "#E4F4EC",
    "warn": "#B07E24", "violet": "#6C5FCC", "violet_soft": "#ECEAF9",
}

# Latin/number display stack; CJK body text falls through to PingFang.
DISPLAY_FAMILIES = ["Space Grotesk", "SF Pro Display", "Helvetica Neue"]
QSS_DISPLAY = '"Space Grotesk", "SF Pro Display", "Helvetica Neue"'

_state = {"mode": "auto", "app": None, "scheme": DARK, "is_dark": True}
_CHECK_PNG = ""
_DOT_PNG = ""
_CHEVRON_PNG = ""


def is_dark() -> bool:
    return _state["is_dark"]


def current_scheme() -> dict:
    return _state["scheme"]


def display_font(pt: int, weight: QFont.Weight = QFont.Weight.Light) -> QFont:
    f = QFont()
    f.setFamilies(DISPLAY_FAMILIES)
    f.setPointSize(pt)
    f.setWeight(weight)
    return f


def _system_is_dark(app: QApplication) -> bool:
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

    effective_dark = (_system_is_dark(app) if _state["mode"] == "auto"
                      else _state["mode"] == "dark")

    app.setStyle("Fusion")
    c = DARK if effective_dark else LIGHT
    _state["scheme"] = c
    _state["is_dark"] = effective_dark

    _make_indicator_pngs(c)
    app.setPalette(_palette(c))
    app.setStyleSheet(_qss(c))

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


def repolish(widget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def set_tone(label, tone: str) -> None:
    label.setProperty("tone", tone)
    repolish(label)


def _palette(c: dict) -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(c["bg"]))
    p.setColor(QPalette.WindowText, QColor(c["ink"]))
    p.setColor(QPalette.Base, QColor(c["card"]))
    p.setColor(QPalette.AlternateBase, QColor(c["bg"]))
    p.setColor(QPalette.Text, QColor(c["ink"]))
    p.setColor(QPalette.Button, QColor(c["card"]))
    p.setColor(QPalette.ButtonText, QColor(c["ink"]))
    p.setColor(QPalette.Highlight, QColor(c["accent"]))
    p.setColor(QPalette.HighlightedText, QColor(c["accent_ink"]))
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
    QMainWindow {{ background: {c['bg']}; }}
    QStatusBar {{ background: transparent; color: {c['ink3']}; font-size: 12px; }}
    QToolTip {{ background: {c['ink']}; color: {c['card']}; border: none;
               padding: 6px 9px; border-radius: 6px; }}

    /* ── top strip ─────────────────────────────────────────── */
    QLabel#wordmark {{ font-family: {QSS_DISPLAY}; font-size: 12px;
                       color: {c['ink3']}; font-weight: 600; }}
    QPushButton#quiet {{ background: transparent; border: none; color: {c['ink3']};
                         font-size: 12px; padding: 4px 8px; }}
    QPushButton#quiet:hover {{ color: {c['ink2']}; }}

    /* ── step rail ─────────────────────────────────────────── */
    QFrame#stepRail {{ background: transparent; border: none;
                       border-right: 1px solid {c['line2']}; }}
    QLabel[railNum="idle"]    {{ font-family: {QSS_DISPLAY}; font-size: 12px; color: {c['ink3']}; }}
    QLabel[railNum="active"]  {{ font-family: {QSS_DISPLAY}; font-size: 12px; color: {c['accent']}; font-weight: 600; }}
    QLabel[railNum="done"]    {{ font-family: {QSS_DISPLAY}; font-size: 12px; color: {c['ok']}; }}
    QLabel[railLabel="idle"]   {{ font-size: 12px; color: {c['ink3']}; }}
    QLabel[railLabel="active"] {{ font-size: 12px; color: {c['ink']}; font-weight: 600; }}
    QLabel[railLabel="done"]   {{ font-size: 12px; color: {c['ok']}; }}
    QFrame[railBar="on"]  {{ background: {c['accent']}; border-radius: 1px; }}
    QFrame[railBar="off"] {{ background: transparent; }}

    /* ── glass cards ───────────────────────────────────────── */
    QFrame#glassCard {{ background: {c['card']}; border: 1px solid {c['line']};
                        border-radius: 20px; }}
    QFrame#glassCardAccent {{ background: {c['card']}; border: 1px solid {c['accent_border']};
                              border-radius: 20px; }}
    QFrame#glassCard QLabel, QFrame#glassCardAccent QLabel {{ background: transparent; border: none; }}
    QFrame#hairline {{ background: {c['line2']}; border: none; }}

    /* drop card hover (drag enter) */
    QFrame#glassCard[drop="hover"] {{ border: 1px solid {c['accent']};
                                      background: {c['accent_soft']}; }}

    /* ── mode cards ────────────────────────────────────────── */
    QFrame#modeCard {{ background: {c['card2']}; border: 1px solid {c['line']};
                       border-radius: 14px; }}
    QFrame#modeCard[selected="true"] {{ background: {c['accent_soft']};
                                        border: 1px solid {c['accent_border']}; }}
    QFrame#modeCard QLabel {{ background: transparent; border: none; }}
    QLabel[cardTitle="on"]  {{ font-size: 13px; font-weight: 700; color: {c['accent']}; }}
    QLabel[cardTitle="off"] {{ font-size: 13px; font-weight: 600; color: {c['ink2']}; }}
    QLabel[cardDesc="on"]   {{ font-size: 11px; color: {c['ink2']}; }}
    QLabel[cardDesc="off"]  {{ font-size: 11px; color: {c['ink3']}; }}

    /* ── pills / buttons ───────────────────────────────────── */
    QPushButton {{ background: {c['card']}; border: 1px solid {c['line']};
                   border-radius: 9px; padding: 7px 14px; color: {c['ink']}; font-weight: 500; }}
    QPushButton:hover {{ border-color: {c['ink3']}; }}
    QPushButton:disabled {{ color: {c['ink3']}; border-color: {c['line2']}; }}

    QPushButton#primaryPill {{ background: {c['accent']}; color: {c['accent_ink']};
        border: none; border-radius: 19px; padding: 10px 26px;
        font-size: 13px; font-weight: 700; }}
    QPushButton#primaryPill:hover {{ background: {c['accent_border']}; color: {c['ink']}; }}
    QPushButton#primaryPill:disabled {{ background: {c['line']}; color: {c['ink3']}; }}

    QPushButton#ghostPill {{ background: transparent; color: {c['accent']};
        border: 1px solid {c['accent_border']}; border-radius: 19px;
        padding: 9px 24px; font-size: 13px; font-weight: 600; }}
    QPushButton#ghostPill:hover {{ background: {c['accent_soft']}; }}

    QPushButton#dangerPill {{ background: {c['danger']}; color: {c['bg']};
        border: none; border-radius: 21px; padding: 11px 34px;
        font-size: 14px; font-weight: 700; }}
    QPushButton#dangerPill:hover {{ background: {c['danger_dk']}; }}

    QPushButton#cancelPill {{ background: transparent; color: {c['danger']};
        border: 1px solid {c['danger_soft']}; border-radius: 16px;
        padding: 7px 20px; font-size: 13px; font-weight: 600; }}
    QPushButton#cancelPill:hover {{ background: {c['danger_soft']}; }}

    QPushButton#outlinePill {{ background: transparent; color: {c['ink2']};
        border: 1px solid {c['line']}; border-radius: 16px;
        padding: 7px 18px; font-size: 12px; font-weight: 600; }}
    QPushButton#outlinePill:hover {{ color: {c['ink']}; }}

    /* export chips + source segments */
    QPushButton#chip {{ background: transparent; border: 1px solid {c['line']};
        border-radius: 12px; padding: 3px 12px; color: {c['ink3']};
        font-family: {QSS_DISPLAY}; font-size: 11px; }}
    QPushButton#chip:checked {{ background: {c['accent_soft']};
        border: 1px solid {c['accent_border']}; color: {c['accent']}; }}
    QPushButton#seg {{ background: transparent; border: none; border-radius: 13px;
        padding: 6px 16px; color: {c['ink3']}; font-size: 12px; }}
    QPushButton#seg:checked {{ background: {c['accent_soft']}; color: {c['accent']};
        font-weight: 600; }}
    QFrame#segWrap {{ background: transparent; border: 1px solid {c['line']};
        border-radius: 17px; }}

    /* ── status label tones ────────────────────────────────── */
    QLabel[tone="hint"]   {{ color: {c['ink3']};   font-size: 12px; }}
    QLabel[tone="ok"]     {{ color: {c['ok']};     font-size: 12px; font-weight: 600; }}
    QLabel[tone="danger"] {{ color: {c['danger']}; font-size: 12px; font-weight: 600; }}
    QLabel[tone="accent"] {{ color: {c['accent']}; font-size: 12px; font-weight: 600; }}
    QLabel[hint="true"]   {{ color: {c['ink3']};   font-size: 11px; }}
    QLabel[mono="true"]   {{ font-family: {QSS_DISPLAY}; font-size: 11px; color: {c['ink3']}; }}

    /* ── inputs ────────────────────────────────────────────── */
    QComboBox {{ background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 9px; padding: 6px 11px; min-height: 18px; color: {c['ink']}; }}
    QComboBox:hover {{ border-color: {c['ink3']}; }}
    QComboBox:focus {{ border-color: {c['accent']}; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox::down-arrow {{ image: url({_CHEVRON_PNG}); width: 11px; height: 11px; }}
    QComboBox QAbstractItemView {{ background: {c['card']}; border: 1px solid {c['line']};
        border-radius: 10px; padding: 5px; selection-background-color: {c['accent']};
        selection-color: {c['accent_ink']}; outline: 0; }}

    QCheckBox {{ spacing: 8px; color: {c['ink2']}; background: transparent;
                 font-size: 12px; }}
    QCheckBox:checked {{ color: {c['ink']}; }}
    QCheckBox::indicator {{ width: 17px; height: 17px; border: 1px solid {c['line']};
                            border-radius: 5px; background: {c['card2']}; }}
    QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
    QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']};
                                    image: url({_CHECK_PNG}); }}

    /* ── tabs (underline) ──────────────────────────────────── */
    QTabWidget::pane {{ border: none; top: 6px; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{ background: transparent; color: {c['ink3']}; border: none;
        border-bottom: 2px solid transparent; border-radius: 0;
        padding: 7px 2px 9px 2px; margin-right: 22px; font-weight: 600; min-width: 40px; }}
    QTabBar::tab:selected {{ color: {c['ink']}; border-bottom: 2px solid {c['accent']}; }}
    QTabBar::tab:hover:!selected {{ color: {c['ink2']}; }}

    /* ── text areas / scrollbars / menus ───────────────────── */
    QPlainTextEdit, QTextEdit {{ background: transparent; border: none;
        padding: 6px; selection-background-color: {c['accent_soft']};
        selection-color: {c['ink']}; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 3px; }}
    QScrollBar::handle:vertical {{ background: {c['line']}; border-radius: 4px;
                                   min-height: 32px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['ink3']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ height: 0px; }}
    QMenuBar {{ background: transparent; }}
    QMenuBar::item:selected {{ background: {c['accent_soft']}; border-radius: 5px; }}
    QMenu {{ background: {c['card']}; border: 1px solid {c['line']};
             border-radius: 10px; padding: 5px; }}
    QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {c['accent']}; color: {c['accent_ink']}; }}
    QMessageBox, QFileDialog {{ background: {c['bg']}; }}
    """


# ── indicator pngs (check / dot / chevron) ──────────────────────────────

def _make_indicator_pngs(c: dict) -> None:
    global _CHECK_PNG, _DOT_PNG, _CHEVRON_PNG
    tmp = Path(tempfile.gettempdir())

    pm = QPixmap(28, 28); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(c["accent_ink"])); pen.setWidth(3)
    pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin); p.setPen(pen)
    p.drawPolyline([QPointF(7, 14), QPointF(12, 19), QPointF(21, 9)]); p.end()
    cpath = tmp / f"aura_check_{'d' if c is DARK else 'l'}.png"
    pm.save(str(cpath), "PNG")
    _CHECK_PNG = str(cpath).replace("\\", "/")

    pm2 = QPixmap(28, 28); pm2.fill(Qt.transparent)
    p2 = QPainter(pm2); p2.setRenderHint(QPainter.Antialiasing)
    p2.setPen(Qt.NoPen); p2.setBrush(QColor(c["accent_ink"]))
    p2.drawEllipse(QPointF(14, 14), 5, 5); p2.end()
    dpath = tmp / f"aura_dot_{'d' if c is DARK else 'l'}.png"
    pm2.save(str(dpath), "PNG")
    _DOT_PNG = str(dpath).replace("\\", "/")

    pm3 = QPixmap(24, 24); pm3.fill(Qt.transparent)
    p3 = QPainter(pm3); p3.setRenderHint(QPainter.Antialiasing)
    pen3 = QPen(QColor(c["ink3"])); pen3.setWidth(3)
    pen3.setCapStyle(Qt.RoundCap); pen3.setJoinStyle(Qt.RoundJoin); p3.setPen(pen3)
    p3.drawPolyline([QPointF(6, 10), QPointF(12, 16), QPointF(18, 10)]); p3.end()
    vpath = tmp / f"aura_chevron_{'d' if c is DARK else 'l'}.png"
    pm3.save(str(vpath), "PNG")
    _CHEVRON_PNG = str(vpath).replace("\\", "/")
