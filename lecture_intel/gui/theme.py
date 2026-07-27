"""
Visual theme for Recorder — light + dark, follow-system, with depth.

Fusion base (fully stylable) + a hand-tuned palette and stylesheet. Design goals:
a calm layered canvas, white/elevated cards with soft shadows, a refined indigo
accent, clear type hierarchy, generous spacing — premium, not flat.

Public API:
    theme.apply(app, mode)          mode ∈ {"auto","light","dark"}
    theme.add_card_shadow(widget)   soft drop shadow for a card/panel
    theme.add_glow(widget, color)   colored shadow for the primary button
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QGraphicsDropShadowEffect

# ── color schemes ───────────────────────────────────────────────────────
LIGHT = {
    "canvas": "#EBEEF3", "canvas2": "#E3E7EF",
    "card": "#FFFFFF", "border": "#E3E7EE", "divider": "#ECEEF3",
    "ink": "#15171C", "ink2": "#565D6D", "ink3": "#959CAA",
    "accent": "#4F6BFF", "accent_dk": "#3F59E8", "accent_soft": "#ECEFFF",
    "accent_border": "#D5DCFF",
    "field": "#FFFFFF", "field_border": "#DBDEE8",
    "danger": "#FF453A", "danger_dk": "#E23B31",
    "ok": "#2BB673",
    "tab_bg": "#E1E5EE",
    "shadow": (20, 28, 56, 38),     # rgba — soft cool shadow
}
DARK = {
    "canvas": "#131519", "canvas2": "#0F1115",
    "card": "#1D2027", "border": "#2B313C", "divider": "#262B34",
    "ink": "#EEF0F5", "ink2": "#A6ADBA", "ink3": "#6C7380",
    "accent": "#6E86FF", "accent_dk": "#5B79FF", "accent_soft": "#222843",
    "accent_border": "#33406B",
    "field": "#242833", "field_border": "#343B47",
    "danger": "#FF6961", "danger_dk": "#E8554D",
    "ok": "#32D583",
    "tab_bg": "#23272F",
    "shadow": (0, 0, 0, 110),
}

_state = {"mode": "auto", "app": None, "scheme": LIGHT, "is_dark": False}
_CHECK_PNG = ""
_DOT_PNG = ""


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
    QMainWindow, QStatusBar {{ background: {c['canvas']}; }}
    QStatusBar {{ color: {c['ink3']}; }}
    QToolTip {{ background: {c['ink']}; color: {c['card']}; border: none;
               padding: 6px 9px; border-radius: 7px; }}

    QGroupBox {{
        background: {c['card']}; border: 1px solid {c['border']};
        border-radius: 16px; margin-top: 20px; padding: 16px 16px 16px 16px;
        font-size: 12px; font-weight: 600; color: {c['ink2']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 16px; top: 3px; padding: 0 3px;
        color: {c['ink3']}; font-size: 11px; font-weight: 700;
    }}

    QComboBox {{
        background: {c['field']}; border: 1px solid {c['field_border']};
        border-radius: 10px; padding: 8px 12px; min-height: 18px; color: {c['ink']};
    }}
    QComboBox:hover {{ border-color: {c['accent']}; }}
    QComboBox:focus {{ border-color: {c['accent']}; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox QAbstractItemView {{
        background: {c['card']}; border: 1px solid {c['border']}; border-radius: 12px;
        padding: 6px; selection-background-color: {c['accent']}; selection-color: #fff;
        outline: 0;
    }}

    QRadioButton, QCheckBox {{ spacing: 9px; color: {c['ink']};
                               background: transparent; font-size: 13px; font-weight: 500; }}
    QRadioButton::indicator, QCheckBox::indicator {{ width: 18px; height: 18px; }}
    QRadioButton::indicator {{ border: 1.5px solid {c['field_border']};
                               border-radius: 9px; background: {c['field']}; }}
    QRadioButton::indicator:checked {{ border: 1.5px solid {c['accent']};
        border-radius: 9px; background: {c['accent']}; image: url({_DOT_PNG}); }}
    QRadioButton::indicator:hover, QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
    QCheckBox::indicator {{ border: 1.5px solid {c['field_border']}; border-radius: 6px;
                            background: {c['field']}; }}
    QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']};
                                    image: url({_CHECK_PNG}); }}

    QPushButton {{
        background: {c['card']}; border: 1px solid {c['field_border']};
        border-radius: 10px; padding: 8px 16px; color: {c['ink']}; font-weight: 500;
    }}
    QPushButton:hover {{ border-color: {c['accent']}; color: {c['accent']}; }}
    QPushButton:pressed {{ background: {c['accent_soft']}; }}
    QPushButton:disabled {{ color: {c['ink3']}; border-color: {c['border']}; }}

    QPushButton#primary {{
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {c['accent']}, stop:1 {c['accent_dk']});
        color: #fff; border: none; border-radius: 13px;
        padding: 13px 18px; font-size: 14px; font-weight: 700;
    }}
    QPushButton#primary:hover {{ background: {c['accent_dk']}; }}
    QPushButton#primary:disabled {{ background: {c['field_border']}; color: {c['ink3']}; }}

    QPushButton#danger {{
        background: {c['danger']}; color: #fff; border: none;
        border-radius: 11px; padding: 10px 16px; font-weight: 700;
    }}
    QPushButton#danger:hover {{ background: {c['danger_dk']}; color: #fff; }}

    QTabWidget::pane {{ border: none; top: 2px; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{
        background: {c['tab_bg']}; color: {c['ink2']}; border: none;
        padding: 8px 18px; margin-right: 5px; border-radius: 9px;
        font-weight: 600; min-width: 64px;
    }}
    QTabBar::tab:selected {{ background: {c['card']}; color: {c['accent']};
                             border: 1px solid {c['border']}; }}
    QTabBar::tab:hover:!selected {{ color: {c['ink']}; }}

    QProgressBar {{ background: {c['tab_bg']}; border: none; border-radius: 4px; height: 7px; }}
    QProgressBar::chunk {{ border-radius: 4px;
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {c['accent']}, stop:1 {c['accent_dk']}); }}

    QPlainTextEdit, QTextEdit {{
        background: {c['card']}; border: 1px solid {c['border']}; border-radius: 14px;
        padding: 14px; selection-background-color: {c['accent_soft']}; selection-color: {c['ink']};
    }}

    QScrollBar:vertical {{ background: transparent; width: 11px; margin: 3px; }}
    QScrollBar::handle:vertical {{ background: {c['field_border']}; border-radius: 5px;
                                   min-height: 32px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['ink3']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ height: 0px; }}

    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {c['accent_soft']}; }}
    QMenuBar {{ background: {c['canvas']}; }}
    QMenuBar::item:selected {{ background: {c['accent_soft']}; border-radius: 6px; }}
    QMenu {{ background: {c['card']}; border: 1px solid {c['border']};
             border-radius: 12px; padding: 6px; }}
    QMenu::item {{ padding: 7px 24px; border-radius: 7px; }}
    QMenu::item:selected {{ background: {c['accent']}; color: #fff; }}
    QMessageBox, QFileDialog {{ background: {c['canvas']}; }}
    """


# ── shadows (QSS can't do these) ────────────────────────────────────────

def add_card_shadow(widget) -> None:
    c = _state["scheme"]
    r, g, b, a = c["shadow"]
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(34)
    eff.setColor(QColor(r, g, b, a))
    eff.setOffset(0, 7)
    widget.setGraphicsEffect(eff)


def add_glow(widget, color: str | None = None) -> None:
    c = _state["scheme"]
    col = QColor(color or c["accent"])
    col.setAlpha(120)
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(26)
    eff.setColor(col)
    eff.setOffset(0, 6)
    widget.setGraphicsEffect(eff)


# ── indicator images ────────────────────────────────────────────────────

def _make_indicator_pngs(c: dict) -> None:
    global _CHECK_PNG, _DOT_PNG
    from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush
    from PySide6.QtCore import QPointF
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
