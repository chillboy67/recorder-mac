"""Shared widget subclasses."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, Qt
from PySide6.QtGui import QColor, QPaintEvent, QPainter, QPalette, QPen
from PySide6.QtWidgets import QComboBox, QFrame, QListView, QPushButton, QStyledItemDelegate

from gui import theme


class NoScrollComboBox(QComboBox):
    """A combo box that ignores mouse-wheel scrolling.

    Qt changes the selected item when the wheel scrolls over a combo box, even
    without clicking. We swallow the wheel event so scrolling moves the page;
    clicking still opens the dropdown normally.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # QComboBox's private delegate paints the current row with a square
        # frame and ignores QSS border-radius; the standard styled delegate is
        # what makes the rounded ::item rules in theme.py take effect.
        view = QListView(self)
        view.setItemDelegate(QStyledItemDelegate(view))
        self.setView(view)
        # The popup is a top-level window and macOS hands it a square, opaque
        # surface. The rounded panel is painted by the view (theme.py); for its
        # rounded corners to be the popup's edge, everything the window itself
        # paints must be gone: translucency + no system background + no auto
        # fill + a fully transparent Window palette role, all requested here,
        # before the native window exists. Anything opaque left behind shows
        # up as the square frame the user keeps (rightly) complaining about.
        box = view.parentWidget()
        self._popup_box = box
        if box is not None:
            # QComboBoxPrivateContainer is a QFrame: even with a fully
            # transparent palette its paintEvent still draws the StyledPanel
            # rect — the square dark frame around the rounded panel. Strip the
            # frame geometry and swallow the container's own paint entirely;
            # its children (the view) keep painting normally, so the view's
            # rounded panel becomes the popup's only visible edge.
            box.setFrameShape(QFrame.NoFrame)
            box.setLineWidth(0)
            box.setAttribute(Qt.WA_TranslucentBackground)
            box.setAttribute(Qt.WA_NoSystemBackground)
            box.setAutoFillBackground(False)
            pal = box.palette()
            pal.setColor(QPalette.Window, Qt.transparent)
            box.setPalette(pal)
            lay = box.layout()
            if lay is not None:
                lay.setContentsMargins(0, 0, 0, 0)
            box.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if obj is self._popup_box and event.type() == QEvent.Paint:
            return True          # the host window contributes no pixels of its own
        return super().eventFilter(obj, event)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        event.ignore()


class ChipButton(QPushButton):
    """A checkable export-format chip that paints its own rounded bevel.

    Qt 6.11 on macOS silently drops QSS border-radius on QPushButton: the
    styled background and border still paint, but as a square rect (QFrame
    is unaffected). So the rounded chip shape is painted here from the live
    theme scheme; theme.py keeps owning text color / font / padding via the
    #chip rules, with background and border left to this paintEvent.
    """

    RADIUS = 12.0

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt naming)
        c = theme.current_scheme()
        on = self.isChecked()
        edge = c["accent_border"] if on else (c["line"] if self.isEnabled() else c["line2"])
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if on:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(c["accent_soft"]))
            p.drawRoundedRect(rect, self.RADIUS, self.RADIUS)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(edge), 1))
        p.drawRoundedRect(rect, self.RADIUS, self.RADIUS)
        p.end()
        super().paintEvent(event)   # label only; the QSS bevel is transparent
