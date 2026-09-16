"""Shared widget subclasses."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QComboBox, QListView, QStyledItemDelegate


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
        if box is not None:
            box.setAttribute(Qt.WA_TranslucentBackground)
            box.setAttribute(Qt.WA_NoSystemBackground)
            box.setAutoFillBackground(False)
            pal = box.palette()
            pal.setColor(QPalette.Window, Qt.transparent)
            box.setPalette(pal)
            lay = box.layout()
            if lay is not None:
                lay.setContentsMargins(0, 0, 0, 0)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        event.ignore()
