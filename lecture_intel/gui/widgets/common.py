"""Shared widget subclasses."""
from __future__ import annotations

from PySide6.QtCore import Qt
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
        # The popup is a top-level window and macOS gives it a square, opaque
        # surface. Translucency must be requested before that window is
        # created — i.e. here, while the container still exists unwrapped —
        # or the square corners paint over the rounded QSS background. With it
        # set early, the container needs no border of its own: the rounded
        # edge is just where its background stops, and the selected row's ring
        # is the only outline left inside.
        box = view.parentWidget()
        if box is not None:
            box.setAttribute(Qt.WA_TranslucentBackground)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        event.ignore()
