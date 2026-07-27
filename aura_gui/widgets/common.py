"""Shared widget subclasses."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox


class NoScrollComboBox(QComboBox):
    """A combo box that ignores mouse-wheel scrolling.

    Qt changes the selected item when the wheel scrolls over a combo box, even
    without clicking. We swallow the wheel event so scrolling moves the page;
    clicking still opens the dropdown normally.
    """

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        event.ignore()
