"""
File drag-and-drop zone widget.

Supports drag-and-drop from Finder and click-to-browse file selection.
Displays the selected file name and size.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

SUPPORTED_EXTENSIONS: set[str] = {
    ".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus",
}


class DropZone(QWidget):
    """A drag-and-drop zone that accepts audio files."""

    file_selected = Signal(str)  # emitted with absolute file path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        self._selected_path: str | None = None
        self._build_ui()

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        self._icon_label = QLabel("\U0001F399️")  # 🎙️
        self._icon_label.setAlignment(Qt.AlignCenter)
        self._icon_label.setStyleSheet("font-size: 36px;")

        self._hint_label = QLabel("Drop recording here\nor click to browse")
        self._hint_label.setAlignment(Qt.AlignCenter)
        self._hint_label.setStyleSheet("color: gray; font-size: 13px;")

        self._ext_label = QLabel("m4a · mp3 · wav · flac · aac")
        self._ext_label.setAlignment(Qt.AlignCenter)
        self._ext_label.setStyleSheet("color: lightgray; font-size: 11px;")

        self._file_label = QLabel("")
        self._file_label.setAlignment(Qt.AlignCenter)
        self._file_label.setStyleSheet(
            "color: #007AFF; font-size: 12px; font-weight: 500;"
        )
        self._file_label.setVisible(False)

        layout.addWidget(self._icon_label)
        layout.addWidget(self._hint_label)
        layout.addWidget(self._ext_label)
        layout.addWidget(self._file_label)

        self._update_style(hover=False)
        self.setCursor(Qt.PointingHandCursor)

    # ── Styling ───────────────────────────────────────────────

    def _update_style(self, hover: bool, has_file: bool = False) -> None:
        if hover:
            border_color = "#007AFF"
            bg_color = "rgba(0, 122, 255, 0.05)"
        elif has_file:
            border_color = "#34C759"
            bg_color = "rgba(52, 199, 89, 0.05)"
        else:
            border_color = "#C7C7CC"
            bg_color = "rgba(0,0,0,0.02)"

        self.setStyleSheet(f"""
            DropZone {{
                border: 2px dashed {border_color};
                border-radius: 12px;
                background: {bg_color};
            }}
        """)

    # ── Drag & drop events ────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            if any(Path(p).suffix.lower() in SUPPORTED_EXTENSIONS for p in paths):
                event.acceptProposedAction()
                self._update_style(hover=True)
                return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._update_style(
            hover=False, has_file=self._selected_path is not None
        )

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
                self._set_file(path)
                break
        self._update_style(hover=False, has_file=True)

    # ── Click to browse ───────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Recording",
                str(Path.home() / "Desktop"),
                "Audio Files (*.m4a *.mp3 *.wav *.flac *.aac *.ogg *.opus)",
            )
            if path:
                self._set_file(path)

    # ── Internal ──────────────────────────────────────────────

    def set_file(self, path: str) -> None:
        """Public method to programmatically set the selected file."""
        self._set_file(path)

    def _set_file(self, path: str) -> None:
        self._selected_path = path
        p = Path(path)
        size_mb = p.stat().st_size / 1_048_576
        self._file_label.setText(
            f"\U0001F4C4 {p.name}  ({size_mb:.1f} MB)"
        )
        self._file_label.setVisible(True)
        self._hint_label.setText("File ready — click Process to start")
        self._update_style(hover=False, has_file=True)
        self.file_selected.emit(path)

    def reset(self) -> None:
        """Clear selection and restore initial state."""
        self._selected_path = None
        self._hint_label.setText("Drop recording here\nor click to browse")
        self._file_label.setVisible(False)
        self._update_style(hover=False, has_file=False)

    @property
    def selected_path(self) -> str | None:
        return self._selected_path
