"""
The one place the app opens its preferences.

Every QSettings the app reads or writes comes from ``app_settings()``. On macOS
the native format is CFPreferences (``~/Library/Preferences/com.lucaslab.Recorder.plist``)
and ``QSettings.setPath`` has no effect on it, so the only reliable way to keep
tests out of the user's real preferences is to point this helper at a file:
tests set ``SETTINGS_FILE`` to an INI file under their tmp dir.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings

ORGANIZATION = "LucasLab"
APPLICATION = "Recorder"

# When set, preferences live in this INI file instead of the native store.
SETTINGS_FILE: str | None = None


def app_settings() -> QSettings:
    if SETTINGS_FILE:
        return QSettings(SETTINGS_FILE, QSettings.IniFormat)
    return QSettings(ORGANIZATION, APPLICATION)
