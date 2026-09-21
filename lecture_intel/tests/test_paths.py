"""
Where Recorder writes user data.

`clone_root` and `default_output_root` are already covered; these tests cover the
other half — `data_root` / `record_dir` — and pin the two promises the layout
makes: recordings live somewhere the user can find in Finder, and both helpers
create their folder on demand rather than failing on a fresh machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import paths  # noqa: E402


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect ~ so these tests never touch the real ~/Documents."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_data_root_is_a_folder_the_user_can_find(fake_home):
    root = paths.data_root()
    assert root == fake_home / "Documents" / "recorder" / "Recorder"
    assert root.is_dir()


def test_data_root_is_created_on_demand(fake_home):
    assert not (fake_home / "Documents").exists()
    assert paths.data_root().is_dir()          # no exception on a fresh machine
    assert paths.data_root().is_dir()          # and it is idempotent


def test_record_dir_nests_under_data_root(fake_home):
    assert paths.record_dir() == paths.data_root() / "record"
    assert paths.record_dir().is_dir()


def test_record_dir_is_created_on_demand(fake_home):
    d = paths.record_dir()
    assert d.is_dir() and list(d.iterdir()) == []


def test_record_dir_is_stable_across_calls(fake_home):
    assert paths.record_dir() == paths.record_dir()


def test_data_root_never_lands_in_an_installed_app_folder(fake_home):
    """The app bundle lives in ~/Library/Application Support; user recordings
    must not be buried there."""
    text = str(paths.data_root())
    assert "Application Support" not in text
    assert "/Applications" not in text
