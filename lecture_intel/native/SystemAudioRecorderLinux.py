#!/usr/bin/env python3
"""Capture the default Linux output monitor to a PCM WAV file.

This is the Linux native-helper boundary used by ``core.sysaudio``.  It keeps
platform/backend details out of the Qt process and exposes the same line-based
stdin protocol as the macOS and Windows helpers.
"""
from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

PROTOCOL_MARKER = "STDIN_CONTROL_V1"
_COMMANDS = "PAUSE RESUME STOP"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _backend_commands(output_path: str) -> list[list[str]]:
    commands: list[list[str]] = []
    pw_record = shutil.which("pw-record")
    if pw_record:
        commands.append([
            pw_record,
            "--properties", '{"stream.capture.sink": true}',
            "--rate", "48000",
            "--channels", "2",
            "--channel-map", "stereo",
            "--format", "s16",
            "--container", "wav",
            output_path,
        ])

    parec = shutil.which("parec")
    if parec:
        commands.append([
            parec,
            "--device=@DEFAULT_MONITOR@",
            "--file-format=wav",
            "--rate=48000",
            "--channels=2",
            "--format=s16le",
            output_path,
        ])
    return commands


def _start_backend(output_path: str) -> subprocess.Popen[bytes]:
    failures: list[str] = []
    for command in _backend_commands(output_path):
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=None,
            )
        except OSError as exc:
            failures.append(f"{Path(command[0]).name}: {exc}")
            continue

        # Backend argument/session errors are normally reported immediately.
        time.sleep(0.35)
        code = proc.poll()
        if code is None:
            return proc
        failures.append(f"{Path(command[0]).name}: exited with status {code}")

    if not failures:
        raise RuntimeError(
            "no Linux system-audio backend found; install pipewire-bin "
            "(pw-record) or pulseaudio-utils (parec)"
        )
    raise RuntimeError("; ".join(failures))


class CaptureProcess:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.paused = False

    def start(self) -> None:
        self.proc = _start_backend(self.output_path)
        _log("RECORDING")

    def pause(self) -> None:
        if self.proc is None or self.proc.poll() is not None or self.paused:
            return
        self.proc.send_signal(signal.SIGSTOP)
        self.paused = True
        _log("PAUSED")

    def resume(self) -> None:
        if self.proc is None or self.proc.poll() is not None or not self.paused:
            return
        self.proc.send_signal(signal.SIGCONT)
        self.paused = False
        _log("RESUMED")

    def stop(self) -> int:
        proc = self.proc
        if proc is None:
            return 0
        if proc.poll() is None:
            if self.paused:
                proc.send_signal(signal.SIGCONT)
                self.paused = False
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        code = proc.returncode or 0
        self.proc = None
        _log("STOPPED")
        return code


def main(argv: list[str]) -> int:
    if argv == ["--capabilities"]:
        print(f"{PROTOCOL_MARKER} {_COMMANDS}")
        return 0
    if len(argv) != 1:
        _log("usage: SystemAudioRecorderLinux.py <output.wav>")
        return 2

    capture = CaptureProcess(argv[0])
    try:
        capture.start()
    except Exception as exc:
        _log(f"START_ERROR {exc}")
        return 1

    def handle_termination(_signum, _frame) -> None:
        capture.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)

    try:
        for raw in sys.stdin:
            command = raw.strip().upper()
            if command == "PAUSE":
                capture.pause()
            elif command == "RESUME":
                capture.resume()
            elif command == "STOP":
                return capture.stop()
        # Parent disappeared or closed the control pipe: do not orphan capture.
        return capture.stop()
    except (BrokenPipeError, KeyboardInterrupt):
        return capture.stop()
    except Exception as exc:
        _log(f"ERROR {exc}")
        capture.stop()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
