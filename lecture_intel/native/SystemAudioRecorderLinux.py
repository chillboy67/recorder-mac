#!/usr/bin/env python3
"""Capture the default Linux output monitor to a PCM WAV file.

This is the Linux native-helper boundary used by ``core.sysaudio``.  It keeps
platform/backend details out of the Qt process and exposes the same line-based
stdin protocol as the macOS and Windows helpers.

The backend (``pw-record`` or ``parec``) streams raw PCM to our stdout pipe and
this helper writes the WAV itself.  That keeps the contract the other helpers
follow: while paused the audio is still drained but dropped, so nothing played
during a pause reaches the file.  (Stopping the backend with SIGSTOP instead
lets the sound server buffer the paused audio and deliver it on resume.)  It
also avoids version-specific output flags: ``pw-record --container`` only
exists from PipeWire 1.2, so Ubuntu 24.04's 1.0.x rejects it.
"""
from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import BinaryIO, Optional

PROTOCOL_MARKER = "STDIN_CONTROL_V1"
_COMMANDS = "PAUSE RESUME STOP"

SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # s16le
_FRAME_BYTES = CHANNELS * SAMPLE_WIDTH
_CHUNK_BYTES = 4096 * _FRAME_BYTES
# Backend argument/session errors are normally reported immediately.
_STARTUP_PROBE_S = 0.35


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _backend_commands() -> list[list[str]]:
    commands: list[list[str]] = []
    pw_record = shutil.which("pw-record")
    if pw_record:
        commands.append([
            pw_record,
            "--properties", '{"stream.capture.sink": true}',
            "--rate", str(SAMPLE_RATE),
            "--channels", str(CHANNELS),
            "--channel-map", "stereo",
            "--format", "s16",
            "-",
        ])

    parec = shutil.which("parec")
    if parec:
        commands.append([
            parec,
            "--device=@DEFAULT_MONITOR@",
            "--raw",
            f"--rate={SAMPLE_RATE}",
            f"--channels={CHANNELS}",
            "--format=s16le",
            # The default record latency buffers ~2 s in the server, which is
            # lost when the stream is interrupted: the end of the recording.
            "--latency-msec=50",
        ])
    return commands


class CaptureProcess:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self._writer: Optional[wave.Wave_write] = None
        self._reader: Optional[threading.Thread] = None
        self._paused = threading.Event()
        self._recording = False
        self._stopping = False

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def start(self) -> None:
        commands = _backend_commands()
        if not commands:
            raise RuntimeError(
                "no Linux system-audio backend found; install pipewire-bin "
                "(pw-record) or pulseaudio-utils (parec)"
            )
        failures: list[str] = []
        for command in commands:
            name = Path(command[0]).name
            try:
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=None,
                )
            except OSError as exc:
                failures.append(f"{name}: {exc}")
                continue

            self._open_writer()
            self.proc = proc
            self._reader = threading.Thread(
                target=self._pump, args=(proc.stdout,), daemon=True)
            self._reader.start()
            time.sleep(_STARTUP_PROBE_S)
            code = proc.poll()
            if code is None:
                self._recording = True
                _log("RECORDING")
                return
            self._reader.join(timeout=2)
            self._close_writer()
            self.proc = None
            failures.append(f"{name}: exited with status {code}")
        raise RuntimeError("; ".join(failures))

    def _open_writer(self) -> None:
        writer = wave.open(self.output_path, "wb")
        writer.setnchannels(CHANNELS)
        writer.setsampwidth(SAMPLE_WIDTH)
        writer.setframerate(SAMPLE_RATE)
        self._writer = writer

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _pump(self, stream: BinaryIO) -> None:
        """Copy backend PCM into the WAV, dropping it while paused."""
        pending = b""
        while True:
            chunk = stream.read1(_CHUNK_BYTES)
            if not chunk:
                break
            data = pending + chunk
            usable = len(data) - len(data) % _FRAME_BYTES
            pending = data[usable:]
            if usable and not self._paused.is_set() and self._writer is not None:
                # writeframes (not writeframesraw) keeps the header's sizes
                # current, so a helper killed mid-capture still leaves a WAV
                # that decoders read in full.
                self._writer.writeframes(data[:usable])
        if self._recording and not self._stopping and self.proc is not None:
            _log(f"ERROR capture backend exited with status {self.proc.poll()}")

    def pause(self) -> None:
        if self.proc is None or self.proc.poll() is not None or self.paused:
            return
        self._paused.set()
        _log("PAUSED")

    def resume(self) -> None:
        if self.proc is None or self.proc.poll() is not None or not self.paused:
            return
        self._paused.clear()
        _log("RESUMED")

    def stop(self) -> int:
        proc = self.proc
        if proc is None:
            return 0
        self._stopping = True
        exited_early = proc.poll() is not None
        if not exited_early:
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
        # Drain what the backend flushed on SIGINT: the end of the recording.
        if self._reader is not None:
            self._reader.join(timeout=5)
        self._close_writer()
        self.proc = None
        _log("STOPPED")
        # pw-record exits 1 on SIGINT; only a backend that died on its own is
        # a failure.
        return (proc.returncode or 0) if exited_early else 0


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
