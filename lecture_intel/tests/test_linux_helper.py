"""End-to-end tests for native/SystemAudioRecorderLinux.py against fake backends.

The fakes stand in for pw-record/parec on PATH: they stream raw s16le stereo
PCM to stdout in 10 ms chunks whose samples all hold the chunk counter, and on
SIGINT flush one last chunk of TAIL before exiting (pw-record exits 1 there).
So the WAV shows exactly which chunks were kept: a jump in the counter is the
pause, and TAIL as the final frame proves the end was drained.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Linux helper needs POSIX signals and shebang backends")

HELPER = Path(__file__).resolve().parent.parent / "native" / "SystemAudioRecorderLinux.py"
TAIL = 32000

FAKE_BACKEND = """#!{python}
import signal, struct, sys, time
from pathlib import Path
Path({argv_log!r}).write_text("\\n".join(sys.argv[1:]))
if {fail_code} is not None:
    sys.exit({fail_code})
out = sys.stdout.buffer
def chunk(value):
    return struct.pack("<h", value) * (480 * 2)
def finish(*_):
    out.write(chunk({tail}))
    out.flush()
    sys.exit({exit_code})
signal.signal(signal.SIGINT, finish)
n = 0
while True:
    out.write(chunk(n % 30000))
    out.flush()
    n += 1
    time.sleep(0.01)
"""


def _install_fake(bin_dir: Path, name: str, *, fail_code=None, exit_code=0) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    argv_log = bin_dir / f"{name}.argv"
    script = bin_dir / name
    script.write_text(FAKE_BACKEND.format(
        python=sys.executable, argv_log=str(argv_log), fail_code=fail_code,
        tail=TAIL, exit_code=exit_code))
    script.chmod(0o755)
    return argv_log


def _start(bin_dir: Path, out: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(HELPER), str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={"PATH": str(bin_dir)},
    )


def _send(proc: subprocess.Popen[str], command: str) -> None:
    proc.stdin.write(command + "\n")
    proc.stdin.flush()


def _finish(proc: subprocess.Popen[str]) -> tuple[int, str]:
    _send(proc, "STOP")
    code = proc.wait(timeout=15)
    return code, proc.stderr.read()


def _chunk_values(path: Path) -> list[int]:
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (48000, 2, 2)
        frames = wav.readframes(wav.getnframes())
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    values: list[int] = []
    for sample in samples:
        if not values or values[-1] != sample:
            values.append(sample)
    return values


def test_capabilities_advertise_the_stdin_protocol():
    result = subprocess.run([sys.executable, str(HELPER), "--capabilities"],
                            capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["STDIN_CONTROL_V1", "PAUSE", "RESUME", "STOP"]


def test_records_a_wav_and_keeps_the_end_of_the_stream(tmp_path):
    bin_dir = tmp_path / "bin"
    argv_log = _install_fake(bin_dir, "pw-record", exit_code=1)
    out = tmp_path / "out.wav"
    proc = _start(bin_dir, out)
    time.sleep(0.8)
    code, err = _finish(proc)

    assert code == 0, err                      # pw-record's exit 1 on SIGINT is not a failure
    assert "RECORDING" in err and "STOPPED" in err
    values = _chunk_values(out)
    assert values[-1] == TAIL                  # what the backend flushed on stop was drained
    assert values[:-1] == sorted(values[:-1])  # chunks arrive in order, none duplicated
    argv = argv_log.read_text().splitlines()
    assert argv[-1] == "-"                     # raw PCM on stdout ...
    assert "--container" not in argv           # ... not a flag PipeWire 1.0.x lacks


def test_pause_drops_the_audio_played_while_paused(tmp_path):
    bin_dir = tmp_path / "bin"
    _install_fake(bin_dir, "pw-record")
    out = tmp_path / "out.wav"
    proc = _start(bin_dir, out)
    time.sleep(0.6)
    _send(proc, "PAUSE")
    time.sleep(0.6)
    _send(proc, "RESUME")
    time.sleep(0.4)
    code, err = _finish(proc)

    assert code == 0, err
    assert "PAUSED" in err and "RESUMED" in err
    values = _chunk_values(out)[:-1]
    gaps = [b - a for a, b in zip(values, values[1:])]
    # ~60 chunks were produced during the pause; they must not be in the file,
    # while everything before and after it is (the backend was never stopped).
    assert max(gaps) >= 30
    assert sum(1 for gap in gaps if gap > 1) == 1
    assert values[-1] - values[0] + 1 - len(values) >= 30


def test_falls_back_to_parec_when_pw_record_fails(tmp_path):
    bin_dir = tmp_path / "bin"
    _install_fake(bin_dir, "pw-record", fail_code=2)
    parec_argv = _install_fake(bin_dir, "parec")
    out = tmp_path / "out.wav"
    proc = _start(bin_dir, out)
    time.sleep(1.2)                            # two startup probes
    code, err = _finish(proc)

    assert code == 0, err
    assert "START_ERROR" not in err and "ERROR" not in err.replace("START_ERROR", "")
    values = _chunk_values(out)
    assert values[0] == 0 and values[-1] == TAIL  # only parec's stream, from its start
    argv = parec_argv.read_text().splitlines()
    assert "--raw" in argv
    assert "--latency-msec=50" in argv         # small server buffer: no lost ending


def test_reports_a_start_error_without_any_backend(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    out = tmp_path / "out.wav"
    proc = _start(bin_dir, out)
    code = proc.wait(timeout=10)
    err = proc.stderr.read()
    assert code == 1
    assert "START_ERROR" in err and "pw-record" in err


def test_reports_every_backend_failure(tmp_path):
    bin_dir = tmp_path / "bin"
    _install_fake(bin_dir, "pw-record", fail_code=2)
    _install_fake(bin_dir, "parec", fail_code=3)
    proc = _start(bin_dir, tmp_path / "out.wav")
    code = proc.wait(timeout=10)
    err = proc.stderr.read()
    assert code == 1
    assert "pw-record: exited with status 2" in err
    assert "parec: exited with status 3" in err
