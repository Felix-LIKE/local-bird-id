#!/usr/bin/env python3
"""Regression test: the ONNX multi-bird detector download + runtime contract.

Pins the behavior `birdid --download-detector` and `--status` added for the
viewer's multi-bird gating:

1) `_onnxruntime_importable()` — reports whether onnxruntime imports in THIS
   interpreter, without loading it (safe in --status).

2) `_onnx_detector_ready()` — the "automated multi-bird detection is fully
   usable" signal behind `backends.onnx_detector`. It must be the AND of
   "model file downloaded" and "onnxruntime importable": a downloaded model
   with no runtime does NOT count (the viewer keeps --multi gated until both
   are in place). Must never raise.

3) `_ensure_onnxruntime()` — the auto-install step `--download-detector` runs.
   No-op (True) when onnxruntime already imports; otherwise it must run
   ``sys.executable -m pip install --disable-pip-version-check onnxruntime``
   (SAME interpreter, check=False so pip's exit code is read, not raised).
   A failed pip run must return False WITHOUT raising — a download must never
   break because the runtime could not be installed.

Runs standalone (no pytest):  python3 test_detector_runtime.py
No network, no model load, no real pip install (all mocked).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import birdid  # noqa: E402  (sys.path must be set first)

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(name)
        print(f"FAIL  {name}  {detail}".rstrip())
    else:
        print(f"PASS  {name}")


def run():
    print("\n== Detector download + runtime contract ==")

    # --- 1) _onnxruntime_importable: bool, no load, equals the real probe ---
    real = importlib.util.find_spec("onnxruntime") is not None
    got = birdid._onnxruntime_importable()
    check("onnxruntime importable reports the real interpreter probe",
          isinstance(got, bool) and got == real,
          f"got={got} real={real}")

    # --- 2) _onnx_detector_ready: model AND runtime, never raises ---
    with mock.patch.object(birdid, "_hub_cached_file", return_value=True), \
         mock.patch.object(birdid, "_onnxruntime_importable", return_value=True):
        check("detector ready when model + runtime present",
              birdid._onnx_detector_ready() is True)
    with mock.patch.object(birdid, "_hub_cached_file", return_value=False), \
         mock.patch.object(birdid, "_onnxruntime_importable", return_value=True):
        check("detector NOT ready when model missing",
              birdid._onnx_detector_ready() is False)
    with mock.patch.object(birdid, "_hub_cached_file", return_value=True), \
         mock.patch.object(birdid, "_onnxruntime_importable", return_value=False):
        check("detector NOT ready when runtime missing (model alone is not enough)",
              birdid._onnx_detector_ready() is False)

    # --- 3) _ensure_onnxruntime: no-op when already importable ---
    with mock.patch.object(birdid, "_onnxruntime_importable", return_value=True), \
         mock.patch("subprocess.run") as fake_run:
        ok = birdid._ensure_onnxruntime()
        check("already importable -> True and NO pip run",
              ok is True and not fake_run.called)

    # --- missing runtime -> runs pip via sys.executable (same interpreter) ---
    state = {"importable": False}
    captured = {}

    def flip():
        return state["importable"]

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        state["importable"] = True          # the pip run "makes it importable"
        return mock.Mock(returncode=0)

    with mock.patch.object(birdid, "_onnxruntime_importable", side_effect=flip), \
         mock.patch("subprocess.run", side_effect=fake_run):
        ok = birdid._ensure_onnxruntime()
        check("missing runtime -> pip run then True", ok is True, f"ok={ok}")
        check("pip argv = sys.executable -m pip install ... onnxruntime",
              captured.get("cmd") == [sys.executable, "-m", "pip", "install",
                                      "--disable-pip-version-check",
                                      "onnxruntime"],
              str(captured.get("cmd")))
        check("pip run uses check=False",
              captured.get("kw", {}).get("check") is False,
              str(captured.get("kw")))

    # --- pip exits non-zero -> False, no raise ---
    with mock.patch.object(birdid, "_onnxruntime_importable", return_value=False), \
         mock.patch("subprocess.run",
                    return_value=mock.Mock(returncode=1)):
        ok = birdid._ensure_onnxruntime()
        check("pip failure -> False, no raise", ok is False)

    # --- subprocess itself raises -> False, no raise ---
    with mock.patch.object(birdid, "_onnxruntime_importable", return_value=False), \
         mock.patch("subprocess.run", side_effect=OSError("no pip")):
        ok = birdid._ensure_onnxruntime()
        check("subprocess error -> False, no raise", ok is False)

    print("\n============================================")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
