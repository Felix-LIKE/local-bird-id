#!/usr/bin/env python3
"""Regression test: the ONNX multi-bird detector download + runtime contract.

Pins the behavior `birdid --download-detector` and `--status` added for the
viewer's multi-bird gating:

1) `_onnxruntime_importable()` — reports whether onnxruntime imports in THIS
   interpreter, without loading it (safe in --status).

2) `_onnx_detector_ready()` — the "automated multi-bird box detection is
   available" signal behind `backends.onnx_detector`. It must be the AND of
   "model file downloaded" and "onnxruntime importable": a downloaded model
   with no runtime does NOT count (no automated box source without it). It
   does NOT gate --multi (which works on user-drawn boxes); it only says
   whether --multi can AUTO-detect boxes. Must never raise.

3) `_ensure_onnxruntime()` — the auto-install step `--download-detector` runs.
   No-op (True) when onnxruntime already imports; otherwise it must run
   ``sys.executable -m pip install --disable-pip-version-check onnxruntime``
   (SAME interpreter, check=False so pip's exit code is read, not raised).
   A failed pip run must return False WITHOUT raising — a download must never
   break because the runtime could not be installed.

4) `make_detector()` — the region-proposal backend for --multi. The ONNX
   detector is the ONLY automated proposal: without it (and without
   --no-detector-model) make_detector returns None so --multi classifies only
   user-drawn boxes; the built-in zero-shot BioCLIP detector is available
   ONLY via the explicit --no-detector-model opt-in; the ONNX path is used
   when the model file + runtime are present.

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

    # --- 4) make_detector: ONNX is the only automated proposal ---
    from types import SimpleNamespace
    DummyClassifier = type("DummyClassifier", (), {})
    args_on = SimpleNamespace(no_detector_model=False)
    args_off = SimpleNamespace(no_detector_model=True)

    # No ONNX (model missing) + no --no-detector-model -> None (manual boxes only)
    with mock.patch.object(birdid, "_detector_onnx_path", return_value=None):
        d = birdid.make_detector(args_on, DummyClassifier())
        check("no ONNX path + no opt-in -> make_detector returns None",
              d is None, f"got={d!r}")

    # ONNX session failure -> None (never a silent zero-shot fallback)
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("session failed")
    with mock.patch.object(birdid, "_detector_onnx_path",
                           return_value=Path("x.onnx")), \
         mock.patch.object(birdid, "OnnxDetector", Boom):
        d = birdid.make_detector(args_on, DummyClassifier())
        check("ONNX session failure -> None (no silent zero-shot fallback)",
              d is None, f"got={d!r}")

    # --no-detector-model -> the zero-shot BioCLIP detector (explicit opt-in)
    d = birdid.make_detector(args_off, DummyClassifier())
    check("--no-detector-model -> BioClipCropDetector",
          isinstance(d, birdid.BioClipCropDetector),
          f"got={type(d).__name__ if d else None}")

    # ONNX path + model file -> the ONNX detector with a note
    with mock.patch.object(birdid, "_detector_onnx_path",
                           return_value=Path("model_int8.onnx")), \
         mock.patch.object(birdid, "OnnxDetector") as FakeOnnx:
        fake_det = FakeOnnx.return_value = mock.Mock()
        d = birdid.make_detector(args_on, DummyClassifier())
        check("ONNX path present -> returns the ONNX detector",
              d is fake_det and FakeOnnx.call_args[1].get("onnx_path")
              == Path("model_int8.onnx"),
              f"d={d is fake_det} onnx_path={FakeOnnx.call_args}")
        check("ONNX detector gets a descriptive note",
              getattr(d, "note", "").startswith("dedicated ONNX detector"),
              getattr(d, "note", ""))

    # classifier None -> None regardless of args
    with mock.patch.object(birdid, "_detector_onnx_path", return_value=None):
        check("classifier None -> None", birdid.make_detector(args_on, None) is None)

    print("\n============================================")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
