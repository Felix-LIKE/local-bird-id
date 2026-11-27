#!/usr/bin/env python3
"""Cross-contract regression test: the viewer <-> pipeline sidecar contract.

The viewer (birdid-viewer) writes bird:* to sidecars with exiftool, the
pipeline writes with lxml. Both must be able to read what the other wrote,
or a box saved in the viewer is invisible to --multi (and vice versa). This
test runs the viewer's EXACT exiftool command lines — the same `-config`
namespace file, the same write tags, the same `-struct -G1 -j` read flags —
against real sidecars and asserts the pipeline's readers agree.

Two directions:

1) VIEWER -> PIPELINE: replicate the viewer's manual-box-save write
   (`exiftool::manual_regions_args` + `multi_flag_args`) on a sidecar, then
   assert `birdid._read_regions` / `birdid.sidecar_multi` read the boxes, the
   per-box manual provenance and the flag back.

2) PIPELINE -> VIEWER: write a multi-bird sidecar with the pipeline's own
   `write_sidecar` + `_set_regions` (what `--multi` produces), then run the
   viewer's exact READ command and assert the `XMP-bird:Regions` struct array
   the viewer's `map_bird` consumes.

The config file used for writes is `docs/birdid.config` — the SAME content the
viewer bundles (`src-tauri/birdid.config`); a divergence breaks the struct
definition (regions would not come back as objects). Requires exiftool on PATH;
skips cleanly when missing. lxml is a pipeline dependency (pulled by bootstrap).
"""
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import birdid  # noqa: E402  (sys.path must be set first)

CONFIG = _HERE.parent / "docs" / "birdid.config"

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS: {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL: {name} {detail}")


def viewer_write_args(sidecar: Path, tag_assignments: list[str]) -> list[str]:
    """The viewer's write surface: `exiftool -config <path> -m` then the tag
    writes, then `-overwrite_original <sidecar>`. Mirrors `exiftool::*_args`
    (which always pass -m to tolerate a duplicate XMP property)."""
    return ["exiftool", "-config", str(CONFIG), "-m", *tag_assignments,
            "-overwrite_original", str(sidecar)]


def viewer_read(sidecar: Path) -> dict:
    """The viewer's exact READ command (`exiftool_cmd`): -config - -G1 -j
    -struct -XMP-bird:all -XMP-dwc:all, config piped on stdin."""
    out = subprocess.run(
        ["exiftool", "-config", "-", "-G1", "-j", "-struct",
         "-XMP-bird:all", "-XMP-dwc:all", str(sidecar)],
        input=CONFIG.read_text(), capture_output=True, text=True,
    )
    assert out.returncode in (0, 2), f"read failed: {out.stderr}"
    arr = json.loads(out.stdout)
    return arr[0]


def write_tiny_jpeg(path: Path) -> Path:
    """A real (tiny) JPEG so exiftool can use a fresh sidecar's `-o` create form
    the same way the viewer does when the capture has no sidecar yet."""
    # 1x1 JPEG (smallest valid file).
    path.write_bytes(bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
        "070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c"
        "1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100"
        "ffc4001f0000010501010101010100000000000000000102030405060708090a0bff"
        "c400b5100002010303020403050504040000017d0102030004110512213141061351"
        "6107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728"
        "292a3435363738393a434445464748494a535455565758595a636465666768696a73"
        "7475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab"
        "2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7"
        "e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000"
        "000102030405060708090a0bffc400b5110002010204040304070504040001027700"
        "0102031104052131061241510761711322328108144291a1b1c109233352f0156272"
        "d10a162434e125f11718191a262728292a35363738393a434445464748494a535455"
        "565758595a636465666768696a737475767778797a82838485868788898a92939495"
        "969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3"
        "d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c0301000210"
        "03110000000000000000000000ffd9"))
    return path


def run():
    if shutil.which("exiftool") is None:
        print("SKIP test_viewer_contract.py: exiftool not on PATH")
        return 0
    if not CONFIG.exists():
        print(f"SKIP test_viewer_contract.py: {CONFIG} not found")
        return 0

    with tempfile.TemporaryDirectory(prefix="birdid-contract-") as tmp:
        tmp = Path(tmp)

        print("\n== V1: VIEWER box-save write -> pipeline reads the boxes ==")
        # The viewer's manual box save (`manual_regions_args`): flag multi,
        # write the count, replace the region list, drop the stale mwg-rs
        # mirror, tolerate duplicates. Two boxes: one user-drawn (manual=true),
        # one untouched model detection (manual=false).
        photo = tmp / "B1.jpg"
        write_tiny_jpeg(photo)
        sidecar = tmp / "B1.xmp"
        subprocess.run(
            viewer_write_args(sidecar, [
                "-XMP-bird:multiBird=true",
                "-XMP-bird:birdCount=2",
                "-XMP-bird:regions=",
                "-XMP-bird:regions=[{x=0.4,y=0.4,w=0.2,h=0.3,manual=true},"
                "{x=0.6,y=0.1,w=0.1,h=0.1,manual=false}]",
                "-XMP-mwg-rs:All=",
            ]),
            check=True, capture_output=True,
        )
        check("write created the sidecar", sidecar.exists())
        regs = birdid._read_regions(sidecar)
        check("pipeline reads both boxes",
              len(regs) == 2, f"len={len(regs)} {regs}")
        if len(regs) == 2:
            check("box 0 geometry + manual provenance",
                  regs[0]["x"] == 0.4 and regs[0]["y"] == 0.4
                  and regs[0]["w"] == 0.2 and regs[0]["h"] == 0.3
                  and regs[0]["manual"] is True,
                  str(regs[0]))
            check("box 1 keeps manual=false",
                  regs[1]["x"] == 0.6 and regs[1]["y"] == 0.1
                  and regs[1]["w"] == 0.1 and regs[1]["h"] == 0.1
                  and regs[1]["manual"] is False,
                  str(regs[1]))
            check("unclassified boxes default to status done",
                  regs[0]["status"] == "done" and regs[1]["status"] == "done")
        check("multiBird flag read by the pipeline",
              birdid.sidecar_multi(sidecar) is True)

        print("\n== V2: VIEWER read of PIPELINE-written multi sidecar ==")
        # The pipeline's own `--multi` output: write_sidecar + _set_regions
        # (lxml, nested rdf:li > rdf:Description bird:*). The viewer must read
        # the same regions back via its exact `-struct -G1 -j` command.
        ps = tmp / "P1.xmp"
        result = {"species": "Great Cormorant", "scientific": "Phalacrocorax carbo",
                  "confidence": 0.92,
                  "alternatives": ["European Shag", "Caspian Gull"],
                  "reasoning": "whole image"}
        class Cfg:
            dc_fields = False
        birdid.write_sidecar(ps, result, "EUROPE", 11, datetime(2026, 11, 21),
                             "done", Cfg(), scientific="Phalacrocorax carbo",
                             regions=[
                                 {"x": 0.10, "y": 0.10, "w": 0.40, "h": 0.30,
                                  "manual": False, "topSpecies": "Great Cormorant",
                                  "topConfidence": 0.97, "status": "done",
                                  "ranked": [("Great Cormorant", 0.97)],
                                  "reasoning": "region 1"},
                                 {"x": 0.60, "y": 0.10, "w": 0.30, "h": 0.30,
                                  "manual": False, "topSpecies": "Caspian Gull",
                                  "topConfidence": 0.90, "status": "done",
                                  "ranked": [("Caspian Gull", 0.90)],
                                  "reasoning": "region 2"},
                             ])
        raw = viewer_read(ps)
        r = raw.get("XMP-bird:Regions")
        check("viewer read returns a Regions struct array",
              isinstance(r, list) and len(r) == 2, f"{type(r)} {r}")
        if isinstance(r, list) and len(r) == 2:
            check("viewer reads box geometry",
                  r[0].get("X") == 0.1 and r[0].get("Y") == 0.1
                  and r[0].get("W") == 0.4 and r[0].get("H") == 0.3,
                  str(r[0]))
            check("viewer reads per-region candidates (struct)",
                  r[0].get("Species") == ["Great Cormorant"]
                  and r[0].get("Confidence") == [0.97], str(r[0]))
            check("viewer reads per-region topSpecies/status",
                  r[0].get("TopSpecies") == "Great Cormorant"
                  and r[0].get("Status") == "done")
            check("viewer reads manual provenance (absent -> false)",
                  not r[0].get("Manual"))
        check("viewer reads multiBird + birdCount",
              raw.get("XMP-bird:MultiBird") is True
              and raw.get("XMP-bird:BirdCount") == 2,
              str(raw.get("XMP-bird:MultiBird")))

        print("\n== V3: VIEWER multi-flag write -> pipeline reads the flag ==")
        # `multi_flag_args` set=true on an existing sidecar (create form on a
        # fresh photo, like the viewer's first toggle).
        photo3 = tmp / "B3.jpg"
        write_tiny_jpeg(photo3)
        sidecar3 = tmp / "B3.xmp"
        subprocess.run(
            ["exiftool", "-config", str(CONFIG), "-m", "-XMP-bird:multiBird=true",
             "-o", str(sidecar3), str(photo3)],
            check=True, capture_output=True,
        )
        check("flag sidecar created", sidecar3.exists())
        check("pipeline sees the multiBird request",
              birdid.sidecar_multi(sidecar3) is True)
        # Un-flag: the viewer's `multi_flag_args` set=false clears the flag AND
        # the per-region records + mwg-rs mirror.
        subprocess.run(
            viewer_write_args(sidecar3, [
                "-XMP-bird:multiBird=",
                "-XMP-bird:birdCount=",
                "-XMP-bird:regions=",
                "-XMP-mwg-rs:All=",
            ]),
            check=True, capture_output=True,
        )
        check("pipeline no longer sees the flag after un-flag",
              birdid.sidecar_multi(sidecar3) is False)

        print("\n== V4: the docs config actually registers the regions struct ==")
        # A guard for the stale-config class of bug: docs/birdid.config must
        # contain the struct definition the viewer bundles, or a viewer write of
        # regions degrades into dotted flat tags the pipeline cannot read back.
        cfg_text = CONFIG.read_text()
        check("config declares the regions struct with x/y/w/h/manual",
              "regions" in cfg_text and "manual" in cfg_text
              and "topSpecies" in cfg_text and "List => 'Seq'" in cfg_text,
              "(missing region struct = stale docs config)")

    print("\n============================================")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
