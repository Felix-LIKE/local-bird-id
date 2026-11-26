#!/usr/bin/env python3
"""Regression test: how birdid CONFIRM behaves on a MULTI-BIRD sidecar.

This pins the user-facing contract "confirm with several birds": a capture
carrying bird:regions (per-bird candidate lists + boxes) can be confirmed
exactly like a single-bird capture — the whole-image bird:* fields collapse
to the confirmed pick (single-entry species at 100%, status 'done') and the
record is FROZEN, but the per-bird region records, bird:birdCount and the
bird:multiBird request flag are PRESERVED. Un-confirm removes the flag and
leaves the regions; a re-run then re-identifies. A GROUP-partial pin (e.g.
'Mammal') on a multi-bird capture stays a re-runnable partial with regions
intact.

Runs standalone (no pytest):  python3 test_multibird_confirm.py
Uses only the pipeline's own writers/readers — no network, no models, no
eBird data. lxml is a pipeline dependency (pulled by bootstrap).
"""
import importlib.util
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import birdid  # noqa: E402  (sys.path must be set first)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS: {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL: {name} {detail}")


def multi_regions():
    """Two per-bird region records (the shape _classify_boxes produces)."""
    return [
        {"x": 0.10, "y": 0.10, "w": 0.40, "h": 0.30, "manual": False,
         "topSpecies": "Great Cormorant", "topConfidence": 0.97, "status": "done",
         "ranked": [("Great Cormorant", 0.97), ("European Shag", 0.87)],
         "reasoning": "region 1"},
        {"x": 0.60, "y": 0.10, "w": 0.30, "h": 0.30, "manual": False,
         "topSpecies": "Great Cormorant", "topConfidence": 0.90, "status": "done",
         "ranked": [("Great Cormorant", 0.90), ("Caspian Gull", 0.80)],
         "reasoning": "region 2"},
    ]


def build_multi_sidecar(root: Path) -> Path:
    """A faithful multi-bird sidecar via the pipeline's own write_sidecar +
    _set_regions (the exact code --multi runs)."""
    class Cfg:
        dc_fields = False
    sidecar = root / "birds_branch.xmp"
    result = {"species": "Great Cormorant", "scientific": "Phalacrocorax carbo",
              "confidence": 0.92,
              "alternatives": ["European Shag", "Caspian Gull"],
              "reasoning": "whole image"}
    birdid.write_sidecar(sidecar, result, "EUROPE", 11, datetime(2026, 11, 21),
                         "done", Cfg(), scientific="Phalacrocorax carbo",
                         avibase_id="3DF5C587",
                         avibase_url="https://avibase.ca/3DF5C587",
                         regions=multi_regions())
    return sidecar


def top_species_list(sidecar: Path) -> list:
    """The WHOLE-IMAGE bird:species list (direct child of the top Description)."""
    root = birdid._xmp_tree(sidecar)
    desc = birdid._find(root, "//rdf:Description")[0]
    ns = birdid._nsmap()["bird"]
    el = desc.find(f"{{{ns}}}species")
    if el is None:
        return []
    return [li.text for li in el.iter(f"{{{birdid._nsmap()['rdf']}}}li")
            if li.text and li.text.strip()]


def run():
    with tempfile.TemporaryDirectory(prefix="birdid-multibird-confirm-") as d:
        root = Path(d)
        sidecar = build_multi_sidecar(root)
        manifest = root / "birdid_manifest.json"

        print("== setup: faithful multi-bird sidecar (2 regions) ==")
        check("sidecar written, flagged, not confirmed",
              birdid.sidecar_multi(sidecar)
              and len(birdid._read_regions(sidecar)) == 2
              and birdid.sidecar_confirmed(sidecar) is None)

        print("\n== T1: run_confirm('<species>') on the multi-bird capture ==")
        touched, skipped, notes = birdid.run_confirm(
            root, "birds_branch", "Great Cormorant", False, manifest, {})
        check("one sidecar touched", touched == 1, f"touched={touched}")
        check("confirmed flag set to the pick",
              birdid.sidecar_confirmed(sidecar) == "Great Cormorant")
        check("whole-image species collapsed to a single confirmed entry",
              top_species_list(sidecar) == ["Great Cormorant"],
              f"got {top_species_list(sidecar)}")
        rootx = birdid._xmp_tree(sidecar)
        check("status -> done",
              birdid._find(rootx, "//bird:status")[0].text.strip() == "done")
        check("REGIONS PRESERVED: still 2 with per-bird species lists",
              len(birdid._read_regions(sidecar)) == 2
              and len(birdid._find(rootx, "//bird:regions/rdf:Seq/rdf:li")) == 2)
        check("birdCount=2 and multiBird flag kept",
              birdid._find(rootx, "//bird:birdCount")[0].text.strip() == "2"
              and birdid.sidecar_multi(sidecar))
        check("record is FROZEN", birdid.sidecar_frozen(sidecar))

        print("\n== T2: re-confirm same species is idempotent (skip) ==")
        touched2, skipped2, _ = birdid.run_confirm(
            root, "birds_branch", "Great Cormorant", False, manifest, {})
        check("second run touches 0, reports 1 skipped",
              touched2 == 0 and skipped2 == 1, f"touched={touched2} skipped={skipped2}")

        print("\n== T3: a frozen sidecar is never rewritten ==")
        before = sidecar.read_bytes()
        # simulate any later run reaching write_sidecar (the backstop)
        birdid.write_sidecar(sidecar, {"species": "Different", "confidence": 0.5,
                                       "alternatives": []}, "EUROPE", 11,
                             datetime(2026, 11, 21), "done", type("C", (), {"dc_fields": False}))
        check("confirmed sidecar byte-identical after a write_sidecar attempt",
              sidecar.read_bytes() == before)

        print("\n== T4: unconfirm leaves the regions, un-freezes ==")
        touched3, _, _ = birdid.run_confirm(root, "birds_branch", None, True,
                                            manifest, {})
        check("unconfirm touched the sidecar", touched3 == 1, f"touched={touched3}")
        check("confirmation removed, not frozen",
              birdid.sidecar_confirmed(sidecar) is None
              and not birdid.sidecar_frozen(sidecar))
        check("regions + flag + count still intact after unconfirm",
              len(birdid._read_regions(sidecar)) == 2
              and birdid.sidecar_multi(sidecar))
        rootu = birdid._xmp_tree(sidecar)
        check("whole-image species left as the confirmed pick (not wiped)",
              top_species_list(sidecar) == ["Great Cormorant"],
              f"got {top_species_list(sidecar)}")

        print("\n== T5: GROUP-partial pin on a multi-bird capture ==")
        birdid.run_confirm(root, "birds_branch", "Mammal", False, manifest, {})
        rootp = birdid._xmp_tree(sidecar)
        check("status -> partial (re-runnable, no freeze)",
              birdid._find(rootp, "//bird:status")[0].text.strip() == "partial"
              and birdid.sidecar_confirmed(sidecar) is None
              and not birdid.sidecar_frozen(sidecar))
        check("regions + flag still intact after a group pin",
              len(birdid._read_regions(sidecar)) == 2 and birdid.sidecar_multi(sidecar))

        # A post-unconfirm --multi re-run must re-identify and keep the boxes:
        # verify the write_sidecar(regions=...) round-trip on the un-frozen file.
        birdid.write_sidecar(sidecar, {"species": "Great Cormorant",
                                       "scientific": "Phalacrocorax carbo",
                                       "confidence": 0.93, "alternatives": []},
                             "EUROPE", 11, datetime(2026, 11, 21), "done",
                             type("C", (), {"dc_fields": False}),
                             regions=multi_regions())
        check("re-identified capture keeps exactly 2 regions",
              len(birdid._read_regions(sidecar)) == 2)

    print("\n============================================")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
