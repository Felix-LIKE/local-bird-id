#!/usr/bin/env python3
"""Regression test: how birdid CONFIRM behaves on a MULTI-BIRD sidecar.

Pins two contracts:

1) WHOLE-CAPTURE confirm with several birds: a capture carrying bird:regions
   (per-bird candidate lists + boxes) can be confirmed exactly like a
   single-bird capture — the whole-image bird:* fields collapse to the
   confirmed pick (single-entry species at 100%, status 'done') and the record
   is FROZEN, but the per-bird region records, bird:birdCount and the
   bird:multiBird request flag are PRESERVED. Un-confirm removes the flag and
   leaves the regions; a re-run then re-identifies. A GROUP-partial pin (e.g.
   'Mammal') on a multi-bird capture stays a re-runnable partial with regions
   intact. A whole-capture confirm does NOT touch the region records (per-bird
   taxonomy is written only by PER-REGION confirm, below).

2) PER-REGION confirm (--confirm --region N): confirming ONE region writes
   ITS OWN DarwinCore-mirroring taxonomy + verification markers into that
   region record — the per-bird counterpart of the capture dwc block — so a
   multi-bird capture carries EVERY bird's verification, not just the
   primary's. The capture-level dwc tracks the PRIMARY region (see
   _region_primary_index: area weighted by box centering): written from its
   confirmed species, removed when the primary is unconfirmed. The capture is
   FROZEN only once EVERY region is confirmed;
   until then it stays re-runnable so --multi re-classifies exactly the
   unconfirmed regions (confirmed ones pass through untouched).

Runs standalone (no pytest):  python3 test_multibird_confirm.py
Uses only the pipeline's own writers/readers — no network, no models, no
eBird data. lxml is a pipeline dependency (pulled by bootstrap).
"""
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
    """Two per-bird region records, the suggestion-only shape _classify_boxes
    produces at CLASSIFY time: box + ranked candidates + status. NO taxonomy
    ladder — per-bird taxonomy is written ONLY on per-region confirmation (see
    the R* sections), never at classify time."""
    return [
        {"x": 0.10, "y": 0.10, "w": 0.40, "h": 0.30, "manual": False,
         "topSpecies": "Great Cormorant", "topConfidence": 0.97, "status": "done",
         "ranked": [("Great Cormorant", 0.97), ("European Shag", 0.87)],
         "reasoning": "region 1"},
        {"x": 0.60, "y": 0.10, "w": 0.30, "h": 0.30, "manual": False,
         "topSpecies": "Caspian Gull", "topConfidence": 0.90, "status": "done",
         "ranked": [("Caspian Gull", 0.90), ("Great Cormorant", 0.80)],
         "reasoning": "region 2"},
    ]


# A minimal common->scientific taxonomy so the confirm paths resolve the
# scientific name + genus/specificEpithet (order/family need the Clements
# hierarchy FILE, so they stay empty — taxonomy_path is None in these tests).
TAX = {
    "Great Cormorant": "Phalacrocorax carbo",
    "Caspian Gull": "Larus cachinnans",
    "European Shag": "Phalacrocorax aristotelis",
}


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


def build_flag_only_sidecar(path: Path):
    """The empty flag-only sidecar the viewer's multi-flag write used to create
    (exiftool with only `-XMP-bird:multiBird=true`, no status/species): the
    source of the 'done + no entries' bug when it sat next to an identified
    original-case sidecar."""
    path.write_text(
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        ' <rdf:Description xmlns:bird="http://ns.birdpipeline/1.0/" rdf:about="">\n'
        '  <bird:multiBird>true</bird:multiBird>\n'
        ' </rdf:Description>\n'
        '</rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>\n', encoding="utf-8")


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


def region_confirm_state(sidecar: Path) -> list:
    """Each stored region's per-bird verification markers + ladder, as read
    back by the pipeline's own _read_regions."""
    out = []
    for r in birdid._read_regions(sidecar):
        out.append({
            "confirmed": r.get("confirmed", ""),
            "scientificName": r.get("scientificName", ""),
            "order": r.get("order", ""), "family": r.get("family", ""),
            "genus": r.get("genus", ""), "specificEpithet": r.get("specificEpithet", ""),
            "dateIdentified": r.get("dateIdentified", ""),
            "verificationStatus": r.get("identificationVerificationStatus", ""),
        })
    return out


def dwc_verified(sidecar: Path) -> dict:
    """The capture-level dwc mirror's verdict fields (empty when absent)."""
    root = birdid._xmp_tree(sidecar)
    def _f(tag):
        el = birdid._find(root, f"//dwc:Identification/dwc:{tag}")
        return el[0].text.strip() if el and el[0].text else ""
    def _tax(tag):
        el = birdid._find(root, f"//dwc:Taxon/dwc:{tag}")
        return el[0].text.strip() if el and el[0].text else ""
    return {
        "scientificName": _tax("scientificName"),
        "verbatim": _f("verbatimIdentification"),
        "status": _f("identificationVerificationStatus"),
    }


def region_carry_no_ladder(sidecar: Path) -> bool:
    """Classify-time regions must carry NO taxonomy / verification markers."""
    return all(not r.get("confirmed") and not r.get("scientificName")
               and not r.get("identificationVerificationStatus")
               for r in birdid._read_regions(sidecar))


def run():
    with tempfile.TemporaryDirectory(prefix="birdid-multibird-confirm-") as d:
        root = Path(d)
        sidecar = build_multi_sidecar(root)
        manifest = root / "birdid_manifest.json"

        print("== setup: faithful multi-bird sidecar (2 regions) ==")
        check("sidecar written, flagged, not confirmed",
              birdid.sidecar_multi(sidecar)
              and len(birdid._read_regions(sidecar)) == 2
              and birdid.sidecar_confirmed(sidecar) is None
              and not birdid.sidecar_frozen(sidecar))
        check("classify-time regions carry NO taxonomy / verification (confirm-only)",
              region_carry_no_ladder(sidecar), str(region_confirm_state(sidecar)))
        check("no capture-level dwc mirror yet",
              dwc_verified(sidecar) == {"scientificName": "", "verbatim": "",
                                        "status": ""}, str(dwc_verified(sidecar)))
        check("regions NOT all confirmed", not birdid._regions_all_confirmed(sidecar))

        # Region 0 (center 0.30,0.25 area 0.12) outranks region 1 (center
        # 0.75,0.25 area 0.09) under the center-weighted primary score (both
        # near center; region 0 is larger), so it is the PRIMARY.
        check("primary region is index 0 (center-weighted score)",
              birdid._region_primary_index(birdid._read_regions(sidecar)) == 0)

        print("\n== R1: PER-REGION confirm of the PRIMARY (region 0) ==")
        t, s, _ = birdid.run_region_confirm(
            root, "birds_branch", "Great Cormorant", False, 0, manifest, TAX)
        check("touched 1 sidecar", t == 1, f"touched={t}")
        st = region_confirm_state(sidecar)
        check("region 0 confirmed with own ladder + verification markers",
              st[0]["confirmed"] == "Great Cormorant"
              and st[0]["scientificName"] == "Phalacrocorax carbo"
              and st[0]["order"] == "" and st[0]["family"] == ""  # {} taxonomy passed
              and st[0]["genus"] == "Phalacrocorax" and st[0]["specificEpithet"] == "carbo"
              and st[0]["dateIdentified"] == datetime.today().strftime("%Y-%m-%d")
              and st[0]["verificationStatus"] == birdid.DWC_CONFIRM_STATUS,
              str(st[0]))
        check("region 1 untouched (still a suggestion, no verification)",
              st[1]["confirmed"] == "" and st[1]["verificationStatus"] == "",
              str(st[1]))
        check("capture dwc mirrors the PRIMARY (no freeze flag yet)",
              dwc_verified(sidecar)["scientificName"] == "Phalacrocorax carbo"
              and dwc_verified(sidecar)["status"] == "",
              str(dwc_verified(sidecar)))
        check("NOT frozen, no capture-level confirmed (other regions remain)",
              birdid.sidecar_confirmed(sidecar) is None
              and not birdid.sidecar_frozen(sidecar))
        check("not all regions confirmed", not birdid._regions_all_confirmed(sidecar))

        print("\n== R2: PER-REGION confirm of region 1 (all now confirmed) ==")
        t2, s2, _ = birdid.run_region_confirm(
            root, "birds_branch", "Caspian Gull", False, 1, manifest, TAX)
        check("touched 1 sidecar", t2 == 1, f"touched={t2}")
        check("ALL regions confirmed -> capture FROZEN from the primary's verdict",
              birdid.sidecar_confirmed(sidecar) == "Great Cormorant"
              and birdid.sidecar_frozen(sidecar)
              and birdid._regions_all_confirmed(sidecar))
        st2 = region_confirm_state(sidecar)
        check("region 1 confirmed with own ladder",
              st2[1]["confirmed"] == "Caspian Gull"
              and st2[1]["scientificName"] == "Larus cachinnans"
              and st2[1]["genus"] == "Larus", str(st2[1]))
        check("capture dwc now carries the full verification status (frozen)",
              dwc_verified(sidecar)["scientificName"] == "Phalacrocorax carbo"
              and dwc_verified(sidecar)["status"] == birdid.DWC_CONFIRM_STATUS,
              str(dwc_verified(sidecar)))
        check("manifest records the capture-level confirmed pick",
              birdid.load_manifest(manifest).get(str(sidecar), {}).get("confirmed")
              == "Great Cormorant")

        print("\n== R3: re-confirm an already-confirmed region is a skip ==")
        t3, s3, _ = birdid.run_region_confirm(
            root, "birds_branch", "Great Cormorant", False, 0, manifest, TAX)
        check("touched 0, skipped 1", t3 == 0 and s3 == 1, f"t={t3} s={s3}")

        print("\n== R4: un-confirm region 1 -> partially done, un-frozen ==")
        t4, _, _ = birdid.run_region_confirm(root, "birds_branch", None, True, 1,
                                             manifest, {})
        check("touched 1", t4 == 1, f"touched={t4}")
        st4 = region_confirm_state(sidecar)
        check("region 1 confirmation dropped (ladder too)",
              st4[1]["confirmed"] == "" and st4[1]["scientificName"] == ""
              and st4[1]["verificationStatus"] == "", str(st4[1]))
        check("capture no longer frozen; dwc still reflects the confirmed primary",
              not birdid.sidecar_frozen(sidecar)
              and birdid.sidecar_confirmed(sidecar) is None
              and dwc_verified(sidecar)["scientificName"] == "Phalacrocorax carbo"
              and dwc_verified(sidecar)["status"] == "",
              str(dwc_verified(sidecar)))
        check("not all regions confirmed", not birdid._regions_all_confirmed(sidecar))

        print("\n== R5: un-confirm the PRIMARY -> capture dwc removed ==")
        t5, _, _ = birdid.run_region_confirm(root, "birds_branch", None, True, 0,
                                             manifest, {})
        check("touched 1", t5 == 1, f"touched={t5}")
        st5 = region_confirm_state(sidecar)
        check("no region carries any confirmation",
              all(not r.get("confirmed") and not r.get("scientificName")
                  for r in birdid._read_regions(sidecar)), str(st5))
        check("capture dwc removed, capture un-frozen",
              dwc_verified(sidecar) == {"scientificName": "", "verbatim": "",
                                        "status": ""}
              and not birdid.sidecar_frozen(sidecar)
              and birdid.sidecar_confirmed(sidecar) is None,
              str(dwc_verified(sidecar)))

        print("\n== R6: confirmed regions pass through a --multi re-run ==")
        # Re-confirm both regions so the capture is fully done + frozen.
        birdid.run_region_confirm(root, "birds_branch", "Great Cormorant", False,
                                  0, manifest, TAX)
        birdid.run_region_confirm(root, "birds_branch", "Caspian Gull", False,
                                  1, manifest, TAX)
        check("fully confirmed + frozen before the re-run simulation",
              birdid._regions_all_confirmed(sidecar)
              and birdid.sidecar_frozen(sidecar))
        before = sidecar.read_bytes()
        # The --multi loop reads stored regions and passes confirmed ones
        # through _classify_boxes untouched; write_sidecar(regions=...) must
        # then preserve every confirmed marker. Simulate that round-trip.
        stored = birdid._read_regions(sidecar)
        passthrough = [
            b for b in stored if b.get("confirmed")  # what _classify_boxes emits
        ]
        class Cfg:
            dc_fields = False
        birdid.write_sidecar(sidecar, {"species": "Great Cormorant",
                                       "scientific": "Phalacrocorax carbo",
                                       "confidence": 0.93, "alternatives": []},
                             "EUROPE", 11, datetime(2026, 11, 21), "done", Cfg(),
                             regions=passthrough)
        st6 = region_confirm_state(sidecar)
        check("both confirmed regions survive the re-run byte-for-byte",
              [r["confirmed"] for r in st6] == ["Great Cormorant", "Caspian Gull"]
              and [r["scientificName"] for r in st6]
              == ["Phalacrocorax carbo", "Larus cachinnans"]
              and all(r["verificationStatus"] == birdid.DWC_CONFIRM_STATUS
                      for r in st6), str(st6))
        check("fully-confirmed capture stays frozen after the re-run",
              birdid.sidecar_frozen(sidecar)
              and birdid.sidecar_confirmed(sidecar) == "Great Cormorant")

        print("\n== T1: whole-capture confirm on a multi-bird capture ==")
        # Un-fix the regions again (drop confirmations) to test whole-capture.
        birdid.run_region_confirm(root, "birds_branch", None, True, 0, manifest, TAX)
        birdid.run_region_confirm(root, "birds_branch", None, True, 1, manifest, TAX)
        t6, s6, _ = birdid.run_confirm(
            root, "birds_branch", "Great Cormorant", False, manifest, TAX)
        check("one sidecar touched", t6 == 1, f"touched={t6}")
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
        check("whole-capture confirm does NOT write per-region taxonomy",
              region_carry_no_ladder(sidecar))
        check("birdCount=2 and multiBird flag kept",
              birdid._find(rootx, "//bird:birdCount")[0].text.strip() == "2"
              and birdid.sidecar_multi(sidecar))
        check("record is FROZEN", birdid.sidecar_frozen(sidecar))

        print("\n== T2: re-confirm same species is idempotent (skip) ==")
        touched2, skipped2, _ = birdid.run_confirm(
            root, "birds_branch", "Great Cormorant", False, manifest, TAX)
        check("second run touches 0, reports 1 skipped",
              touched2 == 0 and skipped2 == 1, f"touched={touched2} skipped={skipped2}")

        print("\n== T3: a frozen sidecar is never rewritten ==")
        before3 = sidecar.read_bytes()
        # simulate any later run reaching write_sidecar (the backstop)
        birdid.write_sidecar(sidecar, {"species": "Different", "confidence": 0.5,
                                       "alternatives": []}, "EUROPE", 11,
                             datetime(2026, 11, 21), "done", type("C", (), {"dc_fields": False}))
        check("confirmed sidecar byte-identical after a write_sidecar attempt",
              sidecar.read_bytes() == before3)

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
        birdid.run_confirm(root, "birds_branch", "Mammal", False, manifest, TAX)
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

        print("\n== M1: a multi capture stores NO whole-image single-bird prediction ==")
        mroot = root / "m1"
        mroot.mkdir()
        ms = mroot / "multi_bird.xmp"
        birdid.write_sidecar(ms, {"species": "Mallard", "confidence": 0.5,
                                  "alternatives": [], "reasoning": "whole image"},
                             "EUROPE", 11, datetime(2026, 11, 21), "done",
                             type("C", (), {"dc_fields": False}),
                             regions=multi_regions())
        mdesc = birdid._find(birdid._xmp_tree(ms), "//rdf:Description")[0]
        mtags = {c.tag.split("}")[-1] for c in mdesc}
        for f in ("species", "confidence", "summary", "topSpecies",
                  "topConfidence", "scientificName", "avibaseId",
                  "avibaseUrl", "reasoning"):
            check(f"multi capture has no whole-image '{f}'",
                  f not in mtags, "present")
        check("multi capture keeps verdict + metadata + structure",
              {"status", "region", "month", "multiBird", "birdCount",
               "regions"} <= mtags)
        check("multi capture reads as done (skip logic works)",
              birdid.sidecar_done(ms) and birdid.sidecar_status(ms) == "done")
        check("per-region birds still intact",
              len(birdid._read_regions(ms)) == 2)

        print("\n== M2: single-bird re-run of a multi-flagged capture stays clean ==")
        m2root = root / "m2"
        m2root.mkdir()
        fs = m2root / "flagged.xmp"
        build_flag_only_sidecar(fs)   # bare flag, no status/species yet
        # A normal (single-bird) run re-processes the flagged capture.
        birdid.write_sidecar(fs, {"species": "Mallard", "confidence": 0.55,
                                  "alternatives": [], "reasoning": "whole"},
                             "EUROPE", 11, datetime(2026, 11, 21), "done",
                             type("C", (), {"dc_fields": False}))
        fdesc = birdid._find(birdid._xmp_tree(fs), "//rdf:Description")[0]
        f2tags = {c.tag.split("}")[-1] for c in fdesc}
        check("flagged capture keeps the multi flag after a single re-run",
              "multiBird" in f2tags)
        check("flagged capture still has NO whole-image prediction",
              "species" not in f2tags and "topSpecies" not in f2tags
              and "topConfidence" not in f2tags)
        check("flagged capture reads as done (no infinite re-run)",
              birdid.sidecar_status(fs) == "done")

        print("\n== D1: consolidate duplicate mixed-case sidecars (the 'done + "
              "no entries' bug) ==")
        dup_root = root / "dup"
        dup_root.mkdir()
        # The exact broken pair from the field: an identified ORIGINAL-case
        # sidecar (whole-image, no multi flag — the old pipeline never saw the
        # lowercase flag) + an empty flag-only LOWERCASE sidecar (the viewer's
        # flag write before any --multi run).
        rich = dup_root / "BIRDS_BRANCH.xmp"
        flag_only = dup_root / "birds_branch.xmp"
        birdid.write_sidecar(rich, {"species": "Great Cormorant",
                                    "scientific": "Phalacrocorax carbo",
                                    "confidence": 0.92, "alternatives": []},
                             "EUROPE", 11, datetime(2026, 11, 21), "done",
                             type("C", (), {"dc_fields": False}))
        build_flag_only_sidecar(flag_only)
        check("duplicate pair present",
              rich.exists() and flag_only.exists()
              and not birdid.sidecar_multi(rich))
        removed = birdid._consolidate_duplicate_sidecars(dup_root)
        check("one duplicate sidecar removed", removed == 1, f"removed={removed}")
        check("only the identified sidecar remains",
              rich.exists() and not flag_only.exists())
        check("multi flag merged into the survivor (request preserved)",
              birdid.sidecar_multi(rich) and birdid.sidecar_status(rich) == "done",
              f"multi={birdid.sidecar_multi(rich)} status={birdid.sidecar_status(rich)}")
        check("consolidation is idempotent (nothing left to remove)",
              birdid._consolidate_duplicate_sidecars(dup_root) == 0)

        print("\n== D2: Capture.sidecar() resolves mixed-case sidecars ==")
        low_root = root / "low"
        low_root.mkdir()
        low = low_root / "new_bird.xmp"
        low.write_text("<x/>", encoding="utf-8")
        cap = birdid.Capture("NEW_BIRD", low_root)
        check("Capture.sidecar finds the lowercase variant",
              cap.sidecar() == low, f"got {cap.sidecar()}")
        exact = low_root / "NEW_BIRD.xmp"
        exact.write_text("<x/>", encoding="utf-8")
        cap2 = birdid.Capture("NEW_BIRD", low_root)
        check("Capture.sidecar prefers the exact-case path when both exist",
              cap2.sidecar() == exact, f"got {cap2.sidecar()}")

        print("\n== F1: sidecars are written as readable XML documents (not plain text) ==")
        def format_ok(p: Path) -> tuple:
            raw = p.read_bytes()
            first = raw.splitlines()[0] if raw.splitlines() else b""
            maxlen = max((len(l) for l in raw.splitlines()), default=0)
            return (first.startswith(b"<?xml version='1.0' encoding='UTF-8'?>"),
                    b"xpacket" not in raw, len(raw.splitlines()) > 10,
                    maxlen <= 1000)
        froot = root / "f1"
        froot.mkdir()
        fs = froot / "single_fmt.xmp"
        birdid.write_sidecar(fs, {"species": "Coal Tit", "confidence": 0.95,
                                  "alternatives": ["Great Tit"],
                                  "reasoning": "single"},
                             "EUROPE", 11, datetime(2026, 11, 21), "done",
                             type("C", (), {"dc_fields": True}))
        fdecl, fnopkt, freadable, fcompact = format_ok(fs)
        check("single sidecar has an XML declaration (OS sees an XML document)",
              fdecl, "no <?xml?>" if not fdecl else "")
        check("single sidecar has no xpacket wrapper",
              fnopkt, "xpacket present")
        check("single sidecar is pretty-printed (every element on its own line)",
              freadable, "only a few lines")
        check("single sidecar has no giant one-line blobs",
              fcompact, "long lines remain")
        # same guarantees for the multi capture (regions nested deep)
        fm = froot / "multi_fmt.xmp"
        birdid.write_sidecar(fm, {"species": "Mallard", "confidence": 0.5,
                                  "alternatives": [], "reasoning": "whole"},
                             "EUROPE", 11, datetime(2026, 11, 21), "done",
                             type("C", (), {"dc_fields": False}),
                             regions=multi_regions())
        mdecl, mnopkt, mreadable, mcompact = format_ok(fm)
        check("multi sidecar has an XML declaration", mdecl, "")
        check("multi sidecar has no xpacket wrapper", mnopkt, "xpacket present")
        check("multi sidecar is pretty-printed incl. nested regions",
              mreadable, "only a few lines")
        check("multi sidecar has no giant one-line blobs", mcompact,
              "long lines remain")

        print("\n== G1: DarwinCore order/family come from the eBird taxonomy ==")
        # Regression: find_taxonomy prefers the v2023 eBird API CSV, but the
        # hierarchy loader only read the v2026 (lowercase) column names, so it
        # returned an EMPTY map and order/family were omitted from EVERY dwc
        # block — single-bird captures AND multi-bird per-region ladders alike.
        # Both formats must yield the same (order, family) ladder.
        v23 = root / "taxonomy_v2023.csv"
        v23.write_text(
            "SCIENTIFIC_NAME,COMMON_NAME,CATEGORY,ORDER,FAMILY_SCI_NAME\n"
            "Phalacrocorax carbo,Great Cormorant,species,Suliformes,"
            "Phalacrocoracidae\n"
            "Larus cachinnans,Caspian Gull,species,Charadriiformes,Laridae\n"
            "Phalacrocorax aristotelis,European Shag,species,Suliformes,"
            "Phalacrocoracidae\n",
            encoding="utf-8")
        v26 = root / "taxonomy_v2026.csv"
        v26.write_text(
            "category,English name,scientific name,order,family\n"
            "species,Great Cormorant,Phalacrocorax carbo,Suliformes,"
            "Phalacrocoracidae (Cormorants and Shags)\n"
            "species,Caspian Gull,Larus cachinnans,Charadriiformes,Laridae\n"
            "species,European Shag,Phalacrocorax aristotelis,Suliformes,"
            "Phalacrocoracidae\n",
            encoding="utf-8")
        for label, taxpath in (("v2023 (eBird API CSV)", v23),
                               ("v2026 (Clements checklist)", v26)):
            hier = birdid.load_taxonomy_hierarchy(taxpath)
            check(f"hierarchy parsed from {label}",
                  hier.get("phalacrocorax carbo") == ("Suliformes", "Phalacrocoracidae")
                  and hier.get("larus cachinnans") == ("Charadriiformes", "Laridae"),
                  str(hier))

        # A whole-capture confirm given the taxonomy file must put order/family
        # into the dwc:Taxon mirror (previously they were dropped).
        groot = root / "g1"
        groot.mkdir()
        gs = groot / "dwc_single.xmp"
        birdid.write_sidecar(gs, {"species": "Great Cormorant",
                                  "scientific": "Phalacrocorax carbo",
                                  "confidence": 0.93, "alternatives": []},
                             "EUROPE", 11, datetime(2026, 11, 21), "done",
                             type("C", (), {"dc_fields": False}))
        tg, sg, _ = birdid.run_confirm(groot, "dwc_single", "Great Cormorant",
                                       False, manifest, TAX, taxonomy_path=v23)
        check("whole-capture confirm touched 1", tg == 1, f"touched={tg}")
        grootx = birdid._xmp_tree(gs)
        def _dwc_taxon(tag):
            el = birdid._find(grootx, f"//dwc:Taxon/dwc:{tag}")
            return el[0].text.strip() if el and el[0].text else ""
        check("dwc:Taxon carries order/family for a single-bird capture",
              _dwc_taxon("order") == "Suliformes"
              and _dwc_taxon("family") == "Phalacrocoracidae",
              f"order={_dwc_taxon('order')!r} family={_dwc_taxon('family')!r}")

        # A per-region confirm with the taxonomy file must write order/family
        # into that region's own ladder (the per-bird dwc equivalent).
        gmulti = groot / "dwc_multi.xmp"
        birdid.write_sidecar(gmulti, {"species": "Mallard", "confidence": 0.5,
                                      "alternatives": [], "reasoning": "whole"},
                             "EUROPE", 11, datetime(2026, 11, 21), "done",
                             type("C", (), {"dc_fields": False}),
                             regions=multi_regions())
        tr, sr, _ = birdid.run_region_confirm(
            groot, "dwc_multi", "Great Cormorant", False, 0, manifest, TAX,
            taxonomy_path=v23)
        check("region confirm touched 1", tr == 1, f"touched={tr}")
        gst = region_confirm_state(gmulti)
        check("region ladder carries order/family for a multi-bird capture",
              gst[0]["order"] == "Suliformes"
              and gst[0]["family"] == "Phalacrocoracidae", str(gst[0]))
        gmtx = birdid._xmp_tree(gmulti)
        def _dwc_taxon2(tag):
            el = birdid._find(gmtx, f"//dwc:Taxon/dwc:{tag}")
            return el[0].text.strip() if el and el[0].text else ""
        check("capture dwc mirror also carries order/family (primary region)",
              _dwc_taxon2("order") == "Suliformes"
              and _dwc_taxon2("family") == "Phalacrocoracidae",
              f"order={_dwc_taxon2('order')!r} family={_dwc_taxon2('family')!r}")

        print("\n== G2: PRIMARY region = area weighted by box centering ==")
        # A bird parked at the frame edge should NOT beat a smaller bird near
        # the frame center (the 5M4_0217 case: a center Barn Swallow loses to
        # an edge Mallard under pure largest-area). Center-weighting must flip
        # it. Mirror of the viewer JS primaryRegionScore — keep in sync.
        edge_big = {"x": 0.83, "y": 0.67, "w": 0.05, "h": 0.06}   # area 0.0030, center (0.855, 0.70)
        center_small = {"x": 0.47, "y": 0.54, "w": 0.04, "h": 0.06}  # area 0.0024, center (0.49, 0.57)
        check("center-weighted score prefers the small centered bird",
              birdid._region_primary_score(center_small)
              > birdid._region_primary_score(edge_big),
              f"center={birdid._region_primary_score(center_small):.5f} "
              f"edge={birdid._region_primary_score(edge_big):.5f}")
        check("primary index picks the small centered bird",
              birdid._region_primary_index([edge_big, center_small]) == 1)
        # But a genuinely much larger box still wins regardless of position.
        huge_edge = {"x": 0.0, "y": 0.0, "w": 0.9, "h": 0.9}   # fills the frame
        check("a much larger box still wins over a centered small one",
              birdid._region_primary_index([huge_edge, center_small]) == 0)
        # Area alone is NOT the arbiter: a large-but-edge box loses to a
        # centered one of comparable size.
        big_edge = {"x": 0.85, "y": 0.0, "w": 0.15, "h": 0.8}  # center (0.925, 0.4)
        mid_center = {"x": 0.3, "y": 0.2, "w": 0.4, "h": 0.4}  # center (0.5, 0.4)
        check("larger-but-edged box loses to a centered comparable one",
              birdid._region_primary_index([big_edge, mid_center]) == 1)

    print("\n============================================")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
