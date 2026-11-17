#!/usr/bin/env python3
"""
Bird Photo Identification Assistant  (birdid.py)

Point it at a folder of photos and it will, for every image it hasn't already
tagged:

  1. read EXIF GPS + timestamp  (exiftool)
  2. map lat/lon -> region (DE / SG / TH) and pick the month
  3. build a candidate shortlist from the offline eBird region x month tables
  4. send the photo + shortlist to a local vision LLM (FastFlowLM on the NPU)
  5. parse the JSON verdict and write an IMatch-safe XMP sidecar per file
  6. keep a resumable manifest + CSV report, and show a live TUI

Originals (CR2/CR3/JPEG/...) are NEVER modified - only `.xmp` sidecars and the
manifest are written. RAW + JPEG of the same capture are processed ONCE (on the
viewable file) and the result is mirrored into both sidecars.

Requirements (all present on the T14):
  - FastFlowLM server with a VISION model:  `flm serve qwen3vl-it:4b`
  - exiftool, python3 (PIL, lxml, requests)

Usage:
  birdid.py /path/to/photos [--region DE] [--model qwen3vl-it:4b] [options]

See --help for all options. Examples:
  birdid.py ~/Pictures/birds --dry-run              # just scan + plan
  birdid.py ~/Pictures/birds                        # process everything
  birdid.py ~/Pictures/birds --max 10 --no-tui      # first 10, plain output
"""

import argparse
import base64
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Config / defaults
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://127.0.0.1:52625/v1"
DEFAULT_MODEL = "qwen3vl-it:4b"
DEFAULT_DATA_DIR = "/data/offline-data/bird/ebird"
DEFAULT_REGION = "DE"
PIPELINE_VERSION = "1"

# Image extensions the scanner understands. Keep the first two groups as the
# "viewable" set used for vision input; the rest are RAW-only (embedded preview).
VIEWABLE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".tif", ".tiff"}
ALL_EXTS = VIEWABLE_EXTS | RAW_EXTS

# Region bounding boxes (lat, lon). Photos without GPS fall back to --region.
REGION_BBOXES = {
    "DE": (47.0, 55.0, 5.5, 15.0),     # Germany
    "SG": (1.1, 1.5, 103.5, 104.2),    # Singapore
    "TH": (5.5, 20.5, 97.0, 106.0),    # Thailand
}

BIRD_NS = "http://ns.birdpipeline/1.0/"

# --------------------------------------------------------------------------
# eBird region x month data
# --------------------------------------------------------------------------

class SpeciesDB:
    """Loads region/*.csv tables: region,species,status,month_mask (12 chars, Jan..Dec)."""

    def __init__(self, regions_dir: Path):
        self.regions_dir = Path(regions_dir)
        # region -> (month index 0..11) -> list[(species, status)]
        self._table: dict[str, dict[int, list[tuple[str, str]]]] = {}

    def load(self):
        if not self.regions_dir.is_dir():
            raise FileNotFoundError(
                f"regions dir not found: {self.regions_dir}\n"
                f"expected region CSVs (DE.csv, SG.csv, TH.csv).")
        for csv_path in sorted(self.regions_dir.glob("*.csv")):
            region = csv_path.stem.upper()
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not row:
                        continue
                    species = (row.get("species") or "").strip()
                    if not species or species.startswith("#"):
                        continue
                    status = (row.get("status") or "resident").strip()
                    mask = (row.get("month_mask") or "").strip()
                    if len(mask) != 12:
                        continue
                    for i, ch in enumerate(mask):
                        if ch == "1":
                            self._table.setdefault(region, {}).setdefault(i, []).append((species, status))
        if not self._table:
            raise FileNotFoundError(f"no region tables loaded from {self.regions_dir}")

    def candidates(self, region: str, month: int) -> list[tuple[str, str]]:
        """Species plausibly present in `region` during `month` (1..12)."""
        return self._table.get(region.upper(), {}).get(month - 1, [])

    def regions(self):
        return sorted(self._table)


def load_taxonomy(path: Path) -> dict[str, str]:
    """Common-name -> scientific-name map from the eBird taxonomy CSV (optional)."""
    tax: dict[str, str] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row and row.get("COMMON_NAME") and row.get("SCIENTIFIC_NAME"):
                    tax[row["COMMON_NAME"].strip()] = row["SCIENTIFIC_NAME"].strip()
    except OSError:
        pass
    return tax


def normalize_species(name: str, tax: dict[str, str]):
    """Return (common_name, scientific_name), tolerating case or a scientific name."""
    name = (name or "").strip()
    if not name or name.lower() in ("unknown", "none", "n/a"):
        return "unknown", ""
    if name in tax:
        return name, tax[name]
    lower = {k.lower(): (k, v) for k, v in tax.items()}
    if name.lower() in lower:
        return lower[name.lower()]
    # maybe the model answered with the scientific name instead
    rev = {v.lower(): k for k, v in tax.items()}
    if name.lower() in rev:
        common = rev[name.lower()]
        return common, tax[common]
    # tolerate the model dropping a leading qualifier, e.g. "Blue Tit" -> the
    # canonical "Eurasian Blue Tit", "Blackcap" -> "Eurasian Blackcap".
    if len(name) >= 5:
        nl = name.lower()
        for common, sci in tax.items():
            cl = common.lower()
            if cl != nl and cl.endswith(nl):
                return common, sci
    return name, ""

# --------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------

def read_exif(path: Path) -> dict:
    """lat/lon (signed decimals) + capture datetime via exiftool."""
    cmd = ["exiftool", "-json", "-n",
           "-GPSLatitude", "-GPSLongitude",
           "-GPSLatitudeRef", "-GPSLongitudeRef",
           "-DateTimeOriginal", "-CreateDate",
           str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"exiftool failed: {e}") from e
    if not out.strip():
        raise RuntimeError("exiftool returned no output")
    d = json.loads(out)[0]

    lat = _gps(d, "GPSLatitude", "GPSLatitudeRef")
    lon = _gps(d, "GPSLongitude", "GPSLongitudeRef")
    dt = None
    for tag in ("DateTimeOriginal", "CreateDate"):
        raw = d.get(tag)
        if raw and raw not in ("0000:00:00 00:00:00", "-", ""):
            try:
                dt = datetime.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S")
                break
            except ValueError:
                continue
    return {"lat": lat, "lon": lon, "datetime": dt}


def _gps(d: dict, val_tag: str, ref_tag: str):
    try:
        v = d.get(val_tag)
        if v is None or v in ("-", ""):
            return None
        val = float(v)
    except (TypeError, ValueError):
        return None
    ref = str(d.get(ref_tag, ""))
    if ref in ("S", "W", "South", "West"):
        val = -abs(val)
    return val


def region_for(lat, lon, default: str) -> str:
    if lat is not None and lon is not None:
        for code, (s, n, w, e) in REGION_BBOXES.items():
            if s <= lat <= n and w <= lon <= e:
                return code
    return default

# --------------------------------------------------------------------------
# Vision input preparation
# --------------------------------------------------------------------------

def prepare_vision_input(path: Path, max_side: int = 1280) -> str:
    """Return base64 JPEG for the model. RAW files: use the embedded preview."""
    from PIL import Image  # imported here so the module can at least be parsed without PIL

    ext = path.suffix.lower()
    data = None
    if ext in RAW_EXTS:
        data = _embedded_preview(path)
    if data is None:
        with open(path, "rb") as f:
            data = f.read()
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _embedded_preview(path: Path):
    """Try to pull a JPEG preview out of a RAW file with exiftool."""
    for tag in ("-PreviewImage", "-JpgFromRaw"):
        try:
            out = subprocess.run(["exiftool", "-b", tag, str(path)],
                                 capture_output=True, timeout=60).stdout
            if out and out[:3] == b"\xff\xd8\xff":  # JPEG magic
                return out
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return None

# --------------------------------------------------------------------------
# FastFlowLM client
# --------------------------------------------------------------------------

def server_ready(base_url: str, timeout: float = 3.0):
    """Return (ok, list_of_model_ids). Connectivity check only."""
    try:
        import requests
        r = requests.get(base_url.rstrip("/") + "/models", timeout=timeout)
        if not r.ok:
            return False, []
        data = r.json().get("data", [])
        return True, [m.get("id") for m in data]
    except Exception:
        return False, []


def ask_vlm(base_url: str, served_model: str, prompt: str, b64: str, timeout: float = 300.0):
    import requests
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": served_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0,
        "max_tokens": 800,
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return extract_json(content)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates fences/extra text)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(text[start:])
    return obj

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

def build_prompt(region: str, dt: datetime, candidates: list[tuple[str, str]],
                 tax: dict[str, str] | None = None) -> str:
    region_name = {"DE": "Germany", "SG": "Singapore", "TH": "Thailand"}.get(region, region)
    date_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown"
    tax = tax or {}
    if candidates:
        parts = []
        for s, st in candidates:
            sci = tax.get(s, "")
            parts.append(f"{s} ({sci}) ({st})" if sci else f"{s} ({st})")
        cand = ", ".join(parts)
    else:
        cand = "none in the local table for this region/month - answer from general knowledge"
    return (
        "You are an expert bird identification assistant. Identify the single main bird in the photo.\n\n"
        f"Photo context:\n"
        f"- Region: {region_name} ({region})\n"
        f"- Date/time: {date_str}\n"
        f"- Species plausibly present here this month (eBird): {cand}\n\n"
        "Rules:\n"
        "1. The species must be a real English common name (matching one from the list when possible).\n"
        "2. If you cannot identify the bird confidently, set confidence < 0.3 and species to \"unknown\".\n"
        "3. Reply with ONLY valid JSON, no markdown, in exactly this shape:\n"
        '{"species": "Common Name", "confidence": 0.0, '
        '"alternatives": ["Other Name", "Another"], "reasoning": "one short sentence"}'
    )

# --------------------------------------------------------------------------
# XMP sidecars (IMatch-safe, merge-aware)
# --------------------------------------------------------------------------

_XMP_TPL = """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
          xmlns:dc="http://purl.org/dc/elements/1.1/"
          xmlns:bird="{bird_ns}">
  <rdf:Description rdf:about="">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def _xmp_tree(sidecar: Path):
    from lxml import etree
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    if sidecar.exists():
        try:
            return etree.parse(str(sidecar), parser).getroot()
        except etree.XMLSyntaxError:
            pass
    return etree.fromstring(_XMP_TPL.format(bird_ns=BIRD_NS).encode(), parser)


def _nsmap():
    from lxml import etree
    return {
        "x": "adobe:ns:meta/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dc": "http://purl.org/dc/elements/1.1/",
        "bird": BIRD_NS,
    }


def _find(root, xpath):
    return root.xpath(xpath, namespaces=_nsmap())


def _set_text(parent, tag, ns_uri, value):
    from lxml import etree
    el = parent.find(f"{{{ns_uri}}}{tag}")
    if el is None:
        el = etree.SubElement(parent, f"{{{ns_uri}}}{tag}")
    el.text = str(value)
    return el


def _set_list(parent, tag, ns_uri, values, container):
    """Set a repeated-field (bird:species/confidence) as an rdf:Seq of rdf:li."""
    from lxml import etree
    rdf = _nsmap()["rdf"]
    for old in parent.findall(f"{{{ns_uri}}}{tag}"):
        parent.remove(old)
    el = etree.SubElement(parent, f"{{{ns_uri}}}{tag}")
    seq = etree.SubElement(el, f"{{{rdf}}}{container}")
    for v in values:
        li = etree.SubElement(seq, f"{{{rdf}}}li")
        li.text = str(v)


def sidecar_done(sidecar: Path) -> bool:
    """A file is 'done' iff its sidecar has bird:species with >= 1 entry."""
    if not sidecar.exists():
        return False
    try:
        root = _xmp_tree(sidecar)
        li = _find(root, "//bird:species/rdf:Seq/rdf:li")
        return bool(li and li[0].text and li[0].text.strip())
    except Exception:
        return False


def write_sidecar(image: Path, result: dict, region: str, month: int,
                  dt: datetime, status: str, cfg, error: str = None, scientific: str = ""):
    """Create/merge an XMP sidecar for `image`. Originals are never touched."""
    from lxml import etree
    sidecar = image.parent / (image.name + ".xmp")
    root = _xmp_tree(sidecar)
    ns = _nsmap()
    descs = _find(root, "//rdf:Description")
    if not descs:
        raise RuntimeError("could not locate rdf:Description in sidecar XML")
    desc = descs[0]

    if error:
        _set_text(desc, "status", BIRD_NS, "error")
        _set_text(desc, "error", BIRD_NS, error[:500])
        _set_text(desc, "pipelineVersion", BIRD_NS, PIPELINE_VERSION)
    else:
        species = result.get("species", "unknown")
        confidence = float(result.get("confidence", 0.0))
        alts = [a for a in result.get("alternatives", []) if a]
        ranked = [species] + [a for a in alts if a and a != species]

        # keep only plausible entries in alternatives for the sidecar.
        # alternatives carry an approximate decaying confidence (the model only
        # gives a confidence for the top pick).
        confs = [confidence] + [
            max(0.05, round(confidence - 0.1 * (i + 1), 2)) for i in range(len(ranked) - 1)
        ]

        _set_list(desc, "species", BIRD_NS, ranked, "Seq")
        _set_list(desc, "confidence", BIRD_NS, [f"{c:.2f}" for c in confs], "Seq")
        _set_text(desc, "summary", BIRD_NS, " | ".join(
            f"{s} {c*100:.0f}%" for s, c in zip(ranked, confs)))
        _set_text(desc, "topSpecies", BIRD_NS, species)
        if scientific:
            _set_text(desc, "scientificName", BIRD_NS, scientific)
        _set_text(desc, "topConfidence", BIRD_NS, f"{confidence:.2f}")
        _set_text(desc, "status", BIRD_NS, status)
        _set_text(desc, "pipelineVersion", BIRD_NS, PIPELINE_VERSION)
        _set_text(desc, "region", BIRD_NS, region)
        _set_text(desc, "month", BIRD_NS, f"{month:02d}")
        if dt:
            _set_text(desc, "datetime", BIRD_NS, dt.strftime("%Y-%m-%d %H:%M:%S"))
        reasoning = result.get("reasoning", "")
        _set_text(desc, "reasoning", BIRD_NS, reasoning[:1000])

        # dc:description (human readable) and dc:subject keywords
        note = f"{species} ({scientific}) {confidence*100:.0f}%" if scientific else f"{species} ({confidence*100:.0f}%)"
        if alts:
            note += "; alternatives: " + ", ".join(alts[:3])
        if reasoning:
            note += f" - {reasoning}"
        _set_text(desc, "description", _nsmap()["dc"], note)

        # dc:subject -> Birds/<species> keywords (convenience, IMatch-managed)
        for old in _find(root, "//dc:subject"):
            old.getparent().remove(old)
        subj = etree.SubElement(desc, f"{{{_nsmap()['dc']}}}subject")
        bag = etree.SubElement(subj, f"{{{_nsmap()['rdf']}}}Bag")
        seen = set()
        for s in ranked[:5]:
            if s not in seen:
                seen.add(s)
                li = etree.SubElement(bag, f"{{{_nsmap()['rdf']}}}li")
                li.text = f"Birds/{s}"

    out = io.BytesIO()
    etree.ElementTree(root).write(out, xml_declaration=True, encoding="UTF-8",
                                  pretty_print=True)
    sidecar.write_bytes(out.getvalue())


# --------------------------------------------------------------------------
# Manifest (resume-safe)
# --------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_manifest(path: Path, manifest: dict):
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------
# Scan & grouping (RAW+JPEG pairs)
# --------------------------------------------------------------------------

class Capture:
    """One photographic capture = one or more files sharing a base name."""

    def __init__(self, base: str):
        self.base = base
        self.files: list[Path] = []

    def primary(self) -> Path:
        """Best file to run the vision model on: a viewable one, else the first."""
        for f in self.files:
            if f.suffix.lower() in VIEWABLE_EXTS:
                return f
        return self.files[0]

    def viewable(self) -> bool:
        return any(f.suffix.lower() in VIEWABLE_EXTS for f in self.files)


def scan_folder(root: Path, manifest: dict) -> list[Capture]:
    captures: dict[str, Capture] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in ALL_EXTS:
            key = p.stem.lower()
            captures.setdefault(key, Capture(key)).files.append(p)

    result = []
    for cap in captures.values():
        cap.files.sort(key=lambda f: (f.suffix.lower() not in VIEWABLE_EXTS, str(f)))
        result.append(cap)
    return result


def sidecar_status(sidecar: Path):
    """Return 'done' / 'error' / None by parsing the sidecar (not string search)."""
    if not sidecar.exists():
        return None
    try:
        root = _xmp_tree(sidecar)
        st = _find(root, "//bird:status")
        status = st[0].text.strip() if st and st[0].text else None
        if status == "error":
            return "error"
        if sidecar_done(sidecar):
            return "done"
    except Exception:
        pass
    return None


def classify_capture(cap: Capture) -> str:
    """done / error / pending - group is done iff every member sidecar is done."""
    statuses = [sidecar_status(f.parent / (f.name + ".xmp")) for f in cap.files]
    if statuses and all(s == "done" for s in statuses):
        return "done"
    if any(s == "error" for s in statuses):
        return "error"
    return "pending"

# --------------------------------------------------------------------------
# TUI (curses) + worker orchestration
# --------------------------------------------------------------------------

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "starting"      # scanning | ready | running | done | error
        self.stage = ""
        self.message = ""
        self.current = ""
        self.last = ""
        self.scanned = 0
        self.pending = 0
        self.done = 0
        self.error = 0
        self.total = 0
        self.processed = 0
        self.start = time.time()
        self.elapsed = 0.0
        self.eta = ""


class TUI:
    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdout.isatty()
        self.state = State()
        self.stop_flag = threading.Event()
        self._win = None

    def start(self):
        if not self.enabled:
            return
        import curses
        self._curses = curses
        self._win = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self._win.keypad(True)
        try:
            curses.curs_set(0)
        except Exception:
            pass

    def stop(self):
        if self._win is not None:
            self._curses.nocbreak()
            self._win.keypad(False)
            self._curses.echo()
            self._curses.endwin()
            self._win = None

    def refresh(self):
        st = self.state
        with st.lock:
            st.elapsed = time.time() - st.start
            if st.processed and st.total:
                per = st.elapsed / st.processed
                left = per * (st.total - st.processed)
                st.eta = f"ETA {int(left//60):02d}:{int(left%60):02d}"
            if not self.enabled:
                return
        import curses
        win = self._win
        win.erase()
        h, w = win.getmaxyx()
        line = 0
        def put(text, attr=0):
            nonlocal line
            if line < h - 1:
                win.addnstr(line, 0, text[:max(0, w - 1)], w - 1, attr)
                line += 1
        put("Bird Photo ID  (q: quit)", curses.A_BOLD)
        put("")
        put(f"phase: {st.phase:<9} stage: {st.stage:<12} message: {st.message}")
        put("")
        put(f"scanned {st.scanned}   pending {st.pending}   done {st.done}   error {st.error}")
        put(f"progress: {st.processed}/{st.total}    elapsed {int(st.elapsed//60):02d}:{int(st.elapsed%60):02d}   {st.eta}")
        if st.total:
            frac = min(1.0, st.processed / st.total)
            width = max(10, w - 20)
            filled = int(width * frac)
            put(f"[{'#' * filled}{'.' * (width - filled)}] {frac*100:4.0f}%")
        put("")
        put(f"current: {st.current}")
        put(f"last   : {st.last[: max(0, w - 10)]}")
        put("")
        put(stage_hint(st.stage))
        win.refresh()

    def set(self, **kw):
        st = self.state
        with st.lock:
            for k, v in kw.items():
                setattr(st, k, v)

    def run_loop(self):
        if not self.enabled:
            return
        import curses
        self._win.timeout(150)
        while not self.stop_flag.is_set():
            key = self._win.getch()
            if key in (ord("q"), ord("Q")):
                self.stop_flag.set()
                break
            self.refresh()
        # NB: do NOT call self.stop() here - main() tears down curses in its
        # finally block (which also joins this thread first).

    def plain_line(self, text):
        if not self.enabled:
            print(text, flush=True)


def stage_hint(stage: str) -> str:
    hints = {
        "exif": "reading EXIF (GPS + timestamp)",
        "region": "mapping location to region / month",
        "candidates": "building eBird candidate shortlist",
        "vision": "sending photo to NPU vision model...",
        "parse": "parsing model JSON verdict",
        "sidecar": "writing XMP sidecar(s)",
        "": "",
    }
    return hints.get(stage, "")


def fmt_result(result: dict, scientific: str = "") -> str:
    if not result:
        return "no result"
    c = float(result.get("confidence", 0.0))
    name = result.get("species", "?")
    if scientific:
        return f"{name} ({scientific}) {c*100:.0f}%"
    return f"{name} ({c*100:.0f}%)"

# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Offline bird-photo identification via local NPU vision model.")
    ap.add_argument("folder", nargs="?", default=None, help="folder (recursive) of photos to scan")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"vision model tag (default {DEFAULT_MODEL})")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FLM OpenAI-compatible base URL")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="dir containing eBird data (regions/, taxonomy)")
    ap.add_argument("--region", default=DEFAULT_REGION, help="default region for photos without GPS (DE/SG/TH)")
    ap.add_argument("--manifest", default=None, help="manifest path (default <folder>/birdid_manifest.json)")
    ap.add_argument("--report", default=None, help="CSV report path (default <folder>/birdid_report.csv)")
    ap.add_argument("--max", type=int, default=0, help="stop after this many captures (0 = all)")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="below this, status is 'low-confidence' (default 0.5)")
    ap.add_argument("--max-side", type=int, default=1280, help="max image dimension sent to the model")
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (s)")
    ap.add_argument("--retry", type=int, default=2, help="retries per failed file")
    ap.add_argument("--dry-run", action="store_true", help="scan + plan only, no NPU calls")
    ap.add_argument("--no-tui", action="store_true", help="plain log output, no fullscreen UI")
    ap.add_argument("--list-candidates", action="store_true",
                    help="print candidate counts per region/month and exit")
    args = ap.parse_args()

    if not args.folder and not args.list_candidates:
        ap.error("the following arguments are required: folder  (or use --list-candidates)")

    root = None
    if args.folder:
        root = Path(args.folder).expanduser().resolve()
        if not root.is_dir():
            sys.exit(f"folder not found: {root}")

    data_dir = Path(args.data_dir)
    regions_dir = data_dir / "regions"
    taxonomy_path = data_dir / "eBird_Taxonomy_v2023.csv"

    db = SpeciesDB(regions_dir)
    db.load()
    taxonomy = load_taxonomy(taxonomy_path)
    print(f"eBird tables: {', '.join(db.regions())}   taxonomy names: {len(taxonomy)}")

    if args.list_candidates:
        for reg in db.regions():
            for m in range(1, 13):
                n = len(db.candidates(reg, m))
                print(f"{reg} {datetime(2020, m, 1):%b}: {n} candidates")
        return

    manifest_path = Path(args.manifest) if args.manifest else root / "birdid_manifest.json"
    report_path = Path(args.report) if args.report else root / "birdid_report.csv"
    manifest = load_manifest(manifest_path)

    tui = TUI(enabled=not args.no_tui)
    tui.start()
    tui_thread = threading.Thread(target=tui.run_loop, daemon=True)
    tui_thread.start()
    summary_lines: list[str] = []
    fatal = False
    try:
        # --- Phase 1: scan & classify ---
        tui.set(phase="scanning", stage="", message="walking folder tree")
        captures = scan_folder(root, manifest)
        tui.set(scanned=len(captures))

        pending = []
        done_n = 0
        error_n = 0
        for cap in captures:
            status = classify_capture(cap)
            if status == "pending":
                pending.append(cap)
            elif status == "done":
                done_n += 1
            else:
                error_n += 1

        tui.set(phase="ready", pending=len(pending), done=done_n, error=error_n, total=len(pending),
                message=f"{len(pending)} to process")

        if args.dry_run:
            summary_lines.append(
                f"scan: {len(captures)} captures | {done_n} done | {error_n} error | {len(pending)} pending")
            for cap in pending[:20]:
                summary_lines.append(f"  [pending] {cap.files[0]}")
            if len(pending) > 20:
                summary_lines.append(f"  ... and {len(pending) - 20} more")
            summary_lines.append("dry-run: nothing processed.")
        elif not pending:
            summary_lines.append(
                f"scan: {len(captures)} captures | {done_n} done | {error_n} error | 0 pending - nothing to do.")

        if not summary_lines:
            # --- Phase 2: verify the NPU server ---
            ok, available = server_ready(args.base_url)
            if not ok:
                raise ConnectionError(
                    f"no FLM server at {args.base_url}.\n"
                    f"Start it first (in another terminal):  flm serve {args.model}")
            if args.model not in available:
                tui.plain_line(f"note: '{args.model}' is not listed by the server "
                               f"(server may serve a different model) - continuing anyway.")

            report_rows = []
            stop = tui.stop_flag

            # --- Phase 3: process ---
            tui.set(phase="running")
            for i, cap in enumerate(pending, 1):
                if stop.is_set() or (args.max and i > args.max):
                    break
                cap_files = cap.files
                primary = cap.primary()
                rel = primary.relative_to(root)
                tui.set(current=str(rel), processed=i - 1)
                last = ""
                try:
                    # EXIF
                    tui.set(stage="exif", message=f"EXIF {primary.name}")
                    meta = read_exif(primary)
                    tui.set(stage="region", message="region/month")
                    region = region_for(meta["lat"], meta["lon"], args.region)
                    month = meta["datetime"].month if meta["datetime"] else datetime.now().month
                    tui.set(stage="candidates", message=f"{region} month {month:02d}")
                    candidates = db.candidates(region, month)
                    if not candidates:
                        # still run with a hint in the prompt
                        pass

                    tui.set(stage="vision", message=f"sending {primary.name}")
                    b64 = prepare_vision_input(primary, args.max_side)
                    prompt = build_prompt(region, meta["datetime"], candidates, taxonomy)
                    result = None
                    last_err = None
                    for attempt in range(args.retry + 1):
                        try:
                            result = ask_vlm(args.base_url, args.model, prompt, b64, args.timeout)
                            break
                        except Exception as e:  # noqa: BLE001
                            last_err = e
                            tui.plain_line(f"  retry {attempt + 1}/{args.retry + 1} for {rel}: {e}")
                            if attempt < args.retry:
                                time.sleep(2)
                    if result is None:
                        raise RuntimeError(f"vision request failed: {last_err}")

                    tui.set(stage="parse", message="parsing verdict")
                    if not isinstance(result, dict) or not result.get("species"):
                        raise RuntimeError(f"bad model output: {result!r}")
                    species, scientific = normalize_species(str(result["species"]), taxonomy)
                    result["species"] = species
                    if scientific:
                        result["scientific"] = scientific
                    confidence = float(result.get("confidence", 0.0))
                    if species not in ("unknown", "") and scientific == "" and taxonomy:
                        tui.plain_line(f"  note: '{species}' not in eBird taxonomy - check spelling")

                    status = "done" if confidence >= args.min_confidence else "low-confidence"
                    tui.set(stage="sidecar", message=f"writing {len(cap_files)} sidecar(s)")
                    for f in cap_files:
                        write_sidecar(f, result, region, month, meta["datetime"], status, args,
                                      scientific=scientific)
                        manifest[str(f)] = {
                            "group": cap.base, "region": region, "month": month,
                            "species": species, "scientific": scientific, "confidence": confidence,
                            "status": status, "pipelineVersion": PIPELINE_VERSION,
                            "mtime": f.stat().st_mtime, "size": f.stat().st_size,
                            "ts": datetime.now().isoformat(timespec="seconds"),
                        }
                    save_manifest(manifest_path, manifest)   # resumable: save after every capture

                    report_rows.append({
                        "capture": cap.base, "file": str(primary), "viewable": primary.name,
                        "region": region, "month": f"{month:02d}", "species": species,
                        "scientific_name": scientific, "confidence": f"{confidence:.2f}",
                        "status": status,
                        "alternatives": "; ".join(result.get("alternatives", [])),
                        "reasoning": result.get("reasoning", ""),
                        "datetime": meta["datetime"].strftime("%Y-%m-%d %H:%M") if meta["datetime"] else "",
                    })
                    last = fmt_result(result, scientific)
                    done_n += 1
                except Exception as e:  # noqa: BLE001
                    status = "error"
                    error_n += 1
                    last = f"ERROR: {e}"
                    try:
                        for f in cap_files:
                            write_sidecar(f, {}, region if "region" in dir() else args.region,
                                          month if "month" in dir() else 1, None, "error", args, error=str(e))
                            manifest[str(f)] = {"group": cap.base, "status": "error",
                                                "error": str(e), "pipelineVersion": PIPELINE_VERSION,
                                                "ts": datetime.now().isoformat(timespec="seconds")}
                        save_manifest(manifest_path, manifest)
                        report_rows.append({"capture": cap.base, "file": str(primary), "viewable": primary.name,
                                            "region": "", "month": "", "species": "", "scientific_name": "",
                                            "confidence": "", "status": "error", "alternatives": "",
                                            "reasoning": str(e)[:500], "datetime": ""})
                    except Exception as e2:  # noqa: BLE001
                        last = f"ERROR (write failed too): {e2}"
                        tui.plain_line(f"  {last}")

                tui.set(processed=i, done=done_n, error=error_n, pending=len(pending) - i,
                        current="", stage="", message="", last=last)
                tui.plain_line(f"[{i}/{len(pending)}] {rel}: {last}")

            # --- report ---
            if report_rows:
                with open(report_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
                    w.writeheader()
                    w.writerows(report_rows)
            tui.set(phase="done", message="finished")
            summary_lines.append(f"\ndone: {done_n} tagged, {error_n} errors -> manifest {manifest_path}")
            if report_rows:
                summary_lines.append(f"report: {report_path}")

        else:
            # early exit (dry-run / nothing to do) - summary_lines already filled
            tui.set(phase="done", message="nothing to process")
    except KeyboardInterrupt:
        tui.set(phase="interrupted", message="interrupted - manifest is resume-safe")
        summary_lines = ["\ninterrupted. rerun the same command to resume."]
    except Exception as e:  # noqa: BLE001
        tui.set(phase="error", message=str(e))
        summary_lines = [f"\nFATAL: {e}"]
        fatal = True
    finally:
        tui.stop_flag.set()
        if tui_thread and tui_thread.is_alive():
            tui_thread.join(timeout=1)
        tui.stop()
    for line in summary_lines:
        print(line, file=sys.stderr if fatal else sys.stdout)
    if fatal:
        sys.exit(1)


if __name__ == "__main__":
    main()
