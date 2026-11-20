#!/usr/bin/env python3
"""
Bird Photo Identification Assistant  (birdid.py)

Point it at a folder of photos and it will, for every image it hasn't already
tagged:

  1. read EXIF GPS + timestamp  (exiftool, batched)
  2. resolve location:
       - embedded GPS if present, else the mean GPS of the surrounding folder
         (a day's photos are usually taken within a small area), else --region
     and pick the month (EXIF date, else parsed from folder names like
     "260128-Botanic_Garden" / "2601-Singapur")
  3. build a candidate shortlist from the offline eBird region x month tables,
     optionally boosted by an eBird checklist export (*_observations.csv) that
     sits in the photo folder
  4. classify the photo with BioCLIP 2 (pybioclip, CPU) and get a ranked
     species + confidence
  5. write ONE IMatch-safe XMP sidecar per capture:  <base>.xmp
     (never <base>.jpg.xmp / <base>.cr2.xmp)
  6. keep a resumable manifest + CSV report, and show a live TUI

Originals (CR2/CR3/JPEG/...) are NEVER modified - only `.xmp` sidecars and the
manifest are written. A RAW + JPEG pair of the same capture is processed ONCE
(on the viewable file) and produces a single shared sidecar.

Data management (see pipeline/ebird_data.py):
  birdid.py --list [country|region]    what's available to download
  birdid.py --download <code>          fill in the missing region tables from
                                       real eBird observations (skips parts you
                                       already have; needs --api key; no key =
                                       keep the curated seed)
  birdid.py --update <code>            refresh only the tables whose eBird data
                                       is newer (each table records the date of
                                       the eBird data itself, not the download
                                       date); default target = current year,
                                       so 2025 data is refreshed with 2026
  birdid.py --cache build|clean|update [REGION...]
                                       manage the species-embedding cache: build
                                       precomputes a region's embeddings so a
                                       run encodes nothing on the fly; clean
                                       removes a region's data (or stale species
                                       no table uses); update syncs the cache
                                       with the current eBird tables
  birdid.py --api [<key>]              store / show the eBird API key
  birdid.py --initialize               one-time: download taxonomy + BioCLIP 2
                                       backends so everything works offline

Requirements:
  - exiftool, python3 (PIL, lxml, requests)
  - pybioclip (BioCLIP 2) - the only classifier backend

Usage:
  birdid.py /path/to/photos [--region DE] [options]

See --help for all options. Examples:
  birdid.py ~/Pictures/birds --dry-run              # just scan + plan
  birdid.py ~/Pictures/birds                        # process everything
  birdid.py ~/Pictures/birds --force                # re-ID an already-tagged
                                                    # folder in place (keeps the
                                                    # IMatch sidecars, overwrites
                                                    # only the bird-ID fields)
  birdid.py ~/Pictures/birds --max 10 --no-tui      # first 10, plain output
  birdid.py ~/Pictures/birds --burst                # combine a rapid series of
                                                    # shots of the same bird into
                                                    # one stronger ID (opt-in:
                                                    # --burst / --burst-gap /
                                                    # --burst-sim; the per-capture
                                                    # mode stays the default)
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ebird_data import (COARSE_LABELS, TAXONOMY_2023, TAXONOMY_2026,
                        country_continent_map,
                        country_subcontinent_map)   # shared with --initialize/--download

# --------------------------------------------------------------------------
# Config / defaults
# --------------------------------------------------------------------------

# data dir: defaults to <repo>/ebird (the bundled eBird data), so a fresh
# clone works without any --data-dir override. Override with --data-dir.
DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent / "ebird")
DEFAULT_REGION = "DE"
PIPELINE_VERSION = "1"


def _default_cache_dir() -> Path:
    """Central per-user BioCLIP 2 species-embedding cache.

    Lives under the XDG cache home (~/.cache/birdid/embeddings) - NOT inside
    the photo folders, the repo, or the eBird data dir - so the derived cache
    files never end up in an IMatch-managed tree. Override with --cache-dir.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "birdid" / "embeddings"

# Centralized eBird checklist exports (one file per checklist, e.g.
# S296663962_observations.csv). Override with --checklists-dir or
# BIRDID_CHECKLISTS_DIR; created by --initialize.
DEFAULT_CHECKLISTS_DIR = os.environ.get(
    "BIRDID_CHECKLISTS_DIR") or str(Path.home() / ".config" / "birdid" / "checklists")

# Avibase species-page URL. The Clements v2026 checklist carries each species'
# 'taxon concept ID' (e.g. 'avibase-0DF27D63'); we surface the link for the
# highest-confidence guess so the user can look up full details there.
# Short form https://avibase.ca/<id> redirects to the full bsc-eoc.org page.
AVIBASE_URL = "https://avibase.ca/"

# Image extensions the scanner understands. Keep the first two groups as the
# "viewable" set used for vision input; the rest are RAW-only (embedded preview).
VIEWABLE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".tif", ".tiff"}
ALL_EXTS = VIEWABLE_EXTS | RAW_EXTS

# Region bounding boxes (lat, lon). Photos without GPS fall back to --region.
# Order matters: `region_for` returns the FIRST box that contains the point, so
# the tightest (country -> subcontinent -> continent) must come first.
REGION_BBOXES = {
    "DE": (47.0, 55.0, 5.5, 15.0),     # Germany
    "SG": (1.1, 1.5, 103.5, 104.2),    # Singapore
    "TH": (5.5, 20.5, 97.0, 106.0),    # Thailand
    # subcontinent / continent pools (GPS resolves to the widest pool)
    "SEA": (-11.0, 29.0, 92.0, 141.0),    # Southeast Asia
    "EUROPE": (34.0, 72.0, -11.0, 45.0),  # Europe
    "NA": (5.0, 84.0, -170.0, -50.0),     # North America
    "SA": (-56.0, 12.5, -82.0, -34.0),    # South America
    "AF": (-35.0, 37.5, -18.0, 52.0),     # Africa
    "AS": (-11.0, 78.0, 26.0, 180.0),     # Asia
    "OC": (-50.0, 22.0, 110.0, 180.0),    # Oceania (excl. Antarctica)
}

# Low-confidence fallback: when the base country list yields no confident
# answer, re-score against the union with a wider regional pool. Birds don't
# respect borders, so a species photographed just across a border (or missed by
# the curated country table) still gets a chance. Pools are ordinary region
# tables (same CSV format) covering the surrounding countries. Explicit country
# -> pool overrides come first; every other country then maps to its
# subcontinent, else its continent pool, via the bundled continents reference
# (ebird/continents.csv).

# Continent / region-box codes that must NEVER be read as ISO country codes in
# the country -> wider-pool map. continents.csv also uses these as ISO keys
# (AF=Afghanistan, NA=Namibia, SA=Saudi Arabia, AS=American Samoa), which
# collided with the same-named GPS boxes below and silently scored e.g. a Ghana
# photo (box AF = Africa) against ASIAN species. Excluding them here makes an
# undownloaded continent box report "no data for this region" instead of a
# wrong continent pool.
CONTINENT_CODES = {"NA", "SA", "AF", "AS", "OC", "AN", "EUROPE", "SEA"}


def _build_region_pools() -> dict[str, str]:
    """Country code -> wider pool code for the two-tier fallback.

    Tightest pool first: a country's subcontinent (MY -> AS-SOUTHEAST/SEA),
    then its continent (MY -> AS). Continent codes are excluded as KEYS so a
    GPS box (AF/SA/NA/AS/OC) never resolves through a same-named ISO country.
    SG/TH -> SEA uses the aliased (curated) subcontinent name; every other
    country keeps its subcontinent (e.g. DE -> EU-WEST -> EUROPE via the tree).
    """
    pools: dict[str, str] = {}
    for code, pool in country_subcontinent_map().items():
        if code not in CONTINENT_CODES and code not in pools:
            pools[code] = pool
    for code, pool in country_continent_map().items():
        if code not in CONTINENT_CODES and code not in pools:
            pools[code] = pool
    pools.update({"SG": "SEA", "TH": "SEA"})
    return pools


REGION_POOLS = _build_region_pools()

BIRD_NS = "http://ns.birdpipeline/1.0/"

# --------------------------------------------------------------------------
# eBird region x month data
# --------------------------------------------------------------------------

class SpeciesDB:
    """Loads region tables: region,species,status,month_mask (12 chars, Jan..Dec).

    Two sources, cleanly separated:
      - `curated_dir` (committed seeds, e.g. regions/EUROPE.csv, regions/SEA.csv),
      - `generated_dir` (gitignored, API-derived from `birdid --download`).
    A generated table for a region REPLACES the curated one (preferred); the
    curated seed remains the fallback when no generated table exists.

    Layout: CSVs are found recursively, and `--download` writes the smallest
    division into a continent/subcontinent tree, e.g.
    regions_generated/AS/AS-SOUTHEAST/SG.csv. A CSV's region is its `region`
    field (not the filename). In addition, every directory that contains CSVs
    acts as a region (its name, uppercased) whose candidate set is the union of
    all CSVs beneath it - so AS-SOUTHEAST/, AS/ and a country stored as a
    directory of subnational tables are all queryable, and the two-tier
    fallback can walk country -> subcontinent -> continent.
    """

    def __init__(self, curated_dir: Path, generated_dir: Path | None = None):
        self.curated_dir = Path(curated_dir)
        self.generated_dir = Path(generated_dir) if generated_dir else None
        # region -> (month index 0..11) -> list[(species, status)]
        self._table: dict[str, dict[int, list[tuple[str, str]]]] = {}
        # region -> "curated" | "generated"
        self._source: dict[str, str] = {}
        # region -> file (or dir) it came from + which base it lives under
        self._region_file: dict[str, Path] = {}
        self._region_base: dict[str, Path] = {}

    @staticmethod
    def _parse_csv(csv_path: Path) -> tuple[str, list[tuple[int, str, str]]]:
        """(region, [(month_index, species, status), ...]) from one CSV.

        The region is the first non-comment `region` field; comment lines may
        contain commas so we always key off the first field.
        """
        region = ""
        rows: list[tuple[int, str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row:
                    continue
                if (row.get("region") or "").lstrip().startswith("#"):
                    continue
                species = (row.get("species") or "").strip()
                if not species:
                    continue
                if not region:
                    region = (row.get("region") or "").strip().upper()
                status = (row.get("status") or "resident").strip()
                mask = (row.get("month_mask") or "").strip()
                if len(mask) != 12:
                    continue
                for i, ch in enumerate(mask):
                    if ch == "1":
                        rows.append((i, species, status))
        return region, rows

    @staticmethod
    def _add_rows(bucket: dict[int, list[tuple[str, str]]],
                  rows: list[tuple[int, str, str]]):
        for i, species, status in rows:
            bucket.setdefault(i, []).append((species, status))

    def _load_dir(self, base_dir: Path, source: str, override: bool):
        if not base_dir.is_dir():
            return
        csv_paths = sorted(p for p in base_dir.rglob("*.csv") if p.is_file())
        contents: dict[Path, tuple[str, list]] = {}
        for p in csv_paths:
            region, rows = self._parse_csv(p)
            if region:
                contents[p] = (region, rows)

        # explicit CSV regions (a generated table replaces the curated one)
        for p, (region, rows) in contents.items():
            if override:
                self._table.pop(region, None)
            self._add_rows(self._table.setdefault(region, {}), rows)
            self._region_file[region] = p
            self._region_base[region] = base_dir
            self._source[region] = source

        # dir-union regions: every directory below the base that contains CSVs
        # is a region = union of all CSVs beneath it. An explicit CSV for the
        # same name (e.g. a pooled AS-SOUTHEAST.csv) takes precedence.
        dirs: dict[str, Path] = {}
        for p in contents:
            for anc in p.parents:
                if anc == base_dir:
                    break
                dirs.setdefault(anc.name.upper(), anc)
        for region in sorted(dirs):
            if region in self._table:
                continue
            rows: list[tuple[int, str, str]] = []
            d = dirs[region]
            for p, (_, pr) in contents.items():
                if d in p.parents:
                    rows.extend(pr)
            if rows:
                self._add_rows(self._table.setdefault(region, {}), rows)
                self._region_file[region] = d
                self._region_base[region] = base_dir
                self._source[region] = source

    def load(self):
        if self.curated_dir.is_dir():
            self._load_dir(self.curated_dir, "curated", override=False)
        if self.generated_dir and self.generated_dir.is_dir():
            self._load_dir(self.generated_dir, "generated", override=True)
        if not self._table:
            raise FileNotFoundError(
                f"no region tables loaded from {self.curated_dir} "
                f"(expected curated seeds like EUROPE.csv / SEA.csv)")

    def candidates(self, region: str, month: int, grace: int = 0
                   ) -> list[tuple[str, str]]:
        """Species plausibly present in `region` during `month` (1..12).

        `grace` widens the window to months [month-grace, month+grace],
        wrapping across the year - a species whose month_mask is off by a month
        or two still counts (birds rarely carry a calendar).
        """
        region = region.upper()
        months = self._table.get(region, {})
        grace = max(0, int(grace))
        seen = set()
        out: list[tuple[str, str]] = []
        for i in range(month - 1 - grace, month + grace):
            for sp, st in months.get(i % 12, []):
                if sp.lower() not in seen:
                    seen.add(sp.lower())
                    out.append((sp, st))
        return out

    def tier_codes(self, region: str) -> list[str]:
        """Ordered region codes from the base region up (narrowest first),
        mirroring the tree walk pool_tiers does: for a country table stored at
        <CONT>/<SUB>/<COUNTRY>.csv this yields the subcontinent, then the
        continent. Regions without a tree placement fall back to the
        REGION_POOLS map (country -> continent pool), or just the base region."""
        region = region.upper()
        base_dir = self._region_base.get(region)
        path = self._region_file.get(region)
        codes: list[str] = []
        if path is not None and base_dir is not None:
            seen = set()
            p = path.parent if path.is_file() else path
            for anc in [p] + list(p.parents):
                if anc == base_dir:
                    break
                r = anc.name.upper()
                if r in self._table and r != region and r not in seen:
                    seen.add(r)
                    codes.append(r)
        if not codes:
            # flat file at the base level (legacy layout) or no tree ancestors
            # (e.g. a country whose table only exists as a curated flat seed) ->
            # fall back to the REGION_POOLS map (country -> continent pool).
            pool_code = REGION_POOLS.get(region)
            if pool_code and pool_code in self._table:
                codes.append(pool_code)
        return [region] + codes

    def pool_tiers(self, region: str, month: int, grace: int = 0
                   ) -> list[list[tuple[str, str]]]:
        """Ordered wider candidate lists (wider than the base region), walking
        up the directory tree. See tier_codes() for the region codes."""
        return [self.candidates(code, month, grace)
                for code in self.tier_codes(region)[1:]]

    def expanded(self, region: str, month: int, grace: int = 0
                 ) -> list[tuple[str, str]]:
        """Base candidates + all wider pools (walk up the tree), deduped.

        Used as the low-confidence fallback so species that occur just across a
        border (or were missed by the curated country table) still get a chance.
        """
        region = region.upper()
        seen = set()
        out: list[tuple[str, str]] = []
        grps = [self.candidates(region, month, grace)] + self.pool_tiers(region, month, grace)
        for grp in grps:
            for sp, st in grp:
                if sp.lower() not in seen:
                    seen.add(sp.lower())
                    out.append((sp, st))
        return out

    def source(self, region: str) -> str:
        """'curated' | 'generated' | '' for a region."""
        return self._source.get(region.upper(), "")

    def regions(self):
        return sorted(self._table)


# v2026 'category' values that carry a real English common name for a bird
# taxon; everything else in the integrated checklist is noise for name lookup
# (subspecies rows, family rows, the citation header).
_V2026_SKIP_CATEGORIES = {"subspecies", "family", ""}


def find_taxonomy(data_dir: Path) -> Path | None:
    """Best available taxonomy: the eBird API v2023 file, else the bundled
    Clements v2026 integrated checklist (for installs without API access)."""
    for name in (TAXONOMY_2023, TAXONOMY_2026):
        p = Path(data_dir) / name
        if p.is_file():
            return p
    return None


def load_taxonomy(path: Path | None) -> dict[str, str]:
    """Common-name -> scientific-name map from either eBird taxonomy format.

    Handles both the eBird API CSV (COMMON_NAME / SCIENTIFIC_NAME, v2023) and
    the Clements v2026 integrated checklist (English name / scientific name +
    category), so installs without API/network access can use the bundled file.
    """
    tax: dict[str, str] = {}
    if not path:
        return tax
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if not row:
                    continue
                common = (row.get("COMMON_NAME") or row.get("English name") or "").strip()
                sci = (row.get("SCIENTIFIC_NAME") or row.get("scientific name") or "").strip()
                if not common or not sci:
                    continue
                if row.get("COMMON_NAME") is None:   # Clements v2026 format
                    if (row.get("category") or "").strip() in _V2026_SKIP_CATEGORIES:
                        continue
                tax[common] = sci
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
# Avibase (from the bundled Clements v2026 checklist)
# --------------------------------------------------------------------------

def load_avibase(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """(sci-name -> Avibase id, common-name -> Avibase id).

    The Avibase taxon concept ID only exists in the bundled Clements v2026
    integrated checklist ('taxon concept ID' column, e.g. 'avibase-0DF27D63');
    the eBird API v2023 taxonomy has no such column. The stored id is the bare
    form ('0DF27D63') used by both the short link (avibase.ca/<id>) and the
    full page (species.jsp?avibaseid=<id>). Both dicts are empty when the v2026
    file isn't present, so the link is simply omitted.
    """
    by_sci: dict[str, str] = {}
    by_common: dict[str, str] = {}
    p = Path(data_dir) / TAXONOMY_2026
    if not p.is_file():
        return by_sci, by_common
    try:
        with open(p, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if not row:
                    continue
                if (row.get("category") or "").strip() != "species":
                    continue
                cid = (row.get("taxon concept ID") or "").strip()
                if not cid.startswith("avibase-"):
                    continue
                cid = cid[len("avibase-"):]        # store the bare id (0DF27D63)
                sci = (row.get("scientific name") or "").strip()
                common = (row.get("English name") or "").strip()
                if sci:
                    by_sci[sci] = cid
                if common:
                    by_common[common] = cid
    except OSError:
        pass
    return by_sci, by_common


def avibase_url_for(by_sci: dict[str, str], by_common: dict[str, str],
                    scientific: str, common: str) -> str:
    """Avibase page URL for the top guess; '' when the id is unknown."""
    cid = by_sci.get(scientific or "") or by_common.get(common or "") or ""
    return f"{AVIBASE_URL}{cid}" if cid else ""

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


EXIF_CHUNK = 500   # files per exiftool call in read_exif_batch


def _exif_one_chunk(paths: list[Path]) -> dict[str, dict]:
    """One exiftool call for a chunk of files -> {resolved_path_str: meta}."""
    meta_map: dict[str, dict] = {}
    if not paths:
        return meta_map
    cmd = ["exiftool", "-json", "-n",
           "-GPSLatitude", "-GPSLongitude",
           "-GPSLatitudeRef", "-GPSLongitudeRef",
           "-DateTimeOriginal", "-CreateDate"] + [str(p) for p in paths]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"exiftool batch failed: {e}") from e
    try:
        rows = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        rows = []
    for d in rows:
        src = str(Path(d.get("SourceFile", "")).resolve())
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
        meta_map[src] = {"lat": lat, "lon": lon, "datetime": dt}
    return meta_map


def read_exif_batch(paths: list[Path], on_progress=None) -> dict[str, dict]:
    """Chunked exiftool calls -> {resolved_path_str: meta} (identical merged
    result to one giant call, but reports per-chunk progress and avoids one
    enormous command line).

    meta = {"lat": float|None, "lon": float|None, "datetime": datetime|None}.
    Missing/broken files yield an empty meta dict.
    on_progress(done, total) is called after every chunk if given.
    """
    meta_map: dict[str, dict] = {}
    if not paths:
        return meta_map
    total = len(paths)
    done = 0
    for start in range(0, total, EXIF_CHUNK):
        meta_map.update(_exif_one_chunk(paths[start:start + EXIF_CHUNK]))
        done += min(EXIF_CHUNK, total - start)
        if on_progress:
            on_progress(done, total)
    return meta_map


def compute_dir_gps(meta_map: dict[str, dict]) -> dict[Path, tuple[float, float]]:
    """Mean GPS per directory, computed from files that DO have coordinates."""
    sums: dict[Path, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for p_str, m in meta_map.items():
        if m.get("lat") is not None and m.get("lon") is not None:
            d = Path(p_str).parent
            s = sums[d]
            s[0] += m["lat"]; s[1] += m["lon"]; s[2] += 1
    return {d: (s[0] / s[2], s[1] / s[2]) for d, s in sums.items() if s[2] > 0}


def resolve_location(path: Path, meta: dict, dir_gps: dict[Path, tuple[float, float]]):
    """(lat, lon, source) - embedded GPS, else folder-mean GPS, else (None, None).

    source is one of "embedded" / "folder-mean" / "none".
    """
    if meta.get("lat") is not None and meta.get("lon") is not None:
        return meta["lat"], meta["lon"], "embedded"
    d = path.resolve().parent
    seen = set()
    while d not in seen and d != d.parent:
        seen.add(d)
        if d in dir_gps:
            lat, lon = dir_gps[d]
            return lat, lon, "folder-mean"
        d = d.parent
    return None, None, "none"


_FOLDER_YYMMDD = re.compile(r"(?:^|[^0-9])(\d{6})(?:[-_]|$)")
_FOLDER_YYMM = re.compile(r"(?:^|[^0-9])(\d{4})(?:[-_]|$)")


def folder_date(path: Path) -> datetime | None:
    """Guess a capture date from folder names like '260128-Botanic_Garden'
    (YYMMDD) inside '2601-Singapur' (YYMM). Returns None if nothing parses."""
    # YYMMDD from the capture's own folder, then YYMM from its parent
    for d, full in ((path.parent, True), (path.parent.parent, True),
                    (path.parent.parent.parent, False)):
        m = _FOLDER_YYMMDD.search(d.name) if full else _FOLDER_YYMM.search(d.name)
        if not m:
            continue
        s = m.group(1)
        try:
            if full and len(s) == 6:
                return datetime(2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]))
            if len(s) == 4:
                return datetime(2000 + int(s[0:2]), int(s[2:4]), 1)
        except ValueError:
            continue
    return None


CHECKLIST_PATTERNS = ("*_observations.csv", "*checklist*.csv", "*ebird*.csv")


def find_checklist(directory: Path, root: Path) -> Path | None:
    """Locate an optional eBird checklist export near the photos."""
    for base in (directory, root):
        try:
            for pat in CHECKLIST_PATTERNS:
                for p in sorted(base.glob(pat)):
                    return p
        except OSError:
            continue
    return None


def load_checklist(path: Path) -> list[str]:
    """Species common names from an eBird export (header: Species,...)."""
    species: list[str] = []
    if not path:
        return species
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                name = (row.get("Species") or "").strip()
                if name and name not in species:
                    species.append(name)
    except OSError:
        pass
    return species


# --------------------------------------------------------------------------
# Centralized eBird checklist exports (one file per checklist, with a date)
# --------------------------------------------------------------------------

def _parse_checklist_date(s: str) -> datetime | None:
    """Parse an eBird 'Observation Date' cell, e.g. 'Jan 28, 2026'."""
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


class CentralChecklists:
    """Centralized eBird checklist exports: species by (year, month).

    One file per exported checklist (e.g. ``S296663962_observations.csv``) in a
    single folder (default ``~/.config/birdid/checklists/``). Each row carries a
    ``Species`` and an ``Observation Date``. A persistent index
    (``<dir>/index.json``) remembers each file's mtime+size and its parsed
    entries, so on reruns only new/changed files are re-read; the rest are
    served from the cached index. Folders are usually one-per-date, so entries
    are keyed by the checklist's (year, month) and matched to a capture's
    ``folder_date()`` (feature 2).
    """

    INDEX_NAME = "index.json"

    def __init__(self, checklists_dir: Path | None, taxonomy: dict[str, str]):
        self.dir = Path(checklists_dir) if checklists_dir else None
        self.taxonomy = taxonomy
        # (year, month) -> list[(species, status)]
        self.by_month: dict[tuple[int, int], list[tuple[str, str]]] = {}
        self.loaded = 0          # species loaded from central checklists
        self.index: dict = {}    # path -> {mtime, size, entries}
        self._changed = False
        if self.dir:
            self._load()

    # -- persistence -------------------------------------------------------

    def _index_path(self) -> Path:
        return self.dir / self.INDEX_NAME

    def _read_index(self):
        try:
            return json.loads(self._index_path().read_text()) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_index(self):
        try:
            self._index_path().write_text(
                json.dumps(self.index, indent=1, ensure_ascii=False))
        except OSError:
            pass

    # -- loading -----------------------------------------------------------

    def _load(self):
        if not self.dir.is_dir():
            return
        self.index = self._read_index()
        try:
            for pat in CHECKLIST_PATTERNS:
                for p in sorted(self.dir.glob(pat)):
                    self._load_file(p)
        except OSError:
            pass
        if self._changed:
            self._write_index()

    def _load_file(self, path: Path):
        try:
            st = path.stat()
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            return
        key = str(path)
        cached = self.index.get(key)
        if cached and (cached.get("mtime_ns"), cached.get("size")) == sig:
            self._apply_entries(cached.get("entries") or [])
            return
        entries = self._parse_file(path)
        self.index[key] = {"mtime_ns": sig[0], "size": sig[1], "entries": entries}
        self._changed = True
        self._apply_entries(entries)

    def _parse_file(self, path: Path) -> list[list]:
        """-> [[species, 'YYYY-MM-DD', status], ...] (raw, for the index)."""
        rows: list[list] = []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    name = (row.get("Species") or "").strip()
                    d = _parse_checklist_date(row.get("Observation Date"))
                    if not name or d is None:
                        continue
                    rows.append([name, d.strftime("%Y-%m-%d"), "central-checklist"])
        except OSError:
            pass
        return rows

    def _apply_entries(self, entries: list[list]):
        for name, date_str, status in entries:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            common = normalize_species(name, self.taxonomy)[0]
            self.by_month.setdefault((d.year, d.month), []).append((common, status))
            self.loaded += 1

    def species_for(self, year: int, month: int) -> list[tuple[str, str]]:
        """Central-checklist species for a (year, month), deduped."""
        return self.by_month.get((year, month), [])

    def report(self) -> str:
        if not self.dir or not self.dir.is_dir():
            return "central checklists: none"
        return (f"central checklists: {len(self.index)} file(s), "
                f"{self.loaded} species (index: {self._index_path()})")


# --------------------------------------------------------------------------
# Per-folder reports + parent index (feature 1)
# --------------------------------------------------------------------------

def _report_cols() -> list[str]:
    return ["capture", "file", "viewable", "region", "month", "species",
            "scientific_name", "confidence", "status", "gps", "region_source",
            "date_source", "alternatives", "reasoning", "avibase_url",
            "datetime", "central_added", "central_outside"]


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = _report_cols()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_existing_report(path: Path) -> dict[str, dict]:
    """Existing per-folder report rows keyed by capture base name."""
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                base = (r.get("capture") or "").strip()
                if base:
                    rows[base] = r
    except OSError:
        pass
    return rows


def _manifest_to_row(base: str, entry: dict) -> dict:
    files = entry.get("files") or []
    primary = files[0] if files else ""
    viewable = Path(primary).name if primary else ""
    month = entry.get("month")
    conf = entry.get("confidence")
    return {
        "capture": base, "file": primary, "viewable": viewable,
        "region": entry.get("region", ""),
        "month": f"{month:02d}" if isinstance(month, int) else "",
        "species": entry.get("species", ""),
        "scientific_name": entry.get("scientific", ""),
        "confidence": f"{conf:.2f}" if isinstance(conf, (int, float)) else "",
        "status": entry.get("status", ""),
        "gps": entry.get("gps", ""),
        "date_source": entry.get("date_source", ""),
        "alternatives": "; ".join(entry.get("alternatives") or []),
        "reasoning": entry.get("reasoning", ""),
        "avibase_url": entry.get("avibase_url", ""),
        "datetime": entry.get("datetime", ""),
        "central_added": entry.get("central_added", ""),
        "central_outside": entry.get("central_outside", ""),
    }


def write_folder_reports(root: Path, manifest: dict, report_rows: list[dict]):
    """Write a full-snapshot report per folder + a parent index per subfolder.

    Feature 1: every folder that has captures gets its own ``birdid_report.csv``
    (a full snapshot: this run + previously recorded). Every folder that has
    subfolders-with-reports gets a ``birdid_index.csv`` pointing at each
    subfolder's report/index with summary counts, so a rerun picks up new
    folders and running on a subfolder alone also works. The run-root manifest
    stays the resumable store; ``--report <path>`` still writes a flat
    aggregate on top.
    """
    written: list[str] = []

    # this run's rows, keyed by capture base per folder
    run_by_folder: dict[Path, dict[str, dict]] = defaultdict(dict)
    for row in report_rows:
        folder = Path(row["file"]).parent
        run_by_folder[folder][row["capture"]] = row

    # manifest entries grouped by folder (captures ever processed for this root)
    manifest_by_folder: dict[Path, dict[str, dict]] = defaultdict(dict)
    for sidecar_str, entry in manifest.items():
        sidecar = Path(sidecar_str)
        manifest_by_folder[sidecar.parent][sidecar.stem] = entry

    # folders of interest: this run's folders + manifest folders + root
    folders = set(run_by_folder) | set(manifest_by_folder)
    # include existing report/index folders under root (from earlier runs)
    if root.is_dir():
        for p in root.rglob("birdid_report.csv"):
            folders.add(p.parent)
        for p in root.rglob("birdid_index.csv"):
            folders.add(p.parent)
    folders = {f for f in folders if f == root or root in f.parents}

    # 1) per-folder full snapshots: existing report <- manifest <- this run
    for folder in sorted(folders):
        merged = _read_existing_report(folder / "birdid_report.csv")
        for base, entry in manifest_by_folder.get(folder, {}).items():
            merged.setdefault(base, _manifest_to_row(base, entry))
        for base, row in run_by_folder.get(folder, {}).items():
            merged[base] = row
        if not merged:
            continue
        path = folder / "birdid_report.csv"
        _write_csv(path, [merged[b] for b in sorted(merged)])
        written.append(str(path))

    # 2) parent index: every folder with subfolders that have a report/index
    report_dirs = {f for f in folders if (f / "birdid_report.csv").is_file()}
    index_dirs = {f for f in folders if (f / "birdid_index.csv").is_file()}
    all_dirs = set()
    for d in report_dirs | index_dirs:
        p = d
        while True:
            all_dirs.add(p)
            if p == root or p.parent == p:
                break
            p = p.parent
    # a folder gets an index iff it has a subfolder in all_dirs
    will_index = {f for f in all_dirs
                  if any(d.parent == f and d != f for d in all_dirs)}
    # deepest first so a child index exists before the parent references it
    for folder in sorted(all_dirs, key=lambda p: -len(p.parts)):
        subdirs = sorted({d for d in all_dirs if d.parent == folder and d != folder})
        if not subdirs:
            continue

        def subtree_counts(folder_d: Path) -> dict:
            """Aggregate report counts over folder_d and its whole subtree."""
            rows = _read_existing_report(folder_d / "birdid_report.csv")
            counts = {"captures": len(rows), "done": 0, "error": 0,
                      "pending": 0, "species": 0}
            species = set()
            for r in rows.values():
                st = r.get("status", "")
                if st in ("done", "non-bird"):
                    counts["done"] += 1
                elif st == "error":
                    counts["error"] += 1
                else:
                    counts["pending"] += 1
                if r.get("species"):
                    species.add(r["species"])
            counts["species"] = len(species)
            for child in {d for d in all_dirs if d.parent == folder_d}:
                c = subtree_counts(child)
                for k in ("captures", "done", "error", "pending"):
                    counts[k] += c[k]
                counts["species"] += c["species"]
            return counts

        rows = []
        for sub in subdirs:
            report = sub / "birdid_report.csv"
            index = sub / "birdid_index.csv"
            counts = subtree_counts(sub)
            rows.append({
                "folder": sub.name,
                "report": report.relative_to(folder).as_posix() if sub in report_dirs else "",
                "index": index.relative_to(folder).as_posix() if sub in will_index else "",
                **counts,
            })
        path = folder / "birdid_index.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["folder", "report", "index",
                                              "captures", "done", "error",
                                              "pending", "species"])
            w.writeheader()
            w.writerows(rows)
        written.append(str(path))

    return written

# --------------------------------------------------------------------------
# Classifier backends
# --------------------------------------------------------------------------

class Classifier:
    """Base interface: classify one photo against a candidate species list.

    Returns a dict with at least:
        {"species": common name, "confidence": 0..1,
         "alternatives": [names], "reasoning": str}
    """

    name = "base"

    def __init__(self, args):
        self.args = args
        # optional callable(stage, message) for live background feedback (wired
        # to the TUI in main()); long steps (model load, encoding, pool re-score)
        # report so a run never looks stalled/crashed.
        self.progress: "callable | None" = None
        # optional callable(stage, done, total, message) for per-step progress
        # bars (cache build, ...) - independent of the text `progress` channel.
        self.progress_bar: "callable | None" = None

    def _report(self, stage: str, message: str):
        if getattr(self, "progress", None):
            self.progress(stage, message)

    def _report_bar(self, stage: str, done: int, total: int, message: str = ""):
        if getattr(self, "progress_bar", None):
            self.progress_bar(stage, done, total, message)

    def describe(self) -> str:
        return self.name

    def close(self):
        pass

    def classify(self, image: Path, ladder: list[tuple[str, list[tuple[str, str]]]],
                 tax: dict[str, str], region: str, month: int, dt: datetime) -> dict:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Shared BioCLIP 2 backbone + on-disk species-embedding cache
# --------------------------------------------------------------------------
#
# BioCLIP's CustomLabelsClassifier encodes EVERY species against all 80
# ImageNet prompt templates (~2.7 s/species on this CPU) each time it is
# constructed - and the old code constructed one per photo AND per pool
# expansion (each also re-loading the ~4.5 s model). For a 369-photo folder
# that was tens of minutes of repeated work. The fix has two halves:
#
#   1. ONE shared compiled backbone (model + tokenizer) for the whole run.
#   2. A reusable on-disk cache of the final per-species embedding vectors
#      (bit-identical to the 80-template mean), so a species is encoded once
#      ever, not once per photo.

_BACKBONE: dict = {"model": None}   # single shared compiled BioCLIP 2 model


def _load_backbone():
    """Shared BioCLIP 2 backbone (model + tokenizer + preprocess) for the whole
    run - a single ~4.5 s model load instead of one per classifier."""
    if _BACKBONE["model"] is None:
        from bioclip.predict import BaseClassifier, create_bioclip_tokenizer
        backbone = BaseClassifier()
        backbone.tokenizer = create_bioclip_tokenizer(backbone.model_str)
        _BACKBONE["model"] = backbone
    return _BACKBONE["model"]


def _encode_text_name(backbone, name: str) -> np.ndarray:
    """One species' text embedding - bit-identical to
    CustomLabelsClassifier._get_txt_embeddings([name]): the 80 ImageNet
    template prompts, tokenized + encoded, L2-normalized per prompt, then the
    normalized mean."""
    import torch
    from bioclip.predict import OPENA_AI_IMAGENET_TEMPLATE
    txts = [t(name) for t in OPENA_AI_IMAGENET_TEMPLATE]
    txts = backbone.tokenizer(txts).to(backbone.device)
    with torch.no_grad():
        feats = backbone.model.encode_text(txts)
        feats = torch.nn.functional.normalize(feats, dim=-1).mean(dim=0)
        feats = feats / feats.norm()
    return feats.cpu().numpy().astype(np.float32)


class SpeciesEmbeddingCache:
    """Reusable on-disk per-species text-embedding cache.

    Layout in a central cache dir (default ~/.cache/birdid/embeddings, override
    with --cache-dir - kept out of the photo folders, the repo and the eBird
    data dir so no derived files land in an IMatch-managed tree):
      index.json      {"model": <bioclip model str>, "names": [n, ...]}
      embeddings.npy  float32 [N, D]   (row i == embedding of names[i];
                                        D = model embedding dim, 768 for
                                        BioCLIP 2 - read from the file, not
                                        hardcoded)

    Embeddings are computed on demand (one species at a time, ~2.7 s each on
    this CPU) and appended to the matrix, so a species is encoded exactly once
    ever across all runs and all folders. Different BioCLIP models are keyed
    separately (a stale matrix for another model is rebuilt).
    """

    def __init__(self, cache_dir: Path | None):
        self.dir = cache_dir
        self._names: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._model: str | None = None

    # -- persistence --------------------------------------------------------

    def _ensure_loaded(self):
        if self._matrix is not None:
            return
        if self.dir is None:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        npy, js = self.dir / "embeddings.npy", self.dir / "index.json"
        if npy.is_file() and js.is_file():
            try:
                meta = json.loads(js.read_text(encoding="utf-8"))
                m = np.load(str(npy))
                names = meta.get("names") or []
                # Accept any 2-D matrix: the embedding dim is whatever the model
                # produced (768 for BioCLIP 2), read from the file - never
                # assumed, so a saved cache is reused instead of recomputed.
                if m.ndim == 2 and len(names) == m.shape[0]:
                    self._matrix = m.astype(np.float32)
                    self._names = list(names)
                    self._index = {n: i for i, n in enumerate(self._names)}
                    self._model = meta.get("model")
                    return
            except Exception:
                pass
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def _reset(self):
        self._names = []
        self._index = {}
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._model = None

    def _save(self):
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        # np.save appends '.npy' when the name lacks it, so the temp file is
        # 'embeddings.tmp' -> 'embeddings.tmp.npy' (the index.json.tmp keeps
        # its own suffix). Write both, then atomically replace.
        tmp_n = self.dir / "embeddings.tmp"
        tmp_j = self.dir / "index.json.tmp"
        np.save(str(tmp_n), self._matrix)
        with open(tmp_j, "w", encoding="utf-8") as f:
            json.dump({"model": self._model, "names": self._names}, f)
        os.replace(self.dir / "embeddings.tmp.npy", self.dir / "embeddings.npy")
        os.replace(tmp_j, self.dir / "index.json")

    # -- lookup / compute ---------------------------------------------------

    def embed(self, name: str, backbone) -> np.ndarray:
        """Embedding for one species name, computing + persisting on first use."""
        self._ensure_loaded()
        model = getattr(backbone, "model_str", None)
        if self._model != model:
            self._reset()
            self._model = model
        if name in self._index:
            return self._matrix[self._index[name]]
        vec = _encode_text_name(backbone, name)
        self._append(name, vec)
        return vec

    def _append(self, name: str, vec: np.ndarray):
        row = vec.reshape(1, -1).astype(np.float32)
        self._matrix = (row if self._matrix.shape[0] == 0
                        else np.vstack([self._matrix, row]))
        self._names.append(name)
        self._index[name] = len(self._names) - 1
        self._save()

    def matrix_for(self, names: list[str], backbone,
                   on_progress=None) -> np.ndarray:
        """[DIM, n] column-stacked embeddings for `names` in order - identical
        to CustomLabelsClassifier._get_txt_embeddings(list(names)). Missing
        species are encoded now (on_progress(name, i, total) if given)."""
        self._ensure_loaded()
        model = getattr(backbone, "model_str", None)
        if self._model != model:
            self._reset()          # different model -> rebuild for this one
            self._model = model
        miss = [n for n in names if n not in self._index]
        for j, n in enumerate(miss, 1):
            if on_progress:
                on_progress(n, j, len(miss))
            self._append(n, _encode_text_name(backbone, n))
        cols = [self._matrix[self._index[n]] for n in names]
        return np.stack(cols, axis=1)

    def names(self) -> list[str]:
        """Every cached species name, in matrix row order (the order they were
        first encoded). Used by --cache build/clean to report + prune."""
        self._ensure_loaded()
        return list(self._names)

    def retain(self, keep: set[str]) -> int:
        """Drop every cached species not in `keep` (case-insensitive), keeping
        the surviving rows in their original order; the COARSE_LABELS coarse
        prompts are ALWAYS kept (the classifier needs them for every photo), so
        they survive even a keep-set that omits them. Rewrites both files and
        returns the number of species removed. No model needed."""
        self._ensure_loaded()
        if not self._names:
            return 0
        keep_l = {k.lower() for k in keep}
        keep_l.update(c.lower() for c in COARSE_LABELS)
        new_names = [n for n in self._names if n.lower() in keep_l]
        removed = len(self._names) - len(new_names)
        if not removed:
            return 0
        pos = {n: i for i, n in enumerate(self._names)}
        self._matrix = self._matrix[[pos[n] for n in new_names]]
        self._names = new_names
        self._index = {n: i for i, n in enumerate(new_names)}
        self._save()
        return removed


def _cache_needed_species(db: SpeciesDB, region: str, grace: int = 1
                          ) -> list[str]:
    """Every species the classifier could ever score for `region`.

    The union over ALL months of the base region plus every wider tier pool
    (subcontinent, then continent), each with the month window grace-widened -
    the same candidate sets the ladder and the low-confidence fallback use, so
    `--cache build <region>` precomputes exactly what a run over that region
    will need, and `--cache clean <region>` / `--cache update` know exactly
    which cached species belong to that region.
    """
    out: list[str] = []
    seen: set[str] = set()
    for code in db.tier_codes(region):
        for month in range(1, 13):
            for sp, _st in db.candidates(code, month, grace):
                if sp.lower() not in seen:
                    seen.add(sp.lower())
                    out.append(sp)
    return out


def _cache_union(db: SpeciesDB, regions: list[str]) -> list[str]:
    """Ordered list of every species a run over `regions` can need: the union
    of each region's _cache_needed_species plus the COARSE_LABELS coarse
    prompts (scored for every photo regardless of region)."""
    out: list[str] = []
    seen: set[str] = set()
    for reg in regions:
        for sp in _cache_needed_species(db, reg):
            if sp.lower() not in seen:
                seen.add(sp.lower())
                out.append(sp)
    for c in COARSE_LABELS:
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def run_cache(data_dir: Path, cache_dir: Path, tokens: list[str]):
    """`--cache build | clean | update [REGION...]`.

    All three work on the central per-species embedding cache and accept
    region codes from `db.regions()` (countries like DE, subcontinents like
    EU-WEST or AS-SOUTHEAST, or continents like EUROPE); no region = every
    loaded region. COARSE_LABELS are never removed.

      build   precompute embeddings for the region(s)' species (base + wider
              pools), so a later run over that region encodes nothing on the
              fly. Loads the BioCLIP 2 backbone; skips already-cached species.
      clean   GET RID of data. With regions: remove exactly that region's
              species (base + its wider pools) from the cache and keep every
              other species - shared species it also uses are dropped too
              (they are part of its cached data) and re-encoded on demand if
              another region needs them. No region: drop the stale species no
              loaded table can use any more (e.g. after an eBird refresh). No
              model needed.
      update  sync the cache with the current eBird tables: drop species no
              longer in any loaded table AND encode species the region(s) now
              need that aren't cached yet (incremental - only the delta). The
              "is my cache still current?" command. No-arg = whole cache.
    """
    cmd = tokens[0].lower() if tokens else ""
    if cmd not in ("build", "clean", "update"):
        given = tokens[0] if tokens else ""
        raise ValueError(
            f"unknown --cache subcommand {given!r} "
            "(expected: --cache build [REGION...]  |  --cache clean [REGION...]  "
            "|  --cache update [REGION...])")
    requested = [t.upper() for t in tokens[1:]]

    db = SpeciesDB(data_dir / "regions", data_dir / "regions_generated")
    db.load()
    available = set(db.regions())
    if requested:
        bad = [r for r in requested if r not in available]
        if bad:
            raise ValueError(
                f"unknown region code(s): {', '.join(bad)} "
                f"(available: {', '.join(sorted(available))})")
        regions = requested
    else:
        regions = sorted(available)     # all loaded regions

    # what each requested region needs, and the union over EVERY loaded region
    # (the "any current table can use this species" reference for pruning)
    need_of = {reg: set(_cache_needed_species(db, reg)) for reg in regions}
    every_need = set(COARSE_LABELS)
    for reg in sorted(available):
        every_need |= set(_cache_needed_species(db, reg))
    every_l = {n.lower() for n in every_need}

    cache = SpeciesEmbeddingCache(cache_dir)

    def _prog(name: str, i: int, total: int):
        sys.stdout.write(f"\r  encoding {i:>3}/{total:<3}  {name[:52]:<52}")
        sys.stdout.flush()

    if cmd == "build":
        needed = _cache_union(db, regions)
        backbone = _load_backbone()
        t0 = time.time()
        print(f"birdid: --cache build {', '.join(regions)} "
              f"({len(needed)} species needed, "
              f"{len(cache.names())} already cached)")
        cache.matrix_for(needed, backbone, on_progress=_prog)
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"birdid: cache ready at {cache.dir} "
              f"({len(cache.names())} species, {time.time() - t0:.0f}s)")

    elif cmd == "clean":
        if not requested:
            # no-arg clean: drop the stale species NO loaded table can use
            # (species from tables that have since been refreshed / removed)
            before = len(cache.names())
            keep = {n for n in cache.names() if n.lower() in every_l}
            removed = cache.retain(keep)
            print(f"birdid: --cache clean (all loaded regions): kept "
                  f"{before - removed}/{before} species "
                  f"({removed} stale removed)")
        else:
            # region clean: get rid of THAT region's cached data - drop the
            # union of its needed species (base + its wider pools), keep every
            # other species. Shared species that the region also needs are
            # dropped too (they are part of its cached data) and simply
            # re-encoded if another region needs them again.
            requested_need = set()
            for reg in regions:
                requested_need |= need_of[reg]
            drop_l = {d.lower() for d in requested_need}
            before = len(cache.names())
            keep = {n for n in cache.names() if n.lower() not in drop_l}
            removed = cache.retain(keep)
            print(f"birdid: --cache clean {', '.join(regions)}: removed "
                  f"{removed} species belonging to that region, kept everything "
                  f"else ({before - removed} species remain)")
            if not removed:
                print("        (none of that region's species were cached)")

    else:                               # update
        # 1) drop species no loaded table can use any more (stale after a
        #    refresh) - no model needed
        before = len(cache.names())
        keep = {n for n in cache.names() if n.lower() in every_l}
        removed = cache.retain(keep)
        # 2) encode the species the requested region(s) now need that aren't
        #    cached yet - incremental, only the delta
        needed = _cache_union(db, regions)
        cached_l = {n.lower() for n in cache.names()}
        miss = [n for n in needed if n.lower() not in cached_l]
        t0 = time.time()
        if miss:
            backbone = _load_backbone()
            print(f"birdid: --cache update: {removed} stale dropped, encoding "
                  f"{len(miss)} newly-needed species...")
            cache.matrix_for(miss, backbone, on_progress=_prog)
            sys.stdout.write("\n")
            sys.stdout.flush()
        if removed or miss:
            print(f"birdid: cache updated at {cache.dir}: dropped {removed} "
                  f"stale, added {len(miss)} -> {len(cache.names())} species "
                  f"({time.time() - t0:.0f}s)")
        else:
            print(f"birdid: --cache update: cache is up to date "
                  f"({len(cache.names())} species, no changes)")


def _build_ladder(db: SpeciesDB, region: str, month: int, grace: int,
                  checklist: list[tuple[str, str]],
                  central_list: list[tuple[str, str]]
                  ) -> list[tuple[str, list[tuple[str, str]]]]:
    """Staged candidate ladder for one capture.

    The base region is ALWAYS scored with the month window already widened to
    [month-grace, month+grace] (strict ∪ ±grace): a species photographed at a
    season boundary (e.g. Asian Openbill in early January, whose month_mask
    starts in February) only appears once the window is widened - so stopping
    on a strict-only softmax can mis-ID a confident-looking wrong species
    (White-shouldered Starling 76% instead of Asian Openbill 95%). The
    --expand-threshold therefore does NOT gate the month grace, only widening
    to wider REGION tiers - and each wider tier is likewise grace-widened, for
    the same season-boundary reason.

    Order: (base ±grace) -> (tier1 ±grace) -> (tier2 ±grace) -> ... with
    every stage = checklist + central + that level's grace-widened region
    candidates, deduped. The classifier scores the small base set first and
    only walks up the region tree while the top score stays below
    --expand-threshold.
    """
    g = max(0, int(grace))
    ladder: list[tuple[str, list[tuple[str, str]]]] = []
    for code in db.tier_codes(region):
        level = db.candidates(code, month, g)
        seen = set()
        stage: list[tuple[str, str]] = []
        for grp in (checklist, central_list, level):
            for s, st in grp:
                if s.lower() not in seen:
                    seen.add(s.lower())
                    stage.append((s, st))
        label = code if not g else f"{code} ±{g}mo"
        ladder.append((label, stage))
    return ladder


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embedding vectors (features are already
    L2-normalized by BioCLIP, so this is their dot product; the norm handles
    the general case). Used as the same-bird gate inside a burst."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BioclipClassifier(Classifier):
    """BioCLIP 2 (pybioclip) - text-conditioned species classifier on CPU.

    Two-stage design (matches the project goal for birds + macro subjects):
      stage 1 (coarse, cheap): softmax over broad categories (bird / reptile /
        insect / ...) - decides whether the subject is a bird.
      stage 2 (fine):
        - bird   -> the staged candidate ladder (region/month/checklist priors)
                    -> ranked percentages.
        - non-bird -> by default the bird pre-filter records the coarse
                    category (status 'non-bird') and SKIPS the fine stage, so
                    non-bird shots cost almost nothing. With --analyze-all the
                    subject is fully analysed via TreeOfLifeClassifier at
                    SPECIES rank (whole tree of life, ~1.6M species).

    Performance: one shared compiled backbone for the whole run, and every
    species' text embedding comes from an on-disk cache (bit-identical to the
    80-ImageNet-template mean), so a photo's fine-stage cost is a single image
    encode + softmax regardless of candidate-list size - no repeated model
    loads, no repeated per-species text encodes.

    Requires:  pip install pybioclip   (pulls torch + torchvision; the
    BioCLIP 2 weights and TreeOfLife datafiles download on first use).
    """

    name = "bioclip"

    def __init__(self, args):
        super().__init__(args)
        try:
            import bioclip  # noqa: F401  (distribution: pybioclip, module: bioclip)
        except ImportError as e:
            raise RuntimeError(
                "pybioclip is not installed - run once:\n"
                "  pip install pybioclip\n"
                "(this pulls in torch + torchvision; model weights download "
                "on first use)") from e
        self._coarse = None
        self._cache = SpeciesEmbeddingCache(Path(args.cache_dir))
        self._tol = None
        self._Rank = None

    def _load_coarse(self):
        if self._coarse is None:
            self._coarse = _load_backbone()
        return self._coarse

    def _encode_image(self, image: str) -> np.ndarray:
        """One normalized image embedding (CPU, float32 [DIM]) - the image side
        of the softmax, encoded exactly once per photo."""
        feat = self._load_coarse().create_image_features_for_image(
            image, normalize=True)
        return feat.cpu().numpy().astype(np.float32)

    def _probs_for(self, img_feat: np.ndarray, names: list[str],
                   on_progress=None) -> np.ndarray:
        """Full softmax probability vector over `names` (float32 [len(names)]),
        bit-identical to CustomLabelsClassifier.create_probabilities: the
        text matrix comes from the shared cache and the logit scale is applied
        to the image features BEFORE the matmul, exactly as pybioclip does.
        Shared by _score_names (top-k) and classify_burst (full distribution
        for cross-frame averaging)."""
        import torch
        backbone = self._load_coarse()
        txt = self._cache.matrix_for(names, backbone, on_progress=on_progress)
        img_t = torch.from_numpy(img_feat[None, :]).to(backbone.device)
        txt_t = torch.from_numpy(txt).to(backbone.device)
        with torch.no_grad():
            logits = (backbone.model.logit_scale.exp() * img_t) @ txt_t
            return torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)

    def _score_names(self, img_feat: np.ndarray, names: list[str], k: int,
                     on_progress=None) -> list[tuple[str, float]]:
        """softmax(scale * img_feat @ txt_embeddings) over `names`, top-k."""
        probs = self._probs_for(img_feat, names, on_progress=on_progress)
        kk = min(k, probs.size)
        order = np.argsort(probs)[::-1][:kk]
        return [(names[int(i)], float(probs[i])) for i in order]

    def _embed_progress(self, name: str, i: int, total: int):
        # routed to the per-step bar (not a log line per species - that got
        # noisy for a ~1500-species first build)
        self._report_bar("cache", i, total,
                         f"encoding species embedding: {name} "
                         f"(one-time, cached on disk afterwards)")

    def _load_tol(self):
        if self._tol is None:
            from bioclip.predict import TreeOfLifeClassifier
            from bioclip import Rank
            self._Rank = Rank
            self._tol = TreeOfLifeClassifier()
        return self._tol

    def _classify_tol(self, image: str, tax) -> dict:
        """Whole-tree-of-life species ID (no regional prefilter) - shared by
        --analyze-all non-bird subjects and no-data regions. The full tree is
        one heavy model: the first call pays a ~2-4 min load, then every photo
        is species-ID'd over the whole tree (no regional shortlist), which is
        slower per photo and lands lower-confidence than a regional ladder
        (more lookalikes)."""
        tol = self._load_tol()
        rows = tol.predict(image, rank=self._Rank.SPECIES, k=3)
        if isinstance(rows, list):
            row = rows[0] if rows else {}
        else:
            row = rows
        # row keys: file_name, kingdom..genus, species (sci), common_name, score
        species = str(row.get("common_name") or row.get("species") or "unknown")
        p = float(row.get("score", 0.0))
        common, sci = normalize_species(species, tax)
        return {
            "species": common,
            "scientific": sci,
            "confidence": round(p, 4),
            "alternatives": [],
            "reasoning": f"BioCLIP-2: whole tree of life -> {common} {p*100:.0f}%",
        }

    def classify_no_prefilter(self, image, tax, region: str) -> dict:
        """Identify a photo whose resolved region has no downloaded eBird data,
        by falling back to the whole tree of life (no regional prefilter)."""
        self._report("classify", f"no eBird data for '{region}': searching "
                                 "whole tree of life (no prefilter)")
        result = self._classify_tol(str(image), tax)
        result["reasoning"] = (f"BioCLIP-2: no eBird data for region "
                               f"'{region}'; whole tree of life (no "
                               f"prefilter) -> {result['species']} "
                               f"{result['confidence']*100:.0f}%")
        return result

    def warm(self):
        """Load the heavy BioCLIP 2 weights + precompute the coarse-stage
        embeddings up front (not lazily on the first photo), reporting progress
        + elapsed so the first-run build reads as 'working', not 'crashed'."""
        t0 = time.time()
        self._report("classify", "loading BioCLIP 2 model (first load ~2-4 min, "
                                 "subsequent runs fast)...")
        backbone = self._load_coarse()
        self._report("classify", "warming species-embedding cache "
                                 "(one-time per species)...")
        self._cache.matrix_for(COARSE_LABELS, backbone,
                               on_progress=self._embed_progress)
        self._report("classify", f"BioCLIP 2 model + cache ready in "
                                 f"{time.time() - t0:.0f}s")

    def classify(self, image, ladder, tax, region, month, dt):
        image = str(image)

        # --- stage 1: coarse category (one shared image encode) ---
        self._report("classify", f"encoding {Path(image).name} (coarse pass)")
        img_feat = self._encode_image(image)
        coarse = self._score_names(img_feat, COARSE_LABELS, 1)
        cat_raw, coarse_conf = coarse[0]
        cat = str(cat_raw).replace("a photo of a ", "").replace("a photo of ", "")
        self._report("classify", f"subject category '{cat}' ({coarse_conf*100:.0f}%)")

        # --- stage 2: fine species ---
        if cat == "bird":
            return self._classify_bird(img_feat, ladder, region, month)

        # non-bird subject (landscape / person / macro ...). The bird pre-filter
        # records the coarse category cheaply and skips the full analysis, so a
        # big collection of non-bird shots is handled without the expensive
        # tree-of-life search. --analyze-all disables this and analyses the
        # subject fully (whole tree of life) as before.
        if not getattr(self.args, "analyze_all", False):
            return {
                "species": cat if cat else "unknown",
                "scientific": "",
                "confidence": round(coarse_conf, 4),
                "alternatives": [],
                "status": "non-bird",
                "reasoning": (f"BioCLIP-2: subject category '{cat}' (non-bird); "
                              f"skipped by the bird pre-filter - rerun with "
                              f"--analyze-all to identify it"),
            }

        self._report("classify", "non-bird subject + --analyze-all: searching "
                                "whole tree of life")
        result = self._classify_tol(str(image), tax)
        result["reasoning"] = (f"BioCLIP-2: subject category '{cat}' (non-bird); "
                               f"TreeOfLife species -> {result['species']} "
                               f"{result['confidence']*100:.0f}%")
        return result

    def _classify_bird(self, img_feat: np.ndarray, ladder, region: str,
                       month: int) -> dict:
        """Staged candidate ladder (see _build_ladder): the image is encoded
        once and every stage is a cached-embedding softmax over a growing
        candidate set (base ±month grace -> wider region tiers), stopping at
        the first stage whose top score clears --expand-threshold. The base
        region is always scored with the month-grace window already applied;
        the threshold only gates walking up the region tree. Per-photo cost is
        one image encode + a few cheap matrix products - independent of how
        many species the ladder eventually covers."""
        if not ladder:
            ladder = [(f"{region}", [])]
        # Skip stages that carry no candidates (a region code with no table -
        # e.g. DE has no country seed, only its REGION_POOLS pool tier has
        # candidates). Scoring an empty stage as softmax(['unknown']) would
        # yield a 100% 'unknown' that stops the ladder. Only if NO stage has
        # any candidates do we fall back to the single 'unknown' stage.
        ladder = [st for st in ladder if st[1]] or [(f"{region}", [])]
        n_stages = len(ladder)
        first_label = ladder[0][0]
        first = [s for s, _st in ladder[0][1]] or ["unknown"]
        self._report("classify", f"scoring {len(first)} eBird candidates "
                                 f"({first_label})")
        best: tuple | None = None     # (score, species, alternatives, stage_i)
        stop_i = 0
        first_p = 0.0
        for i, (label, stage) in enumerate(ladder):
            names = [s for s, _st in stage] or ["unknown"]
            ranked = self._score_names(img_feat, names, 4,
                                       on_progress=self._embed_progress)
            p, species = ranked[0][1], ranked[0][0]
            if i == 0:
                first_p = p
            if best is None or p > best[0]:
                best = (p, species, [n for n, _ in ranked[1:4]], i)
            if p >= self.args.expand_threshold or i == n_stages - 1:
                stop_i = i
                break
            nxt_label, nxt_stage = ladder[i + 1]
            nxt_n = len([s for s, _st in nxt_stage])
            self._report("classify", f"top score {p*100:.0f}% below "
                                     f"{self.args.expand_threshold:.2f} - "
                                     f"expanding to '{nxt_label}' "
                                     f"({nxt_n} candidates)")
        p, species, alts, win_i = best
        if stop_i == 0 and win_i == 0:
            reasoning = (f"BioCLIP-2: bird category; softmax over "
                         f"{len(first)} eBird candidates "
                         f"({first_label}) -> {species} {p*100:.0f}%")
        else:
            stop_label = ladder[stop_i][0]
            reasoning = (f"BioCLIP-2: bird category; top {first_p*100:.0f}% on "
                         f"the {first_label} list below "
                         f"{self.args.expand_threshold:.2f}, widened to "
                         f"{stop_label} (stage {stop_i + 1}/{n_stages}) "
                         f"-> {species} {p*100:.0f}%")
        return {
            "species": species,
            "confidence": round(p, 4),
            "alternatives": alts,
            "reasoning": reasoning,
        }

    def classify_burst(self, frames, ladder, tax, region, month, sim=0.8):
        """Identify a burst: several frames taken of (probably) the same bird.

        Opt-in burst mode (--burst). Each frame's primary image is encoded
        ONCE, and that embedding doubles as the same-bird similarity signal.
        The coarse bird/non-bird stage runs per frame first - a person
        walking through a burst is recorded as non-bird, never averaged into
        a bird posterior. Frames the coarse stage calls bird are then checked
        against the EARLIEST frame via cosine similarity of the embeddings:
        only frames >= `sim` count as the same bird (the gate that stops a
        rapid sequence which actually cut between different subjects from
        being merged). Frames passing the gate are scored with a FULL softmax
        over the SHARED candidate ladder (the union of every ladder stage, so
        every frame sees the same denominator) and the per-frame distributions
        are combined by LOGIT averaging (the normalized geometric mean of the
        per-frame softmax vectors): the per-frame log-probabilities are
        averaged and one softmax applied - a species that is top in every
        frame dominates, and N looks at the same bird ACCUMULATE evidence, so
        confidence rises with N and usually beats any single frame (unlike a
        plain arithmetic mean of the probabilities, which only smooths and
        lands below the best frame).

        Returns {str(primary.resolve()): result} for every member frame:
          - frames that passed the gate -> the SAME shared averaged result
          - frames that failed it        -> their own single-frame result (a
            different bird, or a non-bird -> the normal coarse result)
        The caller writes each frame's sidecar from this map; a frame absent
        from it was never grouped and takes the normal per-capture path.
        """
        results: dict[str, dict] = {}
        bird: list = []                     # (capture, image_feat) bird frames
        for cap in frames:
            path = str(cap.primary().resolve())
            self._report("classify", f"encoding {cap.primary().name} "
                                     f"(burst, {len(frames)} frames)")
            feat = self._encode_image(str(cap.primary()))
            coarse = self._score_names(feat, COARSE_LABELS, 1)
            cat_raw, coarse_conf = coarse[0]
            cat = str(cat_raw).replace("a photo of a ", "").replace("a photo of ", "")
            if cat == "bird":
                bird.append((cap, feat))
                continue
            # a non-bird frame inside a burst: normal single-frame handling,
            # so its sidecar stays a non-bird record, not a merged guess.
            if not getattr(self.args, "analyze_all", False):
                results[path] = {
                    "species": cat if cat else "unknown", "scientific": "",
                    "confidence": round(coarse_conf, 4),
                    "alternatives": [], "status": "non-bird",
                    "reasoning": (f"BioCLIP-2: subject category '{cat}' "
                                  f"(non-bird, burst frame); skipped by the "
                                  f"bird pre-filter"),
                }
            else:
                r = self._classify_tol(str(cap.primary()), tax)
                r["reasoning"] = (f"BioCLIP-2: subject category '{cat}' "
                                  f"(non-bird, burst frame); TreeOfLife -> "
                                  f"{r['species']} {r['confidence']*100:.0f}%")
                results[path] = r

        def single(cap, feat):
            # one frame's own fine-stage result (the burst gave no shared
            # evidence: too few bird frames, or the same-bird gate failed).
            r = self._classify_bird(feat, ladder, region, month)
            r["reasoning"] = (f"BioCLIP-2: burst frame not merged (no same-bird "
                              f"group) -> {r['species']} {r['confidence']*100:.0f}%")
            return r

        if len(bird) < 2:
            # a 'burst' of a single bird frame: no evidence to combine
            for cap, feat in bird:
                results[str(cap.primary().resolve())] = single(cap, feat)
            return results

        # same-bird gate: cosine against the earliest frame's embedding
        anchor = bird[0][1]
        kept: list = []
        rest: list = []
        for cap, feat in bird:
            (kept if _cosine(feat, anchor) >= sim else rest).append((cap, feat))
        if len(kept) < 2:
            # the gate kept nothing - these are different subjects back-to-back
            for cap, feat in bird:
                results[str(cap.primary().resolve())] = single(cap, feat)
            return results

        # shared candidate ladder: the union of every stage, so each frame's
        # distribution is over the SAME denominator (staging is a per-photo
        # efficiency trick; a burst wants one common set).
        names: list[str] = []
        seen: set = set()
        for _label, stage in ladder:
            for s, _st in stage:
                if s.lower() not in seen:
                    seen.add(s.lower())
                    names.append(s)
        if not names:
            names = ["unknown"]

        self._report("classify", f"burst: merging {len(kept)} same-bird "
                                 f"frames over {len(names)} shared candidates")
        # Evidence accumulation: average the per-frame LOG-probabilities (equal
        # to averaging the raw logits up to one per-frame constant) and take a
        # single softmax = the normalized GEOMETRIC mean of the per-frame
        # distributions. A species that is top in every frame dominates and
        # confidence rises with N - unlike a plain arithmetic mean of the
        # softmax vectors, which only smooths and lands below the best frame.
        mean_logp = None
        for cap, feat in kept:
            probs = self._probs_for(feat, names, on_progress=self._embed_progress)
            logp = np.log(np.maximum(probs, 1e-12))
            mean_logp = logp if mean_logp is None else mean_logp + logp
        mean_logp = mean_logp / len(kept)
        e = np.exp(mean_logp - mean_logp.max())
        avg = e / e.sum()
        order = np.argsort(avg)[::-1]
        top, conf = names[int(order[0])], float(avg[order[0]])
        alts = [names[int(i)] for i in order[1:4] if float(avg[i]) > 0.0]
        shared = {
            "species": top,
            "confidence": round(conf, 4),
            "alternatives": alts,
            "reasoning": (f"BioCLIP-2: burst of {len(kept)} same-bird frames "
                          f"(logit-averaged softmax over {len(names)} shared "
                          f"candidates) -> {top} {conf*100:.0f}%"),
        }
        for cap, feat in kept:
            results[str(cap.primary().resolve())] = shared
        for cap, feat in rest:
            results[str(cap.primary().resolve())] = single(cap, feat)
        return results


def make_classifier(args) -> Classifier:
    return BioclipClassifier(args)


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


# Canonical order of rdf:Description's children in a sidecar. write_sidecar
# merges into an existing file, and _set_list / dc:subject remove-and-reappend
# their elements - without this reorder, every re-run would drift the layout
# (species/confidence migrating to the end). Kept in sync with write_sidecar.
_XMP_DESC_ORDER = [
    ("bird", "species"), ("bird", "confidence"), ("bird", "summary"),
    ("bird", "topSpecies"), ("bird", "scientificName"),
    ("bird", "avibaseId"), ("bird", "avibaseUrl"), ("bird", "topConfidence"),
    ("bird", "status"), ("bird", "pipelineVersion"), ("bird", "region"),
    ("bird", "month"), ("bird", "datetime"), ("bird", "reasoning"),
    ("bird", "error"), ("dc", "description"), ("dc", "subject"),
]


def _canonicalize_desc(desc):
    """Reorder rdf:Description's children into _XMP_DESC_ORDER so a merged
    sidecar keeps the same layout as a freshly written one. Elements outside
    the canonical list keep their relative order at the end."""
    from lxml import etree
    nsmap = _nsmap()
    wanted = [f"{{{nsmap[ns]}}}{tag}" for ns, tag in _XMP_DESC_ORDER]
    rank = {tag: i for i, tag in enumerate(wanted)}
    children = list(desc)
    ordered = sorted((el for el in children if el.tag in rank),
                     key=lambda el: rank[el.tag])
    tail = [el for el in children if el.tag not in rank]
    for el in children:
        desc.remove(el)
    for el in ordered + tail:
        desc.append(el)


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


def write_sidecar(sidecar: Path, result: dict, region: str, month: int,
                  dt: datetime, status: str, cfg, error: str = None, scientific: str = "",
                  avibase_url: str = "", avibase_id: str = ""):
    """Create/merge an XMP sidecar at `sidecar`. Originals are never touched."""
    from lxml import etree
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
        if avibase_id:
            _set_text(desc, "avibaseId", BIRD_NS, avibase_id)
        if avibase_url:
            _set_text(desc, "avibaseUrl", BIRD_NS, avibase_url)
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
        if avibase_url:
            note += f" | Avibase: {avibase_url}"
        _set_text(desc, "description", _nsmap()["dc"], note)

        # dc:subject -> Birds/<species> keywords (convenience, IMatch-managed);
        # non-bird subjects get Non-bird/<category> so they stay separate.
        kw_prefix = "Non-bird" if status == "non-bird" else "Birds"
        for old in _find(root, "//dc:subject"):
            old.getparent().remove(old)
        subj = etree.SubElement(desc, f"{{{_nsmap()['dc']}}}subject")
        bag = etree.SubElement(subj, f"{{{_nsmap()['rdf']}}}Bag")
        seen = set()
        for s in ranked[:5]:
            if s not in seen:
                seen.add(s)
                li = etree.SubElement(bag, f"{{{_nsmap()['rdf']}}}li")
                li.text = f"{kw_prefix}/{s}"

    # canonical element order, regardless of whether this sidecar was merged
    # into an existing file (a re-run must not drift the layout)
    _canonicalize_desc(desc)

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


def _low_confidence_review(manifest: dict) -> list[tuple[float, str, str]]:
    """Sorted (confidence, species, folder/stem) list of low-confidence
    captures from the manifest, least-confident first. Used for the review
    summary and for the in-TUI rerun offer."""
    return sorted(
        ((float(e.get("confidence", 0.0)), e.get("species", "?"),
          Path(k).parent.name + "/" + Path(k).stem)
         for k, e in manifest.items() if e.get("status") == "low-confidence"),
        key=lambda t: t[0])


# --------------------------------------------------------------------------
# Scan & grouping (RAW+JPEG pairs -> ONE capture -> ONE sidecar)
# --------------------------------------------------------------------------

class Capture:
    """One photographic capture = one or more files sharing a base name.

    A capture writes a SINGLE sidecar named ``<base>.xmp`` (never
    ``<base>.jpg.xmp`` or ``<base>.cr2.xmp``), regardless of how many files
    (JPG + RAW) it contains.
    """

    def __init__(self, base: str, directory: Path):
        self.base = base            # original-case base name, e.g. "IMG_0001"
        self.directory = directory  # folder the capture lives in
        self.files: list[Path] = []

    def sidecar(self) -> Path:
        """The one sidecar for this capture: <folder>/<base>.xmp"""
        return self.directory / (self.base + ".xmp")

    def primary(self) -> Path:
        """Best file to classify: a viewable one, else the first."""
        for f in self.files:
            if f.suffix.lower() in VIEWABLE_EXTS:
                return f
        return self.files[0]

    def viewable(self) -> bool:
        return any(f.suffix.lower() in VIEWABLE_EXTS for f in self.files)


def _count_image_files(root: Path) -> int:
    """Quick pre-pass counting files that will be grouped (for the scan bar)."""
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALL_EXTS:
            n += 1
    return n


def scan_folder(root: Path, manifest: dict, on_progress=None) -> list[Capture]:
    """Group files by (relative folder, lowercased base name).

    on_progress(done, total) is called once per grouped file (when given) so a
    run can show a 'walking folder tree' step bar - a quick count pass supplies
    the total before the grouping walk reports.
    """
    total = _count_image_files(root) if on_progress else 0
    done = 0
    captures: dict[tuple[str, str], Capture] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in ALL_EXTS:
            rel_dir = str(p.parent.relative_to(root))
            key = (rel_dir, p.stem.lower())
            if key not in captures:
                captures[key] = Capture(p.stem, p.parent)
            captures[key].files.append(p)
            if on_progress:
                done += 1
                on_progress(done, total)

    result = []
    for cap in captures.values():
        cap.files.sort(key=lambda f: (f.suffix.lower() not in VIEWABLE_EXTS, str(f)))
        result.append(cap)
    return result


def sidecar_status(sidecar: Path):
    """Return 'done' / 'low-confidence' / 'non-bird' / 'no-data' / 'error' /
    None by parsing the sidecar. 'low-confidence' is kept distinct from 'done'
    so a re-run can revisit (or skip) uncertain captures (see
    --include-low-confidence); 'no-data' marks a capture whose region had no
    eBird data (best guess blank)."""
    if not sidecar.exists():
        return None
    try:
        root = _xmp_tree(sidecar)
        st = _find(root, "//bird:status")
        status = st[0].text.strip() if st and st[0].text else None
        if status == "error":
            return "error"
        if status == "non-bird":
            return "non-bird"
        if status == "low-confidence":
            return "low-confidence"
        if status == "no-data":
            return "no-data"
        if sidecar_done(sidecar):
            return "done"
    except Exception:
        pass
    return None


def classify_capture(cap: Capture, force: bool = False) -> str:
    """done / low-confidence / non-bird / no-data / error / pending - a capture
    is done iff its single sidecar is done. 'no-data' captures are re-analysed
    on re-run (the region may have been downloaded since). With ``force=True``
    every capture is 'pending': an existing sidecar is re-identified in place
    (only its bird-ID fields are overwritten, nothing is deleted)."""
    if force:
        return "pending"
    return sidecar_status(cap.sidecar()) or "pending"


def _group_bursts(pending: list, meta_map: dict, gap: float):
    """Group captures into bursts by EXIF capture time - NEVER mtime.

    The Canon 5D Mark IV (~7 fps) and R5 (12-20 fps) fire burst frames tens-to-
    hundreds of ms apart, so within the SAME folder any run of consecutive
    captures whose EXIF DateTimeOriginal/CreateDate are <= `gap` seconds apart
    forms one burst; a shot that came later than that starts a new one. The
    time comes from `meta_map['datetime']` (the real shutter time read from
    EXIF) - not the file mtime or the 'pulled from the camera card' timestamp
    a properties GUI shows - so grouping is by when the shutter actually fired.
    Captures with no EXIF time (or whose group never reaches two members) stay
    singletons and take the normal per-capture path.

    Returns (groups, member_of): groups is a list of time-ordered
    [Capture, ...] each of length >= 2; member_of maps
    str(cap.primary().resolve()) -> index into groups.
    """
    by_dir: dict[Path, list] = {}
    for cap in pending:
        meta = meta_map.get(str(cap.primary().resolve()), {})
        dt = meta.get("datetime")
        if dt is None:                       # no EXIF create time -> never burst
            continue
        by_dir.setdefault(cap.directory, []).append((cap, dt))

    groups: list = []
    member_of: dict[str, int] = {}
    for caps in by_dir.values():
        caps.sort(key=lambda t: t[1])        # real capture time order
        cur: list = []
        for cap, dt in caps:
            if cur and (dt - cur[-1][1]).total_seconds() > gap:
                _seal_burst(cur, groups, member_of)
                cur = []
            cur.append((cap, dt))
        _seal_burst(cur, groups, member_of)
    return groups, member_of


def _seal_burst(cur: list, groups: list, member_of: dict):
    """Close a time-run: keep it as a burst only when it has >= 2 frames."""
    if len(cur) >= 2:
        idx = len(groups)
        group = [c for c, _d in cur]
        groups.append(group)
        for c in group:
            member_of[str(c.primary().resolve())] = idx


def _sidecar_best(sidecar: Path) -> tuple[str, float] | None:
    """Existing sidecar's (top species, top confidence), or None if the sidecar
    is missing or has no topConfidence element. A no-data sidecar reads back as
    ('unknown', 0.0); the keep-best guard additionally requires a 'done' /
    'low-confidence' status, so such records are not protected."""
    if not sidecar.exists():
        return None
    try:
        root = _xmp_tree(sidecar)
        conf = _find(root, "//bird:topConfidence")
        if not conf or not conf[0].text:
            return None
        c = float(conf[0].text)
        sp = _find(root, "//bird:topSpecies")
        name = sp[0].text.strip() if sp and sp[0].text else "?"
        return name, c
    except (TypeError, ValueError):
        return None

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
        self.stage_start = time.time()   # reset whenever `message` changes
        self.stage_elapsed = 0.0
        self.stage_done = 0              # per-step bar (e.g. EXIF / cache / status)
        self.stage_total = 0             # 0 = no step bar -> the classify bar is shown


class TUI:
    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdout.isatty()
        self.state = State()
        self.stop_flag = threading.Event()
        self._win = None
        self._io_lock = threading.Lock()   # guards the curses screen

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
            st.stage_elapsed = time.time() - st.stage_start
            if st.processed and st.total:
                per = st.elapsed / st.processed
                left = per * (st.total - st.processed)
                st.eta = f"ETA {int(left//60):02d}:{int(left%60):02d}"
            if not self.enabled:
                return
        import curses
        with self._io_lock:
            win = self._win
            if win is None:
                return
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
            if st.message:
                mm, ss = divmod(int(st.stage_elapsed), 60)
                put(f"  working on this step for {mm:02d}:{ss:02d} - not stuck")
            put("")
            put(f"scanned {st.scanned}   pending {st.pending}   done {st.done}   error {st.error}")
            put(f"progress: {st.processed}/{st.total}    elapsed {int(st.elapsed//60):02d}:{int(st.elapsed%60):02d}   {st.eta}")
            if st.stage_total:
                # a per-step bar (EXIF / cache build / status scan / ...) takes
                # over the bar line while it is running
                frac = min(1.0, st.stage_done / st.stage_total)
                width = max(10, w - 20)
                filled = int(width * frac)
                put(f"[{'#' * filled}{'.' * (width - filled)}] {frac*100:4.0f}%")
            elif st.total:
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
            if "stage" in kw and kw["stage"] != st.stage:
                st.stage_done = 0
                st.stage_total = 0
            if "message" in kw and kw["message"] != st.message:
                st.stage_start = time.time()
            for k, v in kw.items():
                setattr(st, k, v)

    def progress(self, stage: str, done: int, total: int, message: str = ""):
        """Per-step progress: fullscreen bar in TUI mode, an inline
        '\\r[label] done/total pct%' line otherwise. When the step finishes the
        inline line is closed with a newline so later output starts fresh."""
        if self.enabled:
            self.set(stage=stage, stage_done=done, stage_total=total,
                     message=message or "")
            return
        label = stage or "step"
        pct = f"{100.0 * done / total:3.0f}%" if total else "?"
        print(f"\r[{label}] {done}/{total} {pct}   ", end="", flush=True)
        if total and done >= total:
            print(flush=True)   # finish the line once the step is complete

    def run_loop(self):
        if not self.enabled:
            return
        import curses
        while not self.stop_flag.is_set():
            with self._io_lock:
                win = self._win
                if win is None:
                    # a prompt owns the screen right now - wait for it to finish
                    time.sleep(0.05)
                    continue
                win.timeout(150)
                key = win.getch()
            if key in (ord("q"), ord("Q")):
                self.stop_flag.set()
                break
            self.refresh()
        # NB: do NOT call self.stop() here - main() tears down curses in its
        # finally block (which also joins this thread first).

    def plain_line(self, text):
        if not self.enabled:
            print(text, flush=True)

    def prompt(self, text: str, default: str = "") -> str:
        """Ask the user a one-shot question during a run and return the trimmed
        answer ('' = accept the default).

        Non-TUI: plain print + input(). TUI: briefly suspends the fullscreen
        screen, asks on the terminal, then restores it. The caller validates the
        answer and re-prompts as needed; EOF/interrupt returns the default.
        """
        if not self.enabled:
            print(text, flush=True)
            try:
                return input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                return default
        import curses
        with self._io_lock:
            if self._win is not None:
                curses.nocbreak()
                self._win.keypad(False)
                curses.echo()
                curses.endwin()
                self._win = None
        try:
            print(text, flush=True)
            try:
                ans = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                ans = default
            return ans
        finally:
            with self._io_lock:
                self._win = curses.initscr()
                curses.noecho()
                curses.cbreak()
                self._win.keypad(True)
                try:
                    curses.curs_set(0)
                except Exception:
                    pass
            self.refresh()


def stage_hint(stage: str) -> str:
    hints = {
        "scan": "walking folder tree (grouping captures)",
        "status": "checking existing sidecars",
        "cache": "building species-embedding cache (one-time per species)",
        "exif": "reading EXIF (GPS + timestamp, batched)",
        "region": "resolving GPS / folder date -> region + month",
        "candidates": "building candidate shortlist (eBird + checklist)",
        "classify": "classifying photo (local model backend)",
        "parse": "normalising species name",
        "sidecar": "writing XMP sidecar",
        "": "",
    }
    return hints.get(stage, "")


def fmt_result(result: dict, scientific: str = "", avibase_url: str = "") -> str:
    if not result:
        return "no result"
    if result.get("status") == "non-bird":
        return f"{result.get('species', '?')} (non-bird - skipped)"
    if result.get("status") == "no-data":
        return "no region data (flagged no-data)"
    c = float(result.get("confidence", 0.0))
    name = result.get("species", "?")
    base = f"{name} ({scientific}) {c*100:.0f}%" if scientific else f"{name} ({c*100:.0f}%)"
    if avibase_url:
        base += f" | avibase: {avibase_url}"
    return base


def ask_region_override(tui, count: int, default: str,
                        valid: list[str]) -> str | None:
    """Ask (TUI) once for a region to use for `count` GPS-less photos.

    Returns an upper-cased region code, or None to keep the default. Non-TUI
    (batch/--no-tui) never prompts - --region stays the only mechanism there.
    """
    if not tui.enabled:
        return None
    while True:
        ans = tui.prompt(
            f"{count} photo(s) have no GPS location. Which country/region "
            f"were they taken in? (enter a code, or blank for the default "
            f"'{default}'; available: {', '.join(valid)})")
        if not ans:
            return None
        ans = ans.upper()
        if ans in valid:
            return ans
        tui.plain_line(f"  unknown region code '{ans}' - choose from: "
                       f"{', '.join(valid)}")


def ask_no_data_choice(tui, region: str) -> str:
    """Decide how to handle photos in a region with no downloaded eBird data.

    'tol' = analyze WITHOUT the regional prefilter (whole tree of life) - the
    user explicitly opts in after a cost/accuracy disclaimer. 'flag' = skip and
    write status 'no-data' (best guess blank, surfaces in the review list).
    Non-TUI always returns 'flag': the expensive no-prefilter path is never run
    silently.
    """
    if not tui.enabled:
        return "flag"
    tui.plain_line(f"  note: no eBird data for region '{region}' "
                   f"(ladder is empty - no downloaded table)")
    while True:
        ans = tui.prompt(
            f"No eBird data for region '{region}'.\n"
            f"  1) analyze WITHOUT the regional prefilter (whole tree of life)\n"
            f"     what 'significantly longer' means: one extra model load of "
            f"~2-4 min up front, then EVERY photo is species-ID'd over the full "
            f"tree (no regional shortlist) - slower per photo AND lower "
            f"accuracy / more lookalikes than a regional ladder. (The "
            f"all-taxonomy pool of ~18,600 species would be slower still: a "
            f"one-time ~10-14 h embedding precompute - not used.)\n"
            f"  2) skip + flag as 'no-data' (recommended default - surfaces in "
            f"the review list, best guess left blank)\n"
            f"  To avoid this, download the region once and rerun:\n"
            f"     birdid --download {region}\n"
            f"  (curated registry ships SG/TH/DE + the SEA/EUROPE pools; any "
            f"other eBird country/continent needs an API key: "
            f"birdid --api <key>)\n"
            f"  [1/2/Enter=2]: ")
        if ans in ("1", "2", "y", "n", ""):
            break
        tui.plain_line("  enter 1 (analyze) or 2 (flag).")
    return "tol" if ans in ("1", "y") else "flag"

# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Offline bird-photo identification (BioCLIP 2, single XMP sidecar per capture).")
    ap.add_argument("folder", nargs="?", default=None, help="folder (recursive) of photos to scan")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="dir containing eBird data (regions/, regions_generated/, "
                         "taxonomy). Default: <repo>/ebird")
    ap.add_argument("--cache-dir", default=str(_default_cache_dir()),
                    help="central per-user BioCLIP 2 species-embedding cache "
                         "(embeddings.npy + index.json). Kept out of the photo "
                         "folders and the eBird data dir so no derived files "
                         "land in an IMatch-managed tree. "
                         f"Default: {_default_cache_dir()}")
    ap.add_argument("--region", default=DEFAULT_REGION,
                    help="default region when no GPS can be resolved; DE/SG/TH "
                         "have no country seed and fall back to their EUROPE/SEA "
                         "pool. In TUI mode a GPS-less run asks once for the "
                         "region instead of silently using this default")
    ap.add_argument("--no-checklist", action="store_true",
                    help="ignore optional eBird *_observations.csv exports "
                         "(both in photo folders and the centralized checklists dir)")
    ap.add_argument("--checklists-dir", default=DEFAULT_CHECKLISTS_DIR,
                    help="centralized eBird checklist exports (one file per "
                         "checklist, e.g. S296663962_observations.csv); "
                         "entries are keyed by Observation Date and boost the "
                         "candidate list for matching folders. Default: "
                         "~/.config/birdid/checklists (or $BIRDID_CHECKLISTS_DIR)")
    ap.add_argument("--manifest", default=None, help="manifest path (default <folder>/birdid_manifest.json)")
    ap.add_argument("--report", default=None, help="additionally write a flat CSV aggregate of the current run to this path (per-folder reports are always written as <folder>/birdid_report.csv)")
    ap.add_argument("--max", type=int, default=0, help="stop after this many captures (0 = all)")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="below this top score a capture is flagged "
                         "'low-confidence' (default 0.5): a sidecar is still "
                         "written with the best guess, but the photo is listed "
                         "for review and skipped on re-runs unless "
                         "--include-low-confidence is given. Lower this (e.g. "
                         "0.35) to accept more marginal shots as 'done'.")
    ap.add_argument("--include-low-confidence", action="store_true",
                    help="re-process captures whose sidecar is flagged "
                         "'low-confidence' (top score below --min-confidence). "
                         "By default such photos are left untouched and only "
                         "listed for review; this flag puts them back in the "
                         "processing queue.")
    ap.add_argument("--force", action="store_true",
                    help="force a fresh identification run for a folder that "
                         "already has sidecars: every capture is re-processed "
                         "and ONLY the bird-ID fields in the existing XMP are "
                         "overwritten (bird:* + dc:subject/dc:description). "
                         "Everything else in the sidecar - e.g. metadata IMatch "
                         "has written - is preserved, and no file is deleted, "
                         "so a forced re-run is safe on an IMatch-managed "
                         "folder. Also overrides the keep-best guard (the new "
                         "result is written even if it scores lower).")
    ap.add_argument("--expand-threshold", type=float, default=0.35,
                    help="below this top score, re-score against the wider "
                         "regional pool (neighbouring countries) - 0 disables "
                         "the expansion (default 0.35)")
    ap.add_argument("--burst", action="store_true",
                    help="burst-aware identification (OPT-IN): a rapid series "
                         "of shots of the same bird is combined into one "
                         "stronger ID. Captures in the same folder whose EXIF "
                         "capture time (DateTimeOriginal/CreateDate - never the "
                         "file mtime or the 'pulled from the card' timestamp) "
                         "falls within --burst-gap are grouped, then only frames "
                         "the BioCLIP embedding similarity confirms as the SAME "
                         "bird (cosine >= --burst-sim) are merged by logit-"
                         "averaging (geometric mean over the shared candidate "
                         "ladder, so N looks accumulate evidence) and the shared "
                         "result is written to EVERY frame's sidecar. Frames "
                         "that fail the similarity gate keep their own single-"
                         "frame ID. Without this flag behaviour is exactly the "
                         "per-capture mode.")
    ap.add_argument("--burst-gap", type=float, default=3.0,
                    help="max EXIF capture-time gap (seconds) between frames of "
                         "one burst (default 3.0). Tuned for the Canon 5D Mark "
                         "IV (~7 fps) and Canon R5 (12-20 fps): burst frames are "
                         "tens-to-hundreds of ms apart, so a few seconds "
                         "captures a full burst while a new shot 30 s later "
                         "starts a new one. Requires --burst.")
    ap.add_argument("--burst-sim", type=float, default=0.8,
                    help="minimum cosine similarity between a frame's BioCLIP "
                         "embedding and the burst's earliest frame for it to "
                         "count as the SAME bird and join the merged result "
                         "(default 0.8). Lower = merges more; higher = only "
                         "near-identical frames merge. Requires --burst.")
    ap.add_argument("--month-grace", type=int, default=1,
                    help="±N months around the photo's month a species still "
                         "counts as a candidate - birds rarely carry a "
                         "calendar (default 1; 0 disables the grace window)")
    ap.add_argument("--analyze-all", action="store_true",
                    help="disable the bird pre-filter: also fully analyse "
                         "non-bird subjects (landscapes, people, macro - via "
                         "the whole tree of life) instead of recording them as "
                         "'non-bird' and skipping them. Previously skipped "
                         "non-bird photos are re-analysed on rerun.")
    ap.add_argument("--dry-run", action="store_true", help="scan + plan only, no model calls")
    ap.add_argument("--no-tui", action="store_true", help="plain log output, no fullscreen UI")
    ap.add_argument("--list-candidates", action="store_true",
                    help="print candidate counts per region/month and exit")

    # --- data management commands (no photo folder needed) ---
    ap.add_argument("-d", "--download", nargs="+", metavar="SPEC",
                    help="write real eBird region tables to the gitignored "
                         "regions_generated/ tree (preferred over the curated "
                         "seeds; never committed). The smallest division is "
                         "always written: <CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv "
                         "(or <COUNTRY>/<SUB>.csv with --subnational). SPEC is "
                         "a country code (e.g. SG/DE), or 'country CODE'; a "
                         "whole continent/subcontinent downloads every member "
                         "country, creating subregion folders as needed "
                         "(e.g. region AS, region AS-SOUTHEAST, region SEA, "
                         "region EUROPE; see --list region). Bare 'region' "
                         "resolves to the default country's SUBcontinent "
                         "(DE -> EU-WEST); 'region continent' to its continent "
                         "(DE -> EUROPE); 'region subcontinent CODE' / "
                         "'region continent CODE' for a specific country. A "
                         "pooled <REGION>.csv is kept alongside. INCREMENTAL: "
                         "parts already downloaded are skipped, only the "
                         "missing data is fetched. 'taxonomy' (keyless) fetches "
                         "the eBird/Clements taxonomy. Region tables need an "
                         "API key (--api); without one only the curated seeds "
                         "(EUROPE/SEA) apply.")
    ap.add_argument("-u", "--update", nargs="+", metavar="SPEC",
                    help="like --download but refreshes only the tables whose "
                         "eBird data is newer than what is stored - each "
                         "generated CSV records a '# data_date:' (the newest "
                         "observation date sampled: the date of the eBird data "
                         "itself, NOT the download date) that --update compares "
                         "against the newest data the API serves (default "
                         "target: the current year, so 2025-sampled tables get "
                         "2026 data; months not yet complete in the target "
                         "year fall back to the previous year so no month is "
                         "left empty). Missing tables are fetched too. Pass "
                         "--year to pin the target year.")
    ap.add_argument("--list", nargs="*", metavar="KIND",
                    help="list countries/regions available for download "
                         "(optional filter: country | region; or 'downloadable' "
                         "to show every eBird country via the API key, "
                         "'downloadable subnational1 <CODE>' for a country's "
                         "subregions)")
    ap.add_argument("--api", nargs="?", const="__show__", metavar="KEY",
                    help="store your eBird API key (--api <key>); bare --api shows status")
    ap.add_argument("--initialize", action="store_true",
                    help="one-time setup: download the eBird taxonomy + BioCLIP 2/"
                         "TreeOfLife backends so everything works offline")
    ap.add_argument("--download-backends", action="store_true",
                    help="predownload the BioCLIP 2 + TreeOfLife backends so a "
                         "later run works fully offline (same backend part as "
                         "--initialize)")
    ap.add_argument("--api-delay", type=float, default=1.5,
                    help="seconds between eBird API calls (rate-limit guard, default 1.5)")
    ap.add_argument("--sample-dates", type=int, default=2,
                    help="sample dates per month for --download (default 2)")
    ap.add_argument("--year", type=int, default=None,
                    help="historic-data year to sample: default 2025 for "
                         "--download (a complete year), the current year for "
                         "--update (the newest data eBird serves)")
    ap.add_argument("--subnational", action="store_true",
                    help="with --download country <CODE>: also fetch that "
                         "country's subnational1 regions and write them as "
                         "<COUNTRY>/<SUB>.csv (smallest division) instead of a "
                         "single country table")
    ap.add_argument("--cache", nargs="+", metavar="SUB [REGION...]",
                    help="manage the species-embedding cache (no photo folder, "
                         "no API key needed). 'build [REGION...]' precomputes "
                         "embeddings for a region's candidate species so a "
                         "later run encodes nothing on the fly (default: all "
                         "loaded regions). 'clean [REGION...]' gets rid of "
                         "data: with regions it removes those regions' species "
                         "from the cache and keeps everything else; with no "
                         "region it drops the stale species no loaded table "
                         "can use (default: all loaded regions) - no model "
                         "needed. 'update [REGION...]' syncs the cache with "
                         "the current eBird tables: drop species no longer in "
                         "any table and encode newly-needed ones (incremental). "
                         "REGION codes come from the eBird tables, e.g. DE, "
                         "EU-WEST, AS-SOUTHEAST, EUROPE")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    # --- data management commands (no photo folder needed) ---
    mgmt = (args.initialize or args.download_backends or args.api is not None
            or args.list is not None or args.download or args.update
            or args.cache is not None)
    if mgmt:
        try:
            if args.initialize or args.download_backends:
                from ebird_data import initialize
                initialize(data_dir, data_dir / "regions")
                Path(args.checklists_dir).mkdir(parents=True, exist_ok=True)
                print(f"birdid: central checklists dir: {args.checklists_dir} "
                      "(drop eBird exports like S296663962_observations.csv there)")
            elif args.api is not None:
                from ebird_data import handle_api
                handle_api(args.api)
            elif args.list is not None:
                from ebird_data import list_regions
                list_regions(data_dir, args.list)
            elif args.cache is not None:
                run_cache(data_dir, Path(args.cache_dir), args.cache)
            elif args.update:
                from ebird_data import set_api_delay, update
                set_api_delay(args.api_delay)
                update(data_dir, args.update, year=args.year,
                       sample_dates=args.sample_dates,
                       subnational=args.subnational)
            elif args.download:
                from ebird_data import set_api_delay, download
                set_api_delay(args.api_delay)
                download(data_dir, args.download, year=args.year,
                         sample_dates=args.sample_dates,
                         subnational=args.subnational)
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            sys.exit(f"birdid: ERROR: {e}")
        return

    if not args.folder and not args.list_candidates:
        ap.error("the following arguments are required: folder  "
                 "(or use a management command: --download/--update/--list/"
                 "--api/--initialize/--cache/--list-candidates)")

    root = None
    if args.folder:
        root = Path(args.folder).expanduser().resolve()
        if not root.is_dir():
            sys.exit(f"folder not found: {root}")

    data_dir = Path(args.data_dir)
    regions_dir = data_dir / "regions"
    taxonomy_path = find_taxonomy(data_dir)

    db = SpeciesDB(regions_dir, data_dir / "regions_generated")
    db.load()
    taxonomy = load_taxonomy(taxonomy_path)
    tax_src = taxonomy_path.name if taxonomy_path else "none"
    print(f"eBird tables: {', '.join(db.regions())}   taxonomy: "
          f"{tax_src} ({len(taxonomy)} names)")
    gen = [r for r in db.regions() if db.source(r) == "generated"]
    if gen:
        print(f"generated (API-derived) tables in use: {', '.join(gen)} "
              "(gitignored - preferred over the curated seeds)")

    avibase_sci, avibase_common = load_avibase(data_dir)
    if avibase_sci:
        print(f"Avibase: {len(avibase_sci)} species ids loaded "
              f"({AVIBASE_URL}<id>)")

    if args.list_candidates:
        for reg in db.regions():
            for m in range(1, 13):
                n = len(db.candidates(reg, m))
                print(f"{reg} {datetime(2020, m, 1):%b}: {n} candidates")
        return

    manifest_path = Path(args.manifest) if args.manifest else root / "birdid_manifest.json"
    manifest = load_manifest(manifest_path)

    tui = TUI(enabled=not args.no_tui)
    tui.start()
    tui_thread = threading.Thread(target=tui.run_loop, daemon=True)
    tui_thread.start()
    summary_lines: list[str] = []
    fatal = False
    lc_new = 0
    nd_n = 0
    lc_skipped = []
    kept_n = 0
    kept_lines: list[str] = []
    lc_review: list[tuple[float, str, str]] = []
    try:
        # --- Phase 1: scan & classify ---
        tui.set(phase="scanning", stage="scan", message="walking folder tree")
        captures = scan_folder(root, manifest,
                               on_progress=lambda done, total: tui.progress(
                                   "scan", done, total))
        tui.set(scanned=len(captures))

        pending = []
        done_n = 0
        nb_n = 0
        error_n = 0
        for i, cap in enumerate(captures, 1):
            tui.progress("status", i, len(captures),
                         "checking existing sidecars")
            status = classify_capture(cap, args.force)
            if status == "pending":
                pending.append(cap)
            elif status == "done":
                done_n += 1
            elif status == "non-bird":
                if args.analyze_all:
                    # --analyze-all: re-analyse photos the pre-filter skipped
                    pending.append(cap)
                else:
                    nb_n += 1
            elif status == "low-confidence":
                if args.include_low_confidence:
                    # review pass: re-analyse the flagged captures
                    pending.append(cap)
                else:
                    lc_skipped.append(cap)
            elif status == "no-data":
                # flagged for review last run because the region had no eBird
                # data. Re-analyse by default: a re-run after --download picks
                # them up, and the per-region prompt lets the user choose
                # whole-tree-of-life vs flag again.
                pending.append(cap)
            else:
                error_n += 1

        tui.set(phase="ready", pending=len(pending), done=done_n, error=error_n, total=len(pending),
                message=f"{len(pending)} to process")

        if args.dry_run:
            summary_lines.append(
                f"scan: {len(captures)} captures | {done_n} done | {nb_n} non-bird | "
                f"{error_n} error | {len(pending)} pending")
            if args.force:
                summary_lines.append(
                    "forced run: all captures would be re-identified in place "
                    "(sidecars kept, only bird-ID fields overwritten).")
            for cap in pending[:20]:
                summary_lines.append(f"  [pending] {cap.primary()} -> {cap.sidecar()}")
            if len(pending) > 20:
                summary_lines.append(f"  ... and {len(pending) - 20} more")
            summary_lines.append("dry-run: nothing processed.")
        elif not pending:
            summary_lines.append(
                f"scan: {len(captures)} captures | {done_n} done | {nb_n} non-bird | "
                f"{error_n} error | 0 pending - nothing to do.")

        if not summary_lines:
            # --- Phase 1.5: one batched EXIF read for ALL photos, then GPS means ---
            tui.set(stage="exif", message=f"reading EXIF for {len(captures)} captures")
            all_files = [f for cap in captures for f in cap.files]
            meta_map = read_exif_batch(
                all_files,
                on_progress=lambda done, total: tui.progress(
                    "exif", done, total, f"reading EXIF for {len(captures)} captures"))
            dir_gps = compute_dir_gps(meta_map)

            # --- Phase 1.6: no-GPS region override (one question per run) ---
            # Photos with no usable GPS resolve to --region (default DE). In TUI
            # mode, ask once for the country/region they were taken in instead
            # of silently assuming the default. --no-tui keeps --region as the
            # non-interactive form.
            region_override: str | None = None
            if pending and not args.dry_run:
                no_gps = sum(
                    1 for cap in pending
                    if resolve_location(
                        cap.primary(),
                        meta_map.get(str(cap.primary().resolve()), {}),
                        dir_gps)[2] == "none")
                if no_gps:
                    valid = sorted(set(db.regions()) | set(REGION_BBOXES)
                                   | set(REGION_POOLS))
                    region_override = ask_region_override(tui, no_gps,
                                                          args.region, valid)
                    if region_override:
                        tui.plain_line(f"using region override '{region_override}' "
                                       f"for GPS-less photos.")

            # --- Phase 2: classifier backend ---
            # Explicit, timed model load BEFORE the loop - the BioCLIP 2 build
            # can take ~2-4 min on first run, and doing it inside the first
            # classify() used to leave the TUI stuck on "classifying <name>"
            # (looking crashed). warm() reports progress + elapsed instead.
            tui.set(stage="classify",
                    message="loading BioCLIP 2 model (first load ~2-4 min, "
                            "subsequent runs fast)...")
            classifier = make_classifier(args)
            def on_progress(stage, msg):
                tui.set(stage=stage, message=msg)
                if not tui.enabled:      # plain log output: echo the step too
                    print(f"  [{stage}] {msg}", flush=True)
            def on_progress_bar(stage, done, total, msg=""):
                tui.progress(stage, done, total, msg)
            classifier.progress = on_progress
            classifier.progress_bar = on_progress_bar
            classifier.warm()
            tui.plain_line(f"classifier: {classifier.describe()}")

            report_rows = []
            stop = tui.stop_flag
            checklist_cache: dict[str, list[tuple[str, str]]] = {}
            # per-region decision for photos whose region has no eBird data
            # ('tol' = whole-tree-of-life, 'flag' = skip + status 'no-data')
            no_data_choice_cache: dict[str, str] = {}
            central = (CentralChecklists(Path(args.checklists_dir), taxonomy)
                       if not args.no_checklist else CentralChecklists(None, taxonomy))
            tui.plain_line(central.report())

            def capture_checklist(cap: Capture) -> list[tuple[str, str]]:
                if args.no_checklist:
                    return []
                cl_path = find_checklist(cap.directory, root)
                if cl_path is None:
                    return []
                key = str(cl_path)
                if key not in checklist_cache:
                    checklist_cache[key] = [
                        (normalize_species(s, taxonomy)[0], "checklist")
                        for s in load_checklist(cl_path) if s.strip()]
                return checklist_cache[key]

            # --- Phase 2.5: burst grouping (opt-in --burst) ---
            # Group pending captures into bursts by EXIF capture time (never
            # mtime / folder name). Classification is LAZY: when the loop below
            # reaches the first member of a group it classifies the WHOLE burst
            # once (classify_burst) and reuses the shared result for every
            # member, so the per-capture write path stays exactly as-is and the
            # default (no --burst) is untouched.
            burst_groups: list = []
            burst_member_of: dict = {}
            if args.burst:
                burst_groups, burst_member_of = _group_bursts(
                    pending, meta_map, args.burst_gap)
                if burst_groups:
                    n_burst = sum(len(g) for g in burst_groups)
                    tui.plain_line(
                        f"burst mode: {len(burst_groups)} burst(s) over "
                        f"{n_burst} captures (EXIF gap <= {args.burst_gap:g}s, "
                        f"same-bird cosine >= {args.burst_sim:.2f})")
            burst_results: dict = {}      # str(primary.resolve()) -> shared result
            burst_done: set = set()       # group indices already classified

            # --- Phase 3: process ---
            tui.set(phase="running")
            for i, cap in enumerate(pending, 1):
                if stop.is_set() or (args.max and i > args.max):
                    break
                cap_files = cap.files
                primary = cap.primary()
                sidecar = cap.sidecar()
                rel = primary.relative_to(root)
                tui.set(current=str(rel), processed=i - 1)
                last = ""
                try:
                    # location: embedded GPS -> folder-mean GPS -> --region
                    tui.set(stage="region", message="resolving region/month")
                    meta = meta_map.get(str(primary.resolve()), {})
                    lat, lon, gps_src = resolve_location(primary, meta, dir_gps)
                    if region_override is not None and gps_src == "none":
                        # user supplied a region for the GPS-less photos this run
                        region = region_override
                        region_src = "prompt"
                    else:
                        region = region_for(lat, lon, args.region)
                        region_src = "gps" if gps_src != "none" else "default"
                    dt = meta.get("datetime") or folder_date(primary)
                    month = dt.month if dt else datetime.now().month
                    date_src = "exif" if meta.get("datetime") else ("folder" if dt else "today")

                    tui.set(stage="candidates", message=f"{region} month {month:02d}")
                    checklist = capture_checklist(cap)
                    # central checklists: species for this capture's (year, month)
                    central_list: list[tuple[str, str]] = []
                    if dt:
                        central_list = central.species_for(dt.year, dt.month)
                    # region-table species (for the "region too narrow" flag)
                    region_names = {s.lower() for s, _ in
                                    db.candidates(region, month, args.month_grace)}
                    central_outside = sorted(
                        {c[0] for c in central_list
                         if c[0].lower() not in region_names})
                    central_added = sorted({c[0] for c in central_list})
                    # Staged ladder: strict (region, month) first, then the
                    # ±month grace, then each wider region tier again without
                    # then with the grace - so a confident photo is scored
                    # against a small list and only low-confidence shots widen
                    # the pool (see _build_ladder).
                    ladder = _build_ladder(db, region, month, args.month_grace,
                                           checklist, central_list)

                    if central_outside:
                        tui.plain_line(
                            f"  note: {len(central_outside)} central-checklist "
                            f"species outside {region} table: "
                            f"{', '.join(central_outside)} - region prior may be too narrow")

                    # --- no-data handling: an empty ladder (no eBird table at
                    # all, not even a wider pool tier) used to score 'unknown'
                    # silently. Ask once per region (TUI) whether to analyze
                    # WITHOUT the regional prefilter (whole tree of life - with
                    # the cost/accuracy disclaimer) or skip + flag 'no-data'.
                    if sum(len(st) for _, st in ladder) == 0:
                        if region not in no_data_choice_cache:
                            no_data_choice_cache[region] = ask_no_data_choice(tui, region)
                        if no_data_choice_cache[region] == "tol":
                            tui.set(stage="classify",
                                    message=f"classifying {primary.name} "
                                            f"(no prefilter)")
                            result = classifier.classify_no_prefilter(
                                primary, taxonomy, region)
                        else:
                            result = {
                                "species": "unknown",
                                "scientific": "",
                                "confidence": 0.0,
                                "alternatives": [],
                                "status": "no-data",
                                "reasoning": (f"no eBird region data for "
                                              f"'{region}' - flagged for "
                                              f"review; download it with "
                                              f"'birdid --download {region}' "
                                              f"(needs an API key) or rerun in "
                                              f"TUI mode to analyze without "
                                              f"the prefilter"),
                            }
                    else:
                        tui.set(stage="classify", message=f"classifying {primary.name}")
                        key = str(primary.resolve())
                        if key in burst_member_of:
                            # this capture belongs to a burst: classify the WHOLE
                            # burst once (the first member reached triggers it)
                            # and reuse the shared result for every member. A
                            # member that failed the same-bird gate is absent
                            # from burst_results and falls back to its own ID.
                            bix = burst_member_of[key]
                            if bix not in burst_done:
                                burst_done.add(bix)
                                tui.set(stage="classify",
                                        message=f"burst {len(burst_groups[bix])} "
                                                f"frames: {primary.name}")
                                burst_results.update(classifier.classify_burst(
                                    burst_groups[bix], ladder, taxonomy, region,
                                    month, sim=args.burst_sim))
                            if key in burst_results:
                                result = burst_results[key]
                            else:
                                result = classifier.classify(primary, ladder,
                                                             taxonomy, region,
                                                             month, dt)
                        else:
                            result = classifier.classify(primary, ladder, taxonomy,
                                                         region, month, dt)

                    tui.set(stage="parse", message="normalising species")
                    if not isinstance(result, dict) or not result.get("species"):
                        raise RuntimeError(f"bad classifier output: {result!r}")
                    species, scientific = normalize_species(str(result["species"]), taxonomy)
                    result["species"] = species
                    if not scientific and result.get("scientific"):
                        scientific = result["scientific"]
                    if scientific:
                        result["scientific"] = scientific
                    avibase_url = avibase_url_for(avibase_sci, avibase_common,
                                                  scientific, species)
                    avibase_id = avibase_url[len(AVIBASE_URL):] if avibase_url else ""
                    confidence = float(result.get("confidence", 0.0))
                    status = result.get("status") or (
                        "done" if confidence >= args.min_confidence else "low-confidence")
                    if (species not in ("unknown", "") and scientific == ""
                            and taxonomy and status != "non-bird"):
                        tui.plain_line(f"  note: '{species}' not in eBird taxonomy - check spelling")
                    # keep-best guard: a re-analysis (review pass, --analyze-all,
                    # a no-data retry) must never REGRESS a recorded result. The
                    # candidate ladder can change between runs (e.g. a region
                    # table gets downloaded, changing the softmax denominator),
                    # so the same photo can score differently - if the photo
                    # already holds a higher confidence, keep that record and
                    # skip the write instead of downgrading the sidecar.
                    # --force explicitly overrides this: a forced re-ID always
                    # writes the new result, whatever it scores.
                    old_status = sidecar_status(sidecar)
                    old_best = _sidecar_best(sidecar)
                    if (not args.force
                            and old_status in ("done", "low-confidence")
                            and old_best is not None
                            and old_best[1] > confidence + 1e-9):
                        kept_n += 1
                        kept_lines.append(
                            f"  {cap.base}: kept '{old_best[0]}' "
                            f"{old_best[1]*100:.0f}% over new '{species}' "
                            f"{confidence*100:.0f}%")
                        last = (f"kept existing {old_best[0]} "
                                f"({old_best[1]*100:.0f}%) > new {species} "
                                f"({confidence*100:.0f}%)")
                    else:
                        tui.set(stage="sidecar", message=f"writing {sidecar.name}")
                        write_sidecar(sidecar, result, region, month, dt, status, args,
                                      scientific=scientific, avibase_url=avibase_url,
                                      avibase_id=avibase_id)
                        manifest[str(sidecar)] = {
                            "files": [str(f) for f in cap_files],
                            "sidecar": str(sidecar), "region": region, "month": month,
                            "gps": gps_src, "region_source": region_src, "date_source": date_src,
                            "species": species, "scientific": scientific, "confidence": confidence,
                            "avibase_id": avibase_id, "avibase_url": avibase_url,
                            "status": status, "classifier": classifier.describe(),
                            "pipelineVersion": PIPELINE_VERSION,
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "alternatives": result.get("alternatives", []),
                            "reasoning": result.get("reasoning", ""),
                            "datetime": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
                            "central_added": "; ".join(central_added),
                            "central_outside": "; ".join(central_outside),
                        }
                        save_manifest(manifest_path, manifest)   # resumable: save after every capture

                        report_rows.append({
                            "capture": cap.base, "file": str(primary), "viewable": primary.name,
                            "region": region, "month": f"{month:02d}", "species": species,
                            "scientific_name": scientific, "confidence": f"{confidence:.2f}",
                            "status": status, "gps": gps_src, "region_source": region_src,
                            "date_source": date_src,
                            "alternatives": "; ".join(result.get("alternatives", [])),
                            "reasoning": result.get("reasoning", ""),
                            "avibase_url": avibase_url,
                            "datetime": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
                            "central_added": "; ".join(central_added),
                            "central_outside": "; ".join(central_outside),
                        })
                        last = fmt_result(result, scientific, avibase_url)
                        if status == "non-bird":
                            nb_n += 1
                        elif status == "low-confidence":
                            lc_new += 1
                        elif status == "no-data":
                            nd_n += 1
                        else:
                            done_n += 1
                except Exception as e:  # noqa: BLE001
                    status = "error"
                    error_n += 1
                    last = f"ERROR: {e}"
                    try:
                        write_sidecar(sidecar, {}, region if "region" in dir() else args.region,
                                      month if "month" in dir() else 1, None, "error", args, error=str(e))
                        manifest[str(sidecar)] = {"files": [str(f) for f in cap_files],
                                                  "sidecar": str(sidecar),
                                                  "status": "error", "error": str(e),
                                                  "pipelineVersion": PIPELINE_VERSION,
                                                  "ts": datetime.now().isoformat(timespec="seconds"),
                                                  "alternatives": [], "reasoning": str(e)[:500],
                                                  "datetime": "",
                                                  "central_added": "", "central_outside": ""}
                        save_manifest(manifest_path, manifest)
                        report_rows.append({"capture": cap.base, "file": str(primary), "viewable": primary.name,
                                            "region": "", "month": "", "species": "", "scientific_name": "",
                                            "confidence": "", "status": "error", "gps": "", "date_source": "",
                                            "alternatives": "", "reasoning": str(e)[:500], "avibase_url": "",
                                            "datetime": "",
                                            "central_added": "", "central_outside": ""})
                    except Exception as e2:  # noqa: BLE001
                        last = f"ERROR (write failed too): {e2}"
                        tui.plain_line(f"  {last}")

                tui.set(processed=i, done=done_n + lc_new, error=error_n, pending=len(pending) - i,
                        current="", stage="", message="", last=last)
                tui.plain_line(f"[{i}/{len(pending)}] {rel}: {last}")

            # --- report ---
            # Per-folder full-snapshot reports + a parent index (feature 1).
            written = write_folder_reports(root, manifest, report_rows)
            # --report <path> still writes a flat aggregate of this run.
            if args.report and report_rows:
                flat = Path(args.report)
                flat.parent.mkdir(parents=True, exist_ok=True)
                _write_csv(flat, report_rows)
                written.append(str(flat))
            tui.set(phase="done", message="finished")
            if args.force:
                summary_lines.append(
                    "\nforced re-run: captures with existing sidecars were "
                    "re-identified; only the bird-ID fields were overwritten "
                    "(all other XMP content - e.g. IMatch metadata - preserved).")
            summary_lines.append(
                f"\ndone: {done_n} tagged, {lc_new} low-confidence, "
                f"{nd_n} no-data, {nb_n} non-bird, "
                f"{error_n} errors -> manifest {manifest_path}")
            if args.burst and burst_groups:
                summary_lines.append(
                    f"burst mode: {len(burst_groups)} burst(s) grouped by EXIF "
                    f"time (gap <= {args.burst_gap:g}s, same-bird cosine >= "
                    f"{args.burst_sim:.2f}); {len(burst_results)} frames "
                    f"combined into shared results")
            for p in written:
                summary_lines.append(f"wrote: {p}")
            if central and central.loaded and not args.no_checklist:
                summary_lines.append(central.report())

        else:
            # early exit (dry-run / nothing to do) - summary_lines already filled.
            # Still refresh the per-folder reports + parent index from the
            # manifest, so running a parent after its subfolders were processed
            # in separate runs still yields the hierarchical index.
            if not args.dry_run:
                written = write_folder_reports(root, manifest, [])
                for p in written:
                    summary_lines.append(f"wrote: {p}")
            tui.set(phase="done", message="nothing to process")

        # --- in-TUI offer for the low-confidence review pass ---
        # The TUI is still alive here (teardown is in `finally`). tui.prompt()
        # suspends the fullscreen, asks on the terminal and restores it - the
        # same mechanism the region / no-data questions use mid-run. A 'yes'
        # re-execs a fresh process with --include-low-confidence, so the review
        # pass runs with its own full TUI; a 'no' falls through to teardown and
        # the normal summary. Interrupt/error paths bypass this entirely.
        if (tui.enabled and not args.dry_run
                and not args.include_low_confidence):
            lc_offer = _low_confidence_review(manifest)
            if lc_offer:
                listing = "\n".join(
                    f"  {conf*100:4.0f}%  {name}  ({sp})"
                    for conf, sp, name in lc_offer)
                ans = tui.prompt(
                    f"{len(lc_offer)} low-confidence photo(s) (top score < "
                    f"{args.min_confidence:.2f}):\n{listing}\n"
                    f"Re-analyse them now in a TUI review pass? "
                    f"(yes = review now, no = leave them flagged) [y/N] ").lower()
                if ans in ("y", "yes"):
                    # hand off cleanly: tear the TUI down, then re-exec a fresh
                    # process so the review pass starts with its own fullscreen.
                    tui.stop_flag.set()
                    if tui_thread and tui_thread.is_alive():
                        tui_thread.join(timeout=1)
                    tui.stop()
                    argv = [a for a in sys.argv if a not in ("--no-tui", "--force")]
                    if "--include-low-confidence" not in argv:
                        argv.append("--include-low-confidence")
                    print("re-running as a TUI review pass: "
                          "--include-low-confidence\n", flush=True)
                    os.execv(sys.executable,
                             [sys.executable, os.path.abspath(__file__)] + argv[1:])
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

    # low-confidence review list - built here so it appears both after a real
    # processing run and on the early-exit path (a re-run with nothing pending).
    # It covers this run's and earlier runs' flagged captures, least-confident
    # first, straight from the manifest. (The in-TUI review offer already asked
    # about these; this is just the summary.)
    lc_review = _low_confidence_review(manifest)
    if lc_skipped:
        summary_lines.append(
            f"skipped {len(lc_skipped)} low-confidence capture(s) already "
            f"flagged (use --include-low-confidence to revisit)")
    if kept_lines:
        summary_lines.append(
            f"\nkept existing higher-confidence result for {kept_n} capture(s) "
            f"(re-analysis scored lower - sidecar left unchanged):")
        summary_lines.extend(kept_lines)
    if lc_review:
        summary_lines.append(
            f"\nlow-confidence photos for review (top score < "
            f"{args.min_confidence:.2f}, best guess kept in sidecar):")
        for conf, sp, name in lc_review:
            summary_lines.append(f"  {conf*100:4.0f}%  {name}  ({sp})")

    # no-data review list: captures whose region had no downloaded eBird data
    # (status 'no-data', best guess blank). They re-run automatically once the
    # region is downloaded; the list is so they surface for review meanwhile.
    nd_review = sorted(
        (Path(k).parent.name + "/" + Path(k).stem
         for k, e in manifest.items() if e.get("status") == "no-data"))
    if nd_review:
        summary_lines.append(
            f"\nno-data photos for review ({len(nd_review)}; region not "
            f"downloaded - best guess left blank. Download it with "
            f"'birdid --download <region>' (needs an API key) and rerun, or "
            f"rerun in TUI mode to analyze them without the prefilter):")
        for name in nd_review:
            summary_lines.append(f"  {name}")

    for line in summary_lines:
        print(line, file=sys.stderr if fatal else sys.stdout)

    if fatal:
        sys.exit(1)


if __name__ == "__main__":
    main()
