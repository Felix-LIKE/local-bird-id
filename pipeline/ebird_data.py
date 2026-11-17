#!/usr/bin/env python3
"""
birdid data-management helpers (ebird_data.py)

Implements the `--download`, `--list`, `--api` and `--initialize` commands of
birdid. The module is importable standalone: it never imports torch or
pybioclip at module level (those are pulled in lazily only by `initialize`), so
the rest of the pipeline stays light.

Data model
----------
countries  - eBird country regions we ship curated seeds for (SG / TH / DE).
pools      - wider regional unions used by the two-tier fallback
             (SEA = Southeast Asia, EUROPE = Europe), defined by member codes.
regions/   - one CSV per country/pool:  region,species,status,month_mask
             (12 chars, Jan..Dec, '1' = plausibly present that month).

Commands
--------
- `--download country|region [id]`: with an API key, regenerates the region CSV
  in place from real eBird historic observations (sampled dates per month).
  Without a key it keeps the shipped curated seed (the no-key fallback).
- `--list [country|region]`: what is available to download (+ table status).
- `--api [key]`: store the eBird API key under ~/.config/birdid/ebird_api_key
  (chmod 600); bare `--api` prints the current status.
- `--initialize`: one-time setup - downloads the eBird taxonomy and the
  BioCLIP 2 / TreeOfLife backends so everything afterwards works offline.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

API_BASE = "https://api.ebird.org/v2"
TAXONOMY_URL = f"{API_BASE}/ref/taxonomy/ebird?fmt=csv"

DEFAULT_REGION = "DE"                 # matches birdid.py DEFAULT_REGION
DEFAULT_DATA_DIR = "/data/offline-data/bird/ebird"

CONFIG_DIR = Path(os.path.expanduser("~/.config/birdid"))
KEY_PATH = CONFIG_DIR / "ebird_api_key"

DEFAULT_YEAR = 2025                   # complete year with good coverage
DEFAULT_SAMPLE_DATES = 2              # sample dates per month
DEFAULT_API_DELAY = 1.5               # seconds between API calls (rate limit)

# BioCLIP 2 coarse stage-1 categories (single source of truth; birdid.py's
# BioclipClassifier imports this so --initialize pre-downloads the same model).
COARSE_LABELS = [
    "a photo of a bird", "a photo of a reptile", "a photo of an amphibian",
    "a photo of an insect", "a photo of a spider", "a photo of a mammal",
    "a photo of a fish", "a photo of a plant", "a photo of a fungus",
    "a photo of something else",
]

# ---------------------------------------------------------------------------
# Registry: what can be downloaded
# ---------------------------------------------------------------------------

COUNTRIES = {
    "SG": {"name": "Singapore", "pool": "SEA"},
    "TH": {"name": "Thailand", "pool": "SEA"},
    "DE": {"name": "Germany", "pool": "EUROPE"},
}

POOLS = {
    "SEA": {
        "name": "Southeast Asia",
        "members": ["SG", "MY", "ID", "TH", "VN", "PH", "KH", "LA", "MM", "BN", "TL"],
    },
    "EUROPE": {
        "name": "Europe",
        "members": [
            "AL", "AT", "BA", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE",
            "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT",
            "LU", "LV", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO",
            "RS", "SE", "SI", "SK", "UA",
        ],
    },
}


def resolve_spec(tokens: list[str]):
    """Turn `--download` args into (kind, code, label).

    Accepted forms:
      ['SG']                  bare registry code (country or pool)
      ['country']             default country (DEFAULT_REGION)
      ['region']              pool of the default country
      ['country', 'SG']       explicit country (any eBird country code)
      ['region', 'SEA']       explicit pool
    """
    t = [x.strip() for x in tokens if x.strip()]
    if not t:
        raise ValueError("--download needs a country/region code (try: --download SG, "
                         "--download country SG, --download region SEA)")
    if len(t) == 1:
        u = t[0].upper()
        if u == "COUNTRY":
            code = DEFAULT_REGION
            return "country", code, COUNTRIES[code]["name"]
        if u == "REGION":
            code = COUNTRIES[DEFAULT_REGION]["pool"]
            return "region", code, POOLS[code]["name"]
        if u in COUNTRIES:
            return "country", u, COUNTRIES[u]["name"]
        if u in POOLS:
            return "region", u, POOLS[u]["name"]
        raise ValueError(f"unknown country/region code: {t[0]!r} "
                         f"(known: {', '.join(sorted(set(COUNTRIES) | set(POOLS)))})")
    if len(t) == 2:
        kind, u = t[0].lower(), t[1].upper()
        if kind == "country":
            if u in COUNTRIES:
                return "country", u, COUNTRIES[u]["name"]
            return "country", u, u          # arbitrary eBird country code
        if kind == "region":
            if u in POOLS:
                return "region", u, POOLS[u]["name"]
            raise ValueError(f"unknown region/pool: {t[1]!r} "
                             f"(known: {', '.join(sorted(POOLS))})")
        raise ValueError("--download's first value must be 'country' or 'region' "
                         f"(got {t[0]!r})")
    raise ValueError("--download takes 1 or 2 values, e.g. '--download SG' or "
                     "'--download country SG'")


# ---------------------------------------------------------------------------
# API key store
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    try:
        key = KEY_PATH.read_text().strip()
        return key or None
    except OSError:
        return None


def set_api_key(key: str):
    key = key.strip()
    if not key:
        raise ValueError("empty API key")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(key + "\n")
    os.chmod(KEY_PATH, 0o600)
    print(f"birdid: eBird API key stored in {KEY_PATH} (mode 600).")


def handle_api(arg):
    """`--api <key>` stores the key; bare `--api` prints status."""
    if arg == "__show__":
        key = get_api_key()
        if key:
            print(f"birdid: eBird API key configured: ....{key[-4:]} "
                  f"({KEY_PATH})")
        else:
            print("birdid: no eBird API key configured.")
            print("        get a free key at https://ebird.org/api/keygen, then:")
            print("        birdid --api <your-key>")
        return
    set_api_key(arg)
    print("birdid: from now on --download prefers real eBird API data; without")
    print("        the key the curated seeds are used as the fallback.")


# ---------------------------------------------------------------------------
# eBird API (light HTTP helpers, no key needed for the taxonomy)
# ---------------------------------------------------------------------------

_last_call = [0.0]
_api_delay = [DEFAULT_API_DELAY]


def set_api_delay(seconds: float):
    _api_delay[0] = max(0.0, float(seconds))


def _throttle():
    wait = _api_delay[0] - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def http_get_json(url: str, headers: dict | None = None, retries: int = 3):
    import requests
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(url, headers=headers or {}, timeout=60)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"request failed: {e}") from e
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            retry_after = r.headers.get("Retry-After")
            try:
                sleep = min(float(retry_after), 60.0) if retry_after else 2 * (attempt + 1)
            except ValueError:
                sleep = 2 * (attempt + 1)
            time.sleep(sleep)
            continue
        if r.status_code == 403:
            raise RuntimeError(
                f"eBird API returned 403 (forbidden) for {url}\n"
                "Most endpoints need a personal API key - run:  birdid --api <your-key>\n"
                "(get a free key at https://ebird.org/api/keygen)")
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up after {retries} attempts: {url}")


def region_exists(code: str, key: str) -> bool:
    """Cheap validation that a region code is real before burning 24 requests.

    An invalid *key* raises (403 -> the user is told to run `--api`); an
    invalid *code* returns False so the caller can skip/reject it.
    """
    try:
        url = f"{API_BASE}/ref/region/info/{code}"
        data = http_get_json(url, {"X-eBirdApiToken": key})
        return bool(data) and bool(data.get("code"))
    except RuntimeError as e:
        if "403" in str(e):
            raise
        return False
    except Exception:
        return False


def download_taxonomy(data_dir: Path) -> Path:
    import requests
    dest = Path(data_dir) / "eBird_Taxonomy_v2023.csv"
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"birdid: taxonomy already present: {dest} "
              f"({dest.stat().st_size // 1024} KiB) - keeping it.")
        return dest
    print(f"birdid: downloading eBird taxonomy -> {dest} ...", flush=True)
    r = requests.get(TAXONOMY_URL, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"birdid: taxonomy saved: {dest} ({len(r.content) // 1024} KiB).")
    return dest


# ---------------------------------------------------------------------------
# Historic-observation sampling -> species-per-month
# ---------------------------------------------------------------------------

def sample_dates_for_month(year: int, month: int, n: int) -> list[str]:
    """`n` roughly-even dates within a month, returned as YYYY-MM-DD strings."""
    import calendar
    last = calendar.monthrange(year, month)[1]
    days = []
    for i in range(n):
        d = int(round((i + 1) * (last + 1) / (n + 1)))
        days.append(max(1, min(last, d)))
    return [f"{year}-{month:02d}-{d:02d}" for d in days]


def historic_obs(region_code: str, date: str, key: str) -> list[str]:
    """Species common names reported in `region_code` on `date`."""
    url = f"{API_BASE}/data/obs/{region_code}/historic/{date}"
    data = http_get_json(url, {"X-eBirdApiToken": key})
    names = [d.get("comName") for d in data if isinstance(d, dict) and d.get("comName")]
    return [n for n in names if n]


def sample_months(region_code: str, key: str, year: int, sample_dates: int,
                  progress=None) -> dict[str, set[int]]:
    """{species: {months}} from historic observations on sampled dates.

    A species is recorded present in a month iff it was reported on any sampled
    date of that month. `progress` (optional) is a callable(region, done, total).
    """
    sp_months: dict[str, set[int]] = defaultdict(set)
    total = 12 * sample_dates
    done = 0
    for month in range(1, 13):
        for date in sample_dates_for_month(year, month, sample_dates):
            for name in historic_obs(region_code, date, key):
                sp_months[name].add(month)
            done += 1
            if progress:
                progress(region_code, done, total, date)
    return dict(sp_months)


# ---------------------------------------------------------------------------
# CSV generation (overwrites the region table in place)
# ---------------------------------------------------------------------------

def _mask(months: set[int]) -> str:
    return "".join("1" if m in months else "0" for m in range(1, 13))


def write_region_csv(path: Path, region: str, sp_months: dict[str, set[int]],
                     label: str, year: int, sample_dates: int, members: list[str] | None = None):
    header = [
        f"Generated from real eBird API historic observations by `birdid --download` "
        f"on {datetime.now():%Y-%m-%d}.",
        f"Region: {region} ({label}). Source: "
        f"{API_BASE}/data/obs/.../historic/<date>.",
        f"Sample strategy: {sample_dates} date(s) per month in {year}; a species is "
        f"'present' in a month if reported on any sampled date of that month.",
    ]
    if members:
        header.append(f"Pool over member countries: {', '.join(members)}.")
    header.append("month_mask: 12 chars, Jan..Dec, '1' = reported that month. "
                  "status: resident | seasonal (informative only).")
    header.append("This table overwrote the curated seed in place - real data "
                  "is now preferred for this region.")

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("region,species,status,month_mask\n")
        for line in header:
            f.write("# " + line + "\n")
        for species in sorted(sp_months):
            months = sp_months[species]
            status = "resident" if len(months) == 12 else "seasonal"
            f.write(f"{region},{species},{status},{_mask(months)}\n")
    n = len(sp_months)
    print(f"birdid: wrote {path}  ({n} species, {len(header) + 2} lines).")
    return n


# ---------------------------------------------------------------------------
# --download
# ---------------------------------------------------------------------------

def download(data_dir: Path, tokens: list[str], year: int = DEFAULT_YEAR,
             sample_dates: int = DEFAULT_SAMPLE_DATES, key: str | None = None):
    kind, code, label = resolve_spec(tokens)
    data_dir = Path(data_dir)
    key = key or get_api_key()

    if not key:
        print(f"birdid: no API key configured - keeping the curated seed for "
              f"{code} ({label}).")
        print("        without a key, --download uses the curated no-key data "
              "(already in place).")
        print("        to fetch real eBird observations instead:")
        print("          birdid --api <your-ebird-key>   then rerun --download")
        return

    regions_dir = data_dir / "regions"
    if not regions_dir.is_dir():
        raise FileNotFoundError(f"regions dir not found: {regions_dir}")

    if kind == "country":
        if not region_exists(code, key):
            raise RuntimeError(f"eBird region code '{code}' is not valid "
                               f"(check your key / the code).")
        sp_months = _sample_with_progress(code, key, year, sample_dates)
        write_region_csv(regions_dir / f"{code}.csv", code, sp_months,
                         label, year, sample_dates)
    else:
        members = POOLS[code]["members"]
        print(f"birdid: downloading pool {code} ({label}) over "
              f"{len(members)} member countries ...")
        pool_months: dict[str, set[int]] = defaultdict(set)
        for member in members:
            if not region_exists(member, key):
                print(f"birdid:   warning: skipping invalid member code {member}")
                continue
            sp = _sample_with_progress(member, key, year, sample_dates)
            for name, months in sp.items():
                pool_months[name].update(months)
        write_region_csv(regions_dir / f"{code}.csv", code, dict(pool_months),
                         label, year, sample_dates, members=members)
        print("birdid: note: pool tables cover many countries - expect a large "
              "species count.")

    print("birdid: done. The generated table is now the committed source for "
          "this region (real eBird data preferred over the curated seed).")


def _sample_with_progress(region_code: str, key: str, year: int, sample_dates: int):
    total = 12 * sample_dates
    state = {"done": 0}

    def progress(region, done, tot, date):
        state["done"] = done
        pct = done * 100 // tot
        print(f"  {region}: {done:3d}/{tot} ({pct:3d}%)  {date}", end="\r", flush=True)
        if done == tot:
            print()

    sp = sample_months(region_code, key, year, sample_dates, progress=progress)
    return sp


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def region_table_info(regions_dir: Path, region: str) -> tuple[bool, int]:
    """(present, n_species) for a region table in the regions dir."""
    path = Path(regions_dir) / f"{region}.csv"
    if not path.is_file():
        return False, 0
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in f:
            line = row.lstrip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("region,"):
                continue
            n += 1
    return True, n


def list_regions(data_dir: Path, filter_tokens: list[str]):
    data_dir = Path(data_dir)
    regions_dir = data_dir / "regions"
    kinds = {t.lower() for t in filter_tokens if t.strip().lower() in ("country", "region")}
    show_countries = not kinds or "country" in kinds
    show_pools = not kinds or "region" in kinds

    print(f"birdid data registry   (data dir: {data_dir})")
    print(f"download with:  birdid --download <code>   |   birdid --list "
          f"country|region to filter")
    print()

    if show_countries:
        print("countries (pool = wider region used by the two-tier fallback):")
        for code in sorted(COUNTRIES):
            info = COUNTRIES[code]
            present, n = region_table_info(regions_dir, code)
            tbl = f"yes ({n} spp)" if present else "no table"
            print(f"  {code:<5} {info['name']:<22} table: {tbl:<14} pool: {info['pool']}")
        print()

    if show_pools:
        print("regions / pools (aggregate of member countries):")
        for code in sorted(POOLS):
            info = POOLS[code]
            present, n = region_table_info(regions_dir, code)
            tbl = f"yes ({n} spp)" if present else "no table"
            print(f"  {code:<5} {info['name']:<22} table: {tbl:<14} "
                  f"members: {len(info['members'])} countries")
        print()

    key = get_api_key()
    if key:
        print(f"API key: configured (....{key[-4:]}); --download fetches real "
              "eBird data.")
    else:
        print("API key: not configured; --download keeps the curated seeds.")
        print("         add a key with:  birdid --api <your-ebird-key>")


# ---------------------------------------------------------------------------
# --initialize
# ---------------------------------------------------------------------------

def initialize(data_dir: Path, regions_dir: Path):
    print("birdid: --initialize (one-time setup, needs network once) ...", flush=True)
    data_dir = Path(data_dir)
    regions_dir = Path(regions_dir)

    # 1) taxonomy (no key needed)
    download_taxonomy(data_dir)

    # 2) BioCLIP 2 + TreeOfLife backends (cached under ~/.cache/huggingface)
    print("birdid: downloading BioCLIP 2 + TreeOfLife backends ...", flush=True)
    from bioclip.predict import CustomLabelsClassifier, TreeOfLifeClassifier
    from bioclip import Rank
    CustomLabelsClassifier(COARSE_LABELS)                     # BioCLIP 2 weights
    tol = TreeOfLifeClassifier()                              # TOL-200M datafile
    tol.create_taxa_filter(Rank.CLASS, ["Aves"])              # + Aves filter tables

    if not regions_dir.is_dir():
        print(f"birdid: warning: regions dir missing: {regions_dir}")

    print("birdid: initialize complete. Everything now works offline "
          "(the launcher sets HF_HUB_OFFLINE=1 for folder runs).")
