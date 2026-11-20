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
regions_generated/  - gitignored API-derived tables (--download), stored as
             the smallest division in a continent/subcontinent tree:
             <CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv (a country with
             subnationals becomes <COUNTRY>/<SUB>.csv), with a pooled
             <REGION>.csv kept alongside each download.

Commands
--------
- `--download country|region [id]`: with an API key, writes real eBird
  historic-observation tables (sampled dates per month) into the
  regions_generated/ tree; every directory also acts as a queryable union
  region. **Incremental**: parts already downloaded are skipped, only the
  missing data is fetched. Without a key it keeps the shipped curated seed
  (the no-key fallback).
- `--update country|region [id]`: like --download but only refreshes the parts
  whose eBird data is newer than what is stored - each generated CSV records
  a `# data_date:` (the newest observation date sampled, i.e. the date of the
  eBird data itself, not the download date) that --update compares against the
  newest data the API serves (default target: the current year, so 2025-sampled
  tables are refreshed with 2026 observations; `--year` pins it). Months not
  yet complete in the target year fall back to the previous year's
  observations so no month mask is left empty.
- `--list [country|region]`: what is available to download (+ table status).
- `--api [key]`: store the eBird API key under ~/.config/birdid/ebird_api_key
  (chmod 600); bare `--api` prints the current status.
- `--initialize`: one-time setup - downloads the eBird taxonomy and the
  BioCLIP 2 / TreeOfLife backends so everything afterwards works offline.
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from datetime import date, datetime
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

# taxonomy file names (single source of truth; birdid.py imports these)
TAXONOMY_2023 = "eBird_Taxonomy_v2023.csv"          # eBird API download (gitignored)
# bundled offline fallback; VERSION-AGNOSTIC name so a newer Clements download
# simply replaces this file (same name) - see ebird/README.md.
TAXONOMY_2026 = "eBird-Clements_Integrated_Checklist.csv"

# generated (API-derived) region tables live here - gitignored, never committed;
# the pipeline prefers them over the curated seeds in regions/.
GENERATED_DIR = "regions_generated"

# keyless taxonomy sources (no API key needed)
CLEMENTS_V2026_URL = (
    "https://www.birds.cornell.edu/clementschecklist/wp-content/uploads/"
    "2026/11/eBird-Clements_v2026_integrated-checklist-October-2026.csv")
_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

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

# Continent / subcontinent pools derived from the bundled reference
# (ebird/continents.csv, standard geography - not eBird API data). Derived
# pools never override the curated SEA/EUROPE pools above; a derived
# subcontinent that is really an existing pool is aliased to it.
POOL_ALIASES = {"AS-SOUTHEAST": "SEA"}   # derived subcontinent -> existing pool

CONTINENT_NAMES = {
    "EUROPE": "Europe", "NA": "North America", "SA": "South America",
    "AF": "Africa", "AS": "Asia", "OC": "Oceania", "AN": "Antarctica",
}
SUB_NAMES = {
    "EU-NORTH": "Northern Europe", "EU-WEST": "Western Europe",
    "EU-SOUTH": "Southern Europe", "EU-EAST": "Eastern Europe",
    "NA-NORTH": "Northern America", "NA-CENTRAL": "Central America",
    "NA-CARIBBEAN": "Caribbean",
    "SA-NORTH": "Northern South America", "SA-BRAZIL": "Brazil",
    "SA-SOUTH": "Southern South America",
    "AF-NORTH": "Northern Africa", "AF-WEST": "Western Africa",
    "AF-CENTRAL": "Central Africa", "AF-EAST": "Eastern Africa",
    "AF-SOUTH": "Southern Africa",
    "AS-EAST": "Eastern Asia", "AS-SOUTHEAST": "Southeast Asia",
    "AS-SOUTH": "Southern Asia", "AS-CENTRAL": "Central Asia",
    "AS-WEST": "Western Asia", "AS-NORTH": "Northern Asia",
    "OC-AUSTRALASIA": "Australasia", "OC-MELANESIA": "Melanesia",
    "OC-MICRONESIA": "Micronesia", "OC-POLYNESIA": "Polynesia",
    "AN": "Antarctica",
}


def _bundled_continents_path() -> Path:
    return Path(__file__).resolve().parent.parent / "ebird" / "continents.csv"


def _load_continent_pools() -> dict[str, dict]:
    """Build continent + subcontinent pools from the bundled continents.csv.

    Skips a pool whose code already exists (SEA, EUROPE) or whose member set
    duplicates an existing pool (AS-SOUTHEAST is the SEA pool); aliased
    subcontinents (POOL_ALIASES) are not re-created either.
    """
    pools: dict[str, dict] = {}
    path = _bundled_continents_path()
    if not path.is_file():
        return pools
    by_cont: dict[str, set[str]] = defaultdict(set)
    by_sub: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip().upper()
            cont = (row.get("continent") or "").strip().upper()
            sub = (row.get("subcontinent") or "").strip().upper()
            if not code:
                continue
            if cont:
                by_cont[cont].add(code)
            if sub:
                by_sub[sub].add(code)

    existing_member_sets = {frozenset(info["members"]) for info in POOLS.values()}

    def add(code: str, name: str, members: set[str]):
        ms = frozenset(members)
        if code in POOLS or ms in existing_member_sets:
            return
        pools[code] = {"name": name, "members": sorted(members)}
        existing_member_sets.add(ms)

    for cont, members in sorted(by_cont.items()):
        add(cont, CONTINENT_NAMES.get(cont, cont), members)
    for sub, members in sorted(by_sub.items()):
        if sub in POOL_ALIASES:
            continue                     # AS-SOUTHEAST is the SEA pool
        add(sub, SUB_NAMES.get(sub, sub), members)
    return pools


POOLS.update(_load_continent_pools())


_country_continent: dict[str, str] | None = None


def country_continent_map() -> dict[str, str]:
    """country code -> continent pool code, from the bundled reference."""
    global _country_continent
    if _country_continent is None:
        m: dict[str, str] = {}
        path = _bundled_continents_path()
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    code = (row.get("code") or "").strip().upper()
                    cont = (row.get("continent") or "").strip().upper()
                    if code and cont:
                        m[code] = cont
        _country_continent = m
    return _country_continent


_country_subcontinent: dict[str, str] | None = None


def country_subcontinent_map() -> dict[str, str]:
    """country code -> subcontinent pool code, from the bundled reference.

    This drives the `regions_generated/<CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv`
    directory tree that `--download` writes into.
    """
    global _country_subcontinent
    if _country_subcontinent is None:
        m: dict[str, str] = {}
        path = _bundled_continents_path()
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    code = (row.get("code") or "").strip().upper()
                    sub = (row.get("subcontinent") or "").strip().upper()
                    if code and sub:
                        m[code] = sub
        _country_subcontinent = m
    return _country_subcontinent


_subcontinent_continent: dict[str, str] | None = None


def subcontinent_continent_map() -> dict[str, str]:
    """subcontinent pool code -> its continent code, from the bundled reference."""
    global _subcontinent_continent
    if _subcontinent_continent is None:
        m: dict[str, str] = {}
        path = _bundled_continents_path()
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    code = (row.get("code") or "").strip().upper()
                    sub = (row.get("subcontinent") or "").strip().upper()
                    cont = (row.get("continent") or "").strip().upper()
                    if code and sub and cont:
                        m[sub] = cont
        _subcontinent_continent = m
    return _subcontinent_continent


_countries_cache: dict[Path, dict[str, str]] = {}


def load_countries(data_dir: Path) -> dict[str, str]:
    """code -> country name, from the bundled ISO reference (ebird/countries.csv).

    Covers every eBird country code (`birdid --list downloadable` shows them).
    """
    data_dir = Path(data_dir)
    if data_dir in _countries_cache:
        return _countries_cache[data_dir]
    out: dict[str, str] = {}
    path = data_dir / "countries.csv"
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = (row.get("code") or "").strip().upper()
                name = (row.get("name") or code).strip()
                if code and len(code) == 2:
                    out[code] = name
    _countries_cache[data_dir] = out
    return out


def resolve_spec(tokens: list[str], countries: dict[str, str] | None = None):
    """Turn `--download` args into (kind, code, label).

    Accepted forms:
      ['SG']                  bare code (registry country, pool, or any ISO
                              country in the bundled reference)
      ['country']             default country (DEFAULT_REGION)
      ['region']              pool of the default country
      ['taxonomy']            the eBird/Clements taxonomy (no key needed)
      ['country', 'SG']       explicit country (any eBird country/subnational code)
      ['region', 'SEA']       explicit pool
    """
    countries = countries or {}
    t = [x.strip() for x in tokens if x.strip()]
    if not t:
        raise ValueError("--download needs a country/region code (try: --download SG, "
                         "--download country SG, --download region SEA, "
                         "--download taxonomy)")
    if len(t) == 1:
        u = t[0].upper()
        if u == "COUNTRY":
            code = DEFAULT_REGION
            return "country", code, COUNTRIES[code]["name"]
        if u == "REGION":
            code = COUNTRIES[DEFAULT_REGION]["pool"]
            return "region", code, POOLS[code]["name"]
        if u == "TAXONOMY":
            return "taxonomy", "", "eBird/Clements taxonomy"
        if u in COUNTRIES:
            return "country", u, COUNTRIES[u]["name"]
        if u in POOLS:
            return "region", u, POOLS[u]["name"]
        if u in SUB_NAMES:
            # subcontinent (e.g. AS-SOUTHEAST, EU-WEST) - downloadable directly
            return "region", u, SUB_NAMES[u]
        if u in POOL_ALIASES:
            tgt = POOL_ALIASES[u]
            return "region", tgt, POOLS[tgt]["name"]
        if u in countries:
            return "country", u, countries[u]          # any ISO country
        raise ValueError(f"unknown country/region code: {t[0]!r} "
                         f"(registry: {', '.join(sorted(set(COUNTRIES) | set(POOLS) | set(POOL_ALIASES)))}; "
                         f"see `birdid --list downloadable` for all eBird countries)")
    if len(t) == 2:
        kind, u = t[0].lower(), t[1].upper()
        if kind == "country":
            if u in COUNTRIES:
                return "country", u, COUNTRIES[u]["name"]
            if u in countries:
                return "country", u, countries[u]
            return "country", u, u          # arbitrary eBird country/subnational code
        if kind == "region":
            if u in POOLS:
                return "region", u, POOLS[u]["name"]
            if u in SUB_NAMES:
                return "region", u, SUB_NAMES[u]
            if u in POOL_ALIASES:
                tgt = POOL_ALIASES[u]
                return "region", tgt, POOLS[tgt]["name"]
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


def _taxonomy_present(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 100_000
    except OSError:
        return False


def download_taxonomy(data_dir: Path) -> Path:
    """Ensure a taxonomy is available - every source is KEYLESS (no API key).

    Preference: an existing v2023/v2026 file, then the Clements v2026
    integrated checklist (Cornell), then the eBird API v2023 taxonomy, then -
    if fully offline - clear manual download instructions (the Cornell page is
    Cloudflare-protected and sometimes blocks scripted access).
    """
    import requests
    data_dir = Path(data_dir)
    v23 = data_dir / TAXONOMY_2023
    v26 = data_dir / TAXONOMY_2026

    if _taxonomy_present(v23):
        print(f"birdid: taxonomy already present: {v23} "
              f"({v23.stat().st_size // 1024} KiB) - keeping it.")
        return v23
    if _taxonomy_present(v26):
        print(f"birdid: taxonomy already present: {v26} "
              f"({v26.stat().st_size // 1024} KiB) - keeping it.")
        return v26

    # 1) newest + richest: Clements v2026 integrated checklist (Cornell, keyless)
    try:
        print("birdid: downloading the Clements v2026 checklist (no key needed) ...",
              flush=True)
        r = requests.get(CLEMENTS_V2026_URL, headers={"User-Agent": _BROWSER_UA},
                         timeout=120)
        r.raise_for_status()
        v26.write_bytes(r.content)
        print(f"birdid: saved {v26} ({len(r.content) // 1024} KiB, Clements v2026).")
        return v26
    except Exception as e:  # Cloudflare etc.
        print(f"birdid: note: Cornell download not reachable "
              f"({e.__class__.__name__}); trying the eBird API taxonomy instead.")

    # 2) eBird API v2023 taxonomy (keyless, reliable)
    try:
        print("birdid: downloading the eBird API taxonomy (no key needed) ...",
              flush=True)
        r = requests.get(TAXONOMY_URL, timeout=120)
        r.raise_for_status()
        v23.write_bytes(r.content)
        print(f"birdid: saved {v23} ({len(r.content) // 1024} KiB).")
        return v23
    except Exception as e:
        print("birdid: ERROR: could not download a taxonomy - network unavailable.")
        print("birdid: manual fallback (no API key needed):")
        print(f"birdid:   1. open  {CLEMENTS_V2026_URL}")
        print(f"birdid:   2. save the CSV as  {v26.name}")
        print(f"birdid:   3. put it at  {v26}")
        print("birdid: then rerun this command - it will use the file as-is.")
        raise RuntimeError(f"no taxonomy available (network error: {e})") from e


# ---------------------------------------------------------------------------
# Historic-observation sampling -> species-per-month
# ---------------------------------------------------------------------------

def sample_dates_for_month(year: int, month: int, n: int) -> list[tuple[int, int, int]]:
    """`n` roughly-even dates within a month, returned as (year, month, day)."""
    import calendar
    last = calendar.monthrange(year, month)[1]
    days = []
    for i in range(n):
        d = int(round((i + 1) * (last + 1) / (n + 1)))
        days.append(max(1, min(last, d)))
    return [(year, month, d) for d in days]


def sampling_year_for_month(target_year: int, month: int, today: date | None = None) -> int:
    """The observation year to sample for `month` when targeting `target_year`.

    A month's data is only taken from the target year once that month is fully
    in the past (its last day <= today); otherwise the most recent complete
    instance of the month (`target_year - 1`) is sampled instead. So updating
    to the *current* year mid-year never produces empty tail-month masks: on
    2026-11-20, Jan-Oct use 2026 observations while Nov-Dec fall back to 2025
    (December simply has not happened yet in 2026).
    """
    today = today or date.today()
    import calendar
    last = calendar.monthrange(target_year, month)[1]
    if date(target_year, month, last) <= today:
        return target_year
    return target_year - 1


def historic_obs(region_code: str, year: int, month: int, day: int, key: str) -> list[str]:
    """Species common names reported in `region_code` on a historic date.

    URL shape: /v2/data/obs/{regionCode}/historic/{y}/{m}/{d}
    (the date is three path segments, not a single YYYY-MM-DD string).
    """
    url = f"{API_BASE}/data/obs/{region_code}/historic/{year}/{month}/{day}"
    data = http_get_json(url, {"X-eBirdApiToken": key})
    names = [d.get("comName") for d in data if isinstance(d, dict) and d.get("comName")]
    return [n for n in names if n]


def sample_months(region_code: str, key: str, year: int, sample_dates: int,
                  progress=None, today: date | None = None
                  ) -> tuple[dict[str, set[int]], date]:
    """({species: {months}}, newest observation date sampled) from historic
    observations on sampled dates.

    A species is recorded present in a month iff it was reported on any sampled
    date of that month. `year` is the *target* sampling year; any month not yet
    complete in it falls back to `year - 1` observations, so an in-progress
    year never yields empty month masks (see sampling_year_for_month).
    `progress` (optional) is a callable(region, done, total, date).
    """
    sp_months: dict[str, set[int]] = defaultdict(set)
    total = 12 * sample_dates
    done = 0
    newest: date | None = None
    for month in range(1, 13):
        m_year = sampling_year_for_month(year, month, today)
        for y, m, d in sample_dates_for_month(m_year, month, sample_dates):
            for name in historic_obs(region_code, y, m, d, key):
                sp_months[name].add(month)
            done += 1
            dt = date(y, m, d)
            if newest is None or dt > newest:
                newest = dt
            if progress:
                progress(region_code, done, total, f"{dt:%Y-%m-%d}")
    return dict(sp_months), newest or date(year, 12, 31)


# ---------------------------------------------------------------------------
# CSV generation (overwrites the region table in place)
# ---------------------------------------------------------------------------

def _mask(months: set[int]) -> str:
    return "".join("1" if m in months else "0" for m in range(1, 13))


def write_region_csv(path: Path, region: str, sp_months: dict[str, set[int]],
                     label: str, year: int, sample_dates: int,
                     members: list[str] | None = None, data_date: str = ""):
    """Write a region table, recording the eBird data date it was built from.

    `data_date` is the newest observation date sampled (the date of the eBird
    data itself, NOT the download date) - written as a `# data_date:` header
    line that `--update` later compares against to decide what is stale.
    """
    header = [
        f"Generated from real eBird API historic observations by `birdid --download` "
        f"on {datetime.now():%Y-%m-%d}.",
        f"Region: {region} ({label}). Source: "
        f"{API_BASE}/data/obs/.../historic/{{y}}/{{m}}/{{d}}.",
        f"Sample strategy: {sample_dates} date(s) per month in {year}; a species is "
        f"'present' in a month if reported on any sampled date of that month.",
    ]
    if members:
        header.append(f"Pool over member countries: {', '.join(members)}.")
    header.append("month_mask: 12 chars, Jan..Dec, '1' = reported that month. "
                  "status: resident | seasonal (informative only).")
    header.append("This is API-derived eBird data - it lives in the gitignored "
                  "regions_generated/ dir and is NOT committed to git; the "
                  "curated seed in regions/ stays as the fallback.")
    # data freshness metadata (used by --update): the year the data is from and
    # the newest observation date sampled (not the download date).
    if year >= datetime.now().year:
        header.append(f"Months not yet complete in {year} fall back to "
                      f"{year - 1} observations, so no month is left empty.")
    header.append(f"data_year: {year}")
    if data_date:
        header.append(f"data_date: {data_date}")

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

# Smallest-division layout of the gitignored regions_generated/ tree:
#   regions_generated/<CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv
#   regions_generated/<CONTINENT>/<SUBCONTINENT>/<COUNTRY>/<SUB>.csv  (opt-in)
#   regions_generated/<CONTINENT>/<SUBCONTINENT>/<SUBCONTINENT>.csv   (pooled)
#   regions_generated/<CONTINENT>.csv                                  (pooled)
# A continent download creates subcontinent folders for every not-yet-existing
# subregion and drops the country files into them; a subcontinent download
# fills its own folder. Pooled files are kept alongside ("keep both"), so a
# flat union table stays available while the smallest division is also there.


def _country_rel_dir(code: str) -> Path:
    """Relative directory for a country's files in regions_generated/, e.g.
    SG -> AS/AS-SOUTHEAST. Falls back to '.' when the geography is unknown."""
    cont = country_continent_map().get(code)
    sub = country_subcontinent_map().get(code)
    if cont and sub:
        return Path(cont) / sub
    if cont:
        return Path(cont)
    return Path(".")


def _pool_tree_code(code: str) -> str:
    """Map a pool code to the subcontinent/continent it represents in the tree
    (SEA is the curated name of AS-SOUTHEAST; every other code is already a
    tree name)."""
    return "AS-SOUTHEAST" if code == "SEA" else code


def _read_meta(path: Path, key: str) -> str | None:
    """Value of a '# key: value' header line in a generated region CSV."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    k, _, v = line[2:].partition(":")
                    if k.strip() == key:
                        return v.strip()
    except OSError:
        pass
    return None


def _load_region_csv(path: Path) -> dict[str, set[int]]:
    """{species: {months}} from an existing region CSV (comment/header safe)."""
    sp: dict[str, set[int]] = defaultdict(set)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row or (row.get("region") or "").lstrip().startswith("#"):
                    continue
                name = (row.get("species") or "").strip()
                mask = (row.get("month_mask") or "").strip()
                if not name or len(mask) != 12:
                    continue
                for i, ch in enumerate(mask):
                    if ch == "1":
                        sp[name].add(i + 1)
    except OSError:
        pass
    return dict(sp)


def _target_data_date(year: int, sample_dates: int, today: date | None = None) -> str:
    """The newest observation date a fresh sampling of `year` would reach - the
    'data date' `--update` compares existing tables against (ISO string)."""
    today = today or date.today()
    newest: date | None = None
    for month in range(1, 13):
        my = sampling_year_for_month(year, month, today)
        for y, m, d in sample_dates_for_month(my, month, sample_dates):
            dt = date(y, m, d)
            if newest is None or dt > newest:
                newest = dt
    return (newest or date(year, 12, 31)).isoformat()


def _sample_or_skip(dest: Path, region_code: str, label: str, key: str, year: int,
                    sample_dates: int, today: date | None,
                    update: bool, target_date: str = "") -> tuple[dict | None, str]:
    """Sample `region_code` into `dest`, skipping it when not needed.

    Returns (species_months | None, data_date | None); None means the table was
    skipped (already present for --download, or already current for --update).

    - --download only fills gaps: an existing table is left alone.
    - --update re-samples a table whose recorded data_date is older than the
      target data date; a table with no recorded date (written before this
      feature existed) is treated as stale and refreshed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        local = _read_meta(dest, "data_date")
        if not update:
            print(f"birdid:   {region_code}: table exists, skipping "
                  f"(use --update to refresh)")
            return None, ""
        if target_date and local and local >= target_date:
            print(f"birdid:   {region_code}: current ({local}) - skipping")
            return None, ""
        if local:
            print(f"birdid:   {region_code}: data from {local} is older than "
                  f"target {target_date} - refreshing")
        else:
            print(f"birdid:   {region_code}: no data date recorded - refreshing")
    sp, data_date = _sample_with_progress(region_code, key, year, sample_dates,
                                          today=today)
    write_region_csv(dest, region_code, sp, label, year, sample_dates,
                     data_date=data_date.isoformat())
    return sp, data_date.isoformat()


def _pool_data_date(generated_dir: Path, members: list[str]) -> str:
    """Newest recorded data_date across a set of member tables (the pooled
    table's own data date), or '' when none are dated."""
    dates = []
    for m in members:
        p = _find_region_csv(generated_dir, m)
        if p:
            d = _read_meta(p, "data_date")
            if d:
                dates.append(d)
    return max(dates) if dates else ""


def _download_members(generated_dir: Path, members: list[str], key: str,
                      year: int, sample_dates: int,
                      override_dir: Path | None, today: date | None = None,
                      update: bool = False, target_date: str = "") -> dict[str, set[int]]:
    """Sample every member country, skipping those already present (--download)
    or already current (--update), and return the unioned {species: months} for
    the pooled file - built from ALL member tables present after the run, so an
    incremental download does not drop existing members from the pool."""
    pool_months: dict[str, set[int]] = defaultdict(set)
    for member in members:
        if not region_exists(member, key):
            print(f"birdid:   warning: skipping invalid member code {member}")
            continue
        dest_dir = override_dir or (generated_dir / _country_rel_dir(member))
        dest = dest_dir / f"{member}.csv"
        sp, _ = _sample_or_skip(dest, member, member, key, year, sample_dates,
                                today, update, target_date)
        if sp is None:
            sp = _load_region_csv(dest)   # skipped (exists) - still pool it
        for name, months in sp.items():
            pool_months[name].update(months)
    return pool_months


def _download_country(generated_dir: Path, code: str, label: str, year: int,
                      sample_dates: int, key: str, subnational: bool,
                      today: date | None = None, update: bool = False,
                      target_date: str = ""):
    """Write one country's table into <CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv
    (or, with --subnational, its subnational1 tables into
    <CONTINENT>/<SUBCONTINENT>/<COUNTRY>/<SUB>.csv - the smallest division).
    Existing tables are skipped by --download and only refreshed by --update
    when their eBird data is older than the target."""
    if not region_exists(code, key):
        raise RuntimeError(f"eBird region code '{code}' is not valid "
                           f"(check your key / the code).")
    if subnational:
        try:
            subs = ebird_region_list("subnational1", code, key)
        except RuntimeError as e:
            print(f"birdid:   note: could not list subnational1 regions "
                  f"({e}); falling back to a country table")
            subs = []
        if subs:
            country_dir = generated_dir / _country_rel_dir(code) / code
            country_dir.mkdir(parents=True, exist_ok=True)
            # the subnational tables replace the flat country file (smallest
            # division): drop the superseded <COUNTRY>.csv so the loader's
            # dir-union is used, not the old pooled file.
            flat = generated_dir / _country_rel_dir(code) / f"{code}.csv"
            if flat.is_file():
                flat.unlink()
                print(f"birdid:   removed superseded flat table {flat} "
                      f"(replaced by {len(subs)} subnational tables)")
            written = 0
            for sub in subs:
                sub_code = sub["code"]
                sp, _ = _sample_or_skip(country_dir / f"{sub_code}.csv",
                                        sub_code, sub.get("name", sub_code),
                                        key, year, sample_dates, today,
                                        update, target_date)
                if sp is not None:
                    written += 1
            print(f"birdid: {written} subnational table(s) for {code} processed "
                  f"(skipped {len(subs) - written} already present/current) "
                  f"under {country_dir}/ (smallest division; no pooled file - "
                  f"the {code} region is the union of these).")
            print(f"birdid: done. eBird data written to {country_dir} "
                  f"(API-derived, gitignored - not committed).")
            return
        print(f"birdid:   note: {code} has no subnational1 regions - writing a "
              f"country table instead.")
    dest = generated_dir / _country_rel_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{code}.csv"
    sp_months, _ = _sample_or_skip(target, code, label, key, year, sample_dates,
                                   today, update, target_date)
    if sp_months is None:
        print(f"birdid: done. {code} table already "
              f"{'present' if not update else 'current'} - skipped "
              f"(use {'--update' if not update else '--year with a newer year'} "
              f"to refresh it).")
    else:
        print(f"birdid: done. eBird data written to {target} "
              f"(API-derived, gitignored - not committed).")


def _download_pool(generated_dir: Path, code: str, label: str, year: int,
                   sample_dates: int, key: str, today: date | None = None,
                   update: bool = False):
    """Write the smallest division for a continent/subcontinent pool: every
    member country table in the <CONTINENT>/<SUBCONTINENT>/ tree, plus a pooled
    union table (kept alongside, per the 'keep both' choice). Existing member
    tables are skipped (--download) or only refreshed when stale (--update)."""
    tgt = _pool_tree_code(code)
    target_date = _target_data_date(year, sample_dates, today) if update else ""
    verb = "updating" if update else "downloading"
    if tgt in SUB_NAMES:
        # subcontinent (AS-SOUTHEAST, EU-WEST, ...): fill its own folder
        cont = subcontinent_continent_map().get(tgt)
        members = sorted(c for c, s in country_subcontinent_map().items() if s == tgt)
        if not members:
            raise ValueError(f"no countries found for subcontinent {tgt}")
        print(f"birdid: {verb} subcontinent {tgt} ({label}) over "
              f"{len(members)} member countries -> regions_generated/{cont}/{tgt}/ ...")
        sub_dir = generated_dir / cont / tgt
        pool_months = _download_members(generated_dir, members, key, year,
                                        sample_dates, override_dir=sub_dir,
                                        today=today, update=update,
                                        target_date=target_date)
        write_region_csv(sub_dir / f"{tgt}.csv", tgt, dict(pool_months),
                         label, year, sample_dates, members=members,
                         data_date=_pool_data_date(sub_dir, members))
        print(f"birdid: wrote pooled {tgt}.csv ({len(pool_months)} spp) "
              f"alongside {len(members)} country tables in {sub_dir}/")
        print(f"birdid: done. eBird data written to {sub_dir} (API-derived, "
              f"gitignored - not committed).")
    elif tgt in CONTINENT_NAMES:
        # continent (AS, AF, NA, SA, OC, EUROPE): create subcontinent folders
        # for every member country and drop the country files into them.
        members = sorted(c for c, cont in country_continent_map().items() if cont == tgt)
        if not members:
            raise ValueError(f"no countries found for continent {tgt}")
        print(f"birdid: {verb} continent {tgt} ({label}) over "
              f"{len(members)} member countries; subcontinent folders under "
              f"regions_generated/{tgt}/ are created on demand ...")
        pool_months = _download_members(generated_dir, members, key, year,
                                        sample_dates, override_dir=None,
                                        today=today, update=update,
                                        target_date=target_date)
        write_region_csv(generated_dir / f"{tgt}.csv", tgt, dict(pool_months),
                         label, year, sample_dates, members=members,
                         data_date=_pool_data_date(generated_dir, members))
        print(f"birdid: wrote pooled {tgt}.csv ({len(pool_months)} spp) "
              f"alongside the country tables under regions_generated/{tgt}/")
        print(f"birdid: done. eBird data written to regions_generated/{tgt}/ "
              f"(API-derived, gitignored - not committed).")
    else:
        raise ValueError(f"unknown region/pool: {code} "
                         f"(known: {', '.join(sorted(POOLS))})")


def _run_download(data_dir: Path, tokens: list[str], year: int | None,
                  sample_dates: int, key: str | None, subnational: bool,
                  update: bool):
    kind, code, label = resolve_spec(tokens, load_countries(data_dir))
    data_dir = Path(data_dir)
    key = key or get_api_key()

    if kind == "taxonomy":
        # keyless: the eBird/Clements taxonomy (region tables still need a key)
        try:
            download_taxonomy(data_dir)
        except RuntimeError as e:
            print(f"birdid: ERROR: {e}")
            return
        if not key:
            print("birdid: taxonomy is ready; region tables need an API key - "
                  "get a free one at https://ebird.org/api/keygen, then "
                  "birdid --api <your-key> and rerun --download <code>.")
        return

    if not key:
        # still fetch the keyless taxonomy so a no-key user gets at least that
        try:
            download_taxonomy(data_dir)
        except RuntimeError as e:
            print(f"birdid: ERROR: {e}")
        print(f"birdid: no API key configured - no generated table written for "
              f"{code} ({label}).")
        print("        without a key, --download/--update can only fetch the "
              "taxonomy (above); region tables use the curated seeds "
              "(EUROPE/SEA) or none.")
        print("        to fetch real eBird observations instead:")
        print("          birdid --api <your-ebird-key>   then rerun "
              "--download/--update")
        return

    if year is None:
        # --update defaults to the current year (the newest data eBird serves);
        # --download keeps the curated default so a complete year is sampled.
        year = date.today().year if update else DEFAULT_YEAR

    generated_dir = data_dir / GENERATED_DIR
    generated_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()

    if kind == "country":
        _download_country(generated_dir, code, label, year, sample_dates, key,
                          subnational, today=today, update=update)
    else:
        _download_pool(generated_dir, code, label, year, sample_dates, key,
                       today=today, update=update)


def download(data_dir: Path, tokens: list[str], year: int | None = None,
             sample_dates: int = DEFAULT_SAMPLE_DATES, key: str | None = None,
             subnational: bool = False):
    """Fetch the missing part of a region: tables already downloaded are left
    alone (only gaps are fetched), so re-running `--download EU-WEST` does not
    burn API calls on countries you already have. `year` defaults to a complete
    observation year (DEFAULT_YEAR); pass --year to choose."""
    _run_download(data_dir, tokens, year, sample_dates, key, subnational,
                  update=False)


def update(data_dir: Path, tokens: list[str], year: int | None = None,
           sample_dates: int = DEFAULT_SAMPLE_DATES, key: str | None = None,
           subnational: bool = False):
    """Refresh the parts of a region whose eBird data is older than the newest
    available (default target: the current year, so a table sampled from 2025
    data is refreshed with 2026 observations; pass --year to pin it). Missing
    tables are fetched too. Months not yet complete in the target year fall
    back to the previous year's observations, so no month mask is left empty."""
    _run_download(data_dir, tokens, year, sample_dates, key, subnational,
                  update=True)


def _sample_with_progress(region_code: str, key: str, year: int, sample_dates: int,
                          today: date | None = None):
    total = 12 * sample_dates
    state = {"done": 0}

    def progress(region, done, tot, date):
        state["done"] = done
        pct = done * 100 // tot
        print(f"  {region}: {done:3d}/{tot} ({pct:3d}%)  {date}", end="\r", flush=True)
        if done == tot:
            print()

    sp, data_date = sample_months(region_code, key, year, sample_dates,
                                  progress=progress, today=today)
    return sp, data_date


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def _count_rows(path: Path) -> int:
    n = 0
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in f:
                line = row.lstrip()
                if not line or line.startswith("#") or line.startswith("region,"):
                    continue
                n += 1
    except OSError:
        return 0
    return n


def _find_region_csv(base: Path, region: str) -> Path | None:
    """Locate <region>.csv anywhere under `base` (tree + flat legacy)."""
    if not base.is_dir():
        return None
    for p in base.rglob(f"{region}.csv"):
        if p.is_file():
            return p
    return None


def region_table_info(curated_dir: Path, generated_dir: Path, region: str
                      ) -> tuple[str, int]:
    """(source, n_species) for a region table, "" when none.

    source is "generated" (API-derived, preferred), "curated" (committed seed)
    or "" (no table). Handles the nested regions_generated/ tree and a country
    stored as a directory of subnational tables (region = union of the dir).
    """
    for source, base in (("generated", generated_dir), ("curated", curated_dir)):
        if not Path(base).is_dir():
            continue
        path = _find_region_csv(Path(base), region)
        if path:
            return source, _count_rows(path)
        d = Path(base) / region
        if d.is_dir():
            n = sum(_count_rows(p) for p in sorted(d.rglob("*.csv")) if p.is_file())
            if n:
                return source, n
    return "", 0


def ebird_region_list(region_type: str, parent_code: str | None, key: str) -> list[dict]:
    """eBird region reference list: GET /v2/ref/region/list/{type}[/{parent}]."""
    url = f"{API_BASE}/ref/region/list/{region_type}"
    if parent_code:
        url += f"/{parent_code}"
    data = http_get_json(url, {"X-eBirdApiToken": key})
    return [d for d in data if isinstance(d, dict) and d.get("code")]


def list_downloadable(data_dir: Path, tokens: list[str], key: str | None):
    """`--list downloadable [subnational1 <CODE>]` - what can be downloaded.

    Plain `downloadable` lists every eBird country from the bundled ISO
    reference (offline, no key needed). `subnational1 <CODE>` /
    `subnational2 <CODE>` lists a country's subregions via the API (needs key).
    """
    data_dir = Path(data_dir)
    curated_dir = data_dir / "regions"
    generated_dir = data_dir / GENERATED_DIR

    rest = [t for t in tokens if t != "downloadable"]
    if rest:
        rtype = rest[0]
        if rtype in ("subnational1", "subnational2"):
            if not key:
                print("birdid: --list downloadable %s needs your eBird API key "
                      "(birdid --api <your-key>)" % rtype)
                return
            if len(rest) < 2:
                print(f"birdid: --list downloadable {rtype} needs a parent country "
                      f"code, e.g.  birdid --list downloadable {rtype} DE")
                return
            parent = rest[1].upper()
            try:
                rows = ebird_region_list(rtype, parent, key)
            except RuntimeError as e:
                print(f"birdid: ERROR: {e}")
                return
            print(f"eBird {rtype} regions of {parent} available to download "
                  f"({len(rows)}):   birdid --download country <CODE>")
            for r in sorted(rows, key=lambda d: d["code"]):
                print(f"  {r['code']:<5} {r['name']}")
            print()
            print("  subnational codes download via:  birdid --download country <CODE>")
            return
        print(f"birdid: unknown downloadable filter {rest[0]!r} "
              f"(use 'country' or 'subnational1 <CODE>'/'subnational2 <CODE>')")
        return

    # plain `downloadable` -> bundled ISO country reference (offline)
    countries = load_countries(data_dir)
    keyline = f" (API key ....{key[-4:]})" if key else " (no key - local reference)"
    print(f"eBird countries available to download ({len(countries)}){keyline}:")
    print("  birdid --download <CODE>   or   --download country <CODE>")
    print("  pool hierarchy: country -> subcontinent -> continent (continents.csv)")
    for code in sorted(countries):
        name = countries[code]
        src, n = region_table_info(curated_dir, generated_dir, code)
        local = f"{src} ({n} spp)" if src else "-"
        sub = country_subcontinent_map().get(code, "-")
        sub_disp = POOL_ALIASES.get(sub, sub)      # AS-SOUTHEAST shows as SEA
        cont = (subcontinent_continent_map().get(sub)
                or country_continent_map().get(code, "-"))
        print(f"  {code:<5} {name:<34} local: {local:<16} "
              f"sub: {sub_disp:<12} continent: {cont}")
    print()
    print("  local = a region CSV already in the data dir: 'curated' = committed "
          "seed (no key), 'generated' = API-derived (gitignored, preferred);")
    print("  sub/continent = the pool each country downloads into "
          "(e.g. DE -> EU-WEST -> EUROPE): download 'region <SUB>' to get that "
          "subcontinent, 'region <CONTINENT>' for the whole continent.")
    print("  subnational regions:  birdid --list downloadable subnational1 <CODE>")


def list_regions(data_dir: Path, filter_tokens: list[str]):
    data_dir = Path(data_dir)
    curated_dir = data_dir / "regions"
    generated_dir = data_dir / GENERATED_DIR
    lowered = [t.lower() for t in filter_tokens]

    if "downloadable" in lowered:
        list_downloadable(data_dir, lowered, get_api_key())
        return

    kinds = {t for t in lowered if t in ("country", "region")}
    show_countries = not kinds or "country" in kinds
    show_pools = not kinds or "region" in kinds

    print(f"birdid data registry   (data dir: {data_dir})")
    print(f"download with:  birdid --download <code>   |   birdid --list "
          f"country|region to filter")
    print()
    print("table: curated = committed no-key seed; generated = API-derived "
          "(gitignored, preferred); - = none (needs a key to create).")
    print()

    if show_countries:
        print("countries (pool hierarchy: country -> subcontinent -> continent):")
        for code in sorted(COUNTRIES):
            info = COUNTRIES[code]
            src, n = region_table_info(curated_dir, generated_dir, code)
            tbl = f"{src} ({n} spp)" if src else "none"
            sub = country_subcontinent_map().get(code, "-")
            sub_disp = POOL_ALIASES.get(sub, sub)
            cont = (subcontinent_continent_map().get(sub)
                    or country_continent_map().get(code, "-"))
            print(f"  {code:<5} {info['name']:<22} table: {tbl:<20} "
                  f"sub: {sub_disp:<10} continent: {cont}")
        print()

    if show_pools:
        continents = sorted(c for c in POOLS if c in CONTINENT_NAMES)
        subcons = sorted(set(country_subcontinent_map().values()))
        print("regions / pools (smallest-division country files grouped by "
              "continent / subcontinent folders):")
        print("  continent pools:")
        for code in continents:
            info = POOLS[code]
            src, n = region_table_info(curated_dir, generated_dir, code)
            tbl = f"{src} ({n} spp)" if src else "none"
            n_mem = sum(1 for c, cc in country_continent_map().items() if cc == code)
            print(f"    {code:<10} {info['name']:<22} table: {tbl:<18} "
                  f"members: {n_mem} countries (all subregions)")
        if subcons:
            print("  subcontinent pools:")
            for code in subcons:
                name = SUB_NAMES.get(code, code)
                cont = subcontinent_continent_map().get(code, "")
                src, n = region_table_info(curated_dir, generated_dir, code)
                tbl = f"{src} ({n} spp)" if src else "none"
                n_mem = sum(1 for s in country_subcontinent_map().values() if s == code)
                print(f"    {code:<14} {name:<24} table: {tbl:<18} "
                      f"members: {n_mem} countries   (under {cont}/)")
        if POOL_ALIASES:
            print(f"  alias: {', '.join(f'{a} -> {t}' for a, t in sorted(POOL_ALIASES.items()))} "
                  f"(no separate table; downloads/uses the target pool)")
        print()

    key = get_api_key()
    if key:
        print(f"API key: configured (....{key[-4:]}); --download writes real "
              "eBird data to regions_generated/ (preferred over the seeds).")
    else:
        print("API key: not configured; only the curated seeds (EUROPE/SEA) "
              "are used. Other regions need a key:  birdid --api <your-ebird-key>")


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

    (data_dir / GENERATED_DIR).mkdir(parents=True, exist_ok=True)

    print("birdid: initialize complete. Everything now works offline "
          "(the launcher sets HF_HUB_OFFLINE=1 for folder runs).")
