# local-bird-id

Fully offline bird-photo identification on a laptop, using a local
text-image classifier (**BioCLIP 2** on CPU) prioritized by GPS + time-of-year
from eBird data.

Point it at a folder of bird photos and it writes an XMP sidecar (common name
**and** scientific name), a resumable manifest, and a CSV report for every
photo. **Originals are never modified** - only sidecars + manifest + report.

```
photo → EXIF (GPS + timestamp) → region + month → eBird candidate list
      → BioCLIP 2 (CPU) → XMP + report
```

## Features

- **One-command launcher** (`birdid <folder>`): BioCLIP 2 on CPU, no server,
  fully offline after a one-time `--initialize`.
- **Data management built in**: `birdid --list`, `--download`, `--api` and
  `--initialize` turn `birdid` into a small eBird data CLI (see below).
- **Common + scientific names**: every result is normalised against the official
  eBird taxonomy (e.g. `Great Tit` / *`Parus major`*), and the top guess links
  to its **Avibase** species page (`https://avibase.ca/<id>`).
- **Two-tier neighbour-region expansion**: birds don't respect borders - if the
  country list yields no confident answer, the photo is re-scored against the
  wider regional pool (Southeast Asia / Europe).
- **Resumable**: the manifest is saved after every capture; reruns skip done
  work and never redo model calls.
- **RAW+JPEG pairs** = one capture → one model call → result mirrored to both
  sidecars; RAW-only files use the embedded preview (no dcraw needed).
- **Live TUI** progress (q = quit, resume-safe) or `--no-tui` for plain output.
- **Completely offline**: the launcher forces `HF_HUB_OFFLINE=1` for folder
  runs; run `birdid --initialize` once to cache everything.

### Never needs a key vs needs a key

| Feature | API key? |
|---|---|
| `birdid --initialize` (taxonomy + BioCLIP 2 backends) | **Never** |
| eBird/Clements taxonomy (names, scientific names, Avibase links) | **Never** |
| Curated species prior for the two pools: **EUROPE** & **SEA** | **Never** |
| Avibase link on every top guess | **Never** |
| Real, current species-per-month table for a specific country (e.g. `--download SG`) | **Yes** (free at ebird.org/api/keygen) |
| Subnational region lists (`--list downloadable subnational1 …`) | **Yes** |

**Without an API key** the repo ships exactly two curated region seeds
(**Europe** and **Southeast Asia**) as the classification prior — photos from
other regions need a generated table (`--download <code>`, key required).
Generated tables are API-derived and therefore **never committed** (they live
in the gitignored `ebird/regions_generated/`); the curated seeds stay as the
fallback.

## Layout

| Path | What it is |
|---|---|
| `pipeline/birdid` | one-command launcher (wraps `birdid.py`; offline for folder runs) |
| `pipeline/birdid.py` | the pipeline (scan → manifest → EXIF → region → candidates → classify → sidecar → report) + data-management CLI |
| `pipeline/ebird_data.py` | eBird data management: registry, API key store, `--download` / `--list` / `--initialize` |
| `ebird/regions/*.csv` | **Committed curated seeds** (EUROPE + SEA, the two no-key regions) — the GPS+time prior. |
| `ebird/regions_generated/*.csv` | **Gitignored** API-derived tables from `birdid --download <code>` (preferred over the seeds; never committed — eBird API data may not be redistributed) |
| `ebird/eBird-Clements_Integrated_Checklist.csv` | Bundled Clements checklist (version-agnostic name): offline taxonomy fallback + Avibase ids |
| `ebird/` | eBird data + schema docs |

See `pipeline/README.md` for usage and behaviour, and `ebird/README.md` for the
data schema and how `--download` upgrades the tables with real eBird data.

## Quick start

```bash
# one time: cache the eBird taxonomy + BioCLIP 2 + TreeOfLife backends
# (needs network once; everything else works offline from here on)
pipeline/birdid --initialize

# run (fully offline)
pipeline/birdid ~/Pictures/birds            # BioCLIP 2, no server
pipeline/birdid ~/Pictures/birds --dry-run  # plan only
pipeline/birdid --list-candidates           # inspect region/month tables
```

## Data management (eBird)

```bash
pipeline/birdid --list              # what countries/regions can be downloaded
pipeline/birdid --list country      # countries only
pipeline/birdid --list region       # pools (SEA / EUROPE) only
pipeline/birdid --list downloadable # every eBird country (offline reference)
pipeline/birdid --list downloadable subnational1 DE   # a country's subregions (API)

# optional: bring your own eBird API key to prefer real data
pipeline/birdid --api               # show current key status
pipeline/birdid --api <your-key>    # store it (~/.config/birdid/ebird_api_key, mode 600)

# regenerate a region table from real eBird historic observations
pipeline/birdid --download SG       # country  (or: --download country SG)
pipeline/birdid --download SEA      # region   (or: --download region SEA)
pipeline/birdid --download region   # default region's pool (DE -> EUROPE)

# no key needed - fetch the keyless eBird/Clements taxonomy
pipeline/birdid --download taxonomy
```

- With an API key, `--download <code>` writes a real-eBird table to the
  **gitignored** `ebird/regions_generated/` (preferred over the committed seed,
  but **never committed** — eBird API data may not be redistributed). Without a
  key it keeps the curated seeds and fetches the **keyless taxonomy** instead
  (`--download taxonomy`: Cornell Clements v2026 → eBird API v2023, both
  keyless; fully offline it prints the download link + filename + target path
  so the install can be completed by hand).
- **No-key coverage**: only EUROPE and SEA have committed seeds. A photo that
  resolves to a region without a table (or Germany/Singapore/Thailand, which
  ship no country seed) falls back to its pool (EUROPE / SEA) automatically;
  every other region needs `--download <code>` + a key.
- Tune the sampling: `--sample-dates 2` (dates/month), `--year 2025`,
  `--api-delay 1.5` (seconds between calls, rate-limit guard).
- The species tables use the canonical eBird names; the bundled
  `ebird/eBird-Clements_Integrated_Checklist.csv` serves as the scientific-name
  taxonomy for installs with no API/network access, and its `taxon concept ID`
  column provides the **Avibase link** (`https://avibase.ca/<id>`) recorded for
  every top guess (sidecar + report + manifest + console). To use a newer
  Clements release, download it from the Cornell checklist page and save it
  under the **same filename** (`eBird-Clements_Integrated_Checklist.csv`) in
  `ebird/` — see `ebird/README.md`.

> Re-tagging a folder: a capture is *done* once its `<base>.xmp` exists, so a
> rerun skips it. Delete the stale sidecars (+ `birdid_manifest.json` +
> `birdid_report.csv`) to force a fresh run - originals are never touched.

## Requirements

Arch Linux, `exiftool`, and python3 with `PIL`, `lxml`, `requests` and
`pybioclip` (BioCLIP 2; pulls torch + torchvision). Install once:

```bash
python3 -m venv --system-site-packages pipeline/.venv
pipeline/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pipeline/.venv/bin/pip install pybioclip
```

## Status

Working end-to-end (validated on real photos: Blue Tit, Robin, Great Tit with
100% and correct scientific names; a 369-photo Singapore garden set including
Asian Openbill, Pink-necked Green-Pigeon and Malayan Night Heron, 0 errors).
The repo ships curated **EUROPE** and **SEA** seeds with canonical eBird names;
`birdid --download <code>` (with an API key) writes real eBird tables to the
gitignored `ebird/regions_generated/` for any country/region.
