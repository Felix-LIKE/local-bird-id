# Bird Photo ID pipeline (offline)

`birdid.py` scans a folder of photos, resolves where/when each photo was taken
(embedded GPS → folder-average GPS → `--region`; EXIF date → folder name like
`260128-Botanic_Garden`), builds a species candidate list from the offline eBird
tables (optionally boosted by an eBird checklist export in the photo folder),
and classifies each bird with **BioCLIP 2** (pybioclip, CPU). It writes
IMatch-safe XMP sidecars and a resumable manifest. **Originals are never
modified** - only one `<base>.xmp` sidecar per capture, the manifest, and the
report are written. Every result records both the **common name** and the
**scientific name** (from the official eBird taxonomy).

## One-command launcher (recommended)

`birdid` is a shell wrapper (installed as `~/.local/bin/birdid`) that does
everything for you:

```bash
birdid /path/to/photos            # BioCLIP 2 on CPU, fully offline
birdid /path/to/photos --max 10   # extra birdid.py args pass straight through
birdid /path/to/photos --dry-run  # plan only, no model work
birdid --list-candidates          # inspect the region/month tables
birdid --list                     # what's available to download
birdid --list downloadable        # every eBird country (offline reference)
birdid --api [<key>]              # store / show the eBird API key
birdid --download <code>          # regenerate a region table from real eBird data
birdid --initialize               # one-time: taxonomy + BioCLIP 2 backends
```

- The only backend is **BioCLIP 2** (CPU, runs in the project venv) - no server
  is ever needed.
- **Fully offline by default**: for folder runs the launcher sets
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so the pipeline never touches
  the network. Run `birdid --initialize` once (the management commands run
  online) to cache the BioCLIP 2 weights, the TreeOfLife-200M datafile, the Aves
  filter and the eBird taxonomy under `~/.cache/huggingface`.
- Env override: `BIRDID_PY` (python to use; defaults to the project venv).

## Data management

`birdid --download` / `--list` / `--api` / `--initialize` need **no photo
folder**; they manage the eBird data used as the classifier's prior.

- `--list [country|region]` prints the registry: the countries we ship seeds
  for (SG / TH / DE), the wider pools (SEA / EUROPE, by member countries),
  whether each table exists locally, and the API-key status.
  `--list downloadable` shows every eBird country from the bundled ISO
  reference (offline, no key); with the API key,
  `--list downloadable subnational1 <CODE>` (or `subnational2`) lists a
  country's subregions - any listed code works with
  `--download country <CODE>`.
- `--api <key>` stores your eBird API key in
  `~/.config/birdid/ebird_api_key` (chmod 600); bare `--api` shows the status.
- `--download <code>` regenerates a region table **in place** from real eBird
  historic observations: for each month it samples dates (`--sample-dates 2`
  by default, `--year 2025`) and unions the species actually reported that
  month. A country downloads that country; a pool aggregates its member
  countries. With no key it keeps the curated seed (the no-key fallback).
- `--initialize` downloads the eBird taxonomy (no key) and the BioCLIP 2 +
  TreeOfLife backends once, so everything afterwards works offline.

## Prerequisites (one time)

```bash
python3 -m venv --system-site-packages pipeline/.venv
pipeline/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pipeline/.venv/bin/pip install pybioclip
pipeline/birdid --initialize
```

Needs `exiftool` and python3 with `PIL`, `lxml`, `requests` (all present).

## Usage (direct)

```bash
# Plan only (no model work, no sidecars):
python3 birdid.py /path/to/photos --dry-run

# Process everything (shows a live TUI; q = quit, resume-safe):
python3 birdid.py /path/to/photos

# Plain output, first 10 photos:
python3 birdid.py /path/to/photos --max 10 --no-tui

# Force a region for photos without GPS, and see the per-region tables:
python3 birdid.py /path/to/photos --region DE
python3 birdid.py --list-candidates
```

Defaults are fine: data in `/data/offline-data/bird/ebird`, region `DE`.

## Behaviour (per the project plan)

- **Scan → manifest → worklist → process**: the manifest
  (`<folder>/birdid_manifest.json`) is updated after every capture, so a rerun
  skips anything already tagged and never redoes model work.
- **Idempotency**: a capture is *done* iff its single sidecar
  (`<base>.xmp`) has `bird:species` with >=1 entry; *error* iff
  `bird:status=error`.
- **ONE sidecar per capture**: a RAW+JPEG pair (same base name) is processed
  once (on the viewable file) and produces a single `<base>.xmp` - never
  `<base>.jpg.xmp` / `<base>.cr2.xmp`.
- **RAW-only files**: the embedded JPEG preview is extracted with exiftool for
  the model (temporary, never written over the RAW).
- **Location fallbacks**: embedded GPS -> mean GPS of the surrounding folder
  (a day's photos are usually one area) -> `--region`. The fallback source is
  recorded in the report (`gps` column: embedded / folder-mean / none).
- **Date fallbacks**: EXIF date -> folder names `YYMMDD-Location` /
  `YYMM-Event` (`date_source` column: exif / folder / today).
- **Optional eBird checklist**: a `*_observations.csv` export in a photo folder
  adds that day's observed species to the candidate list (highest priority).
  Use `--no-checklist` to ignore. Works fully offline - it is just a CSV.
- **Two-tier neighbour-region expansion**: birds don't respect borders. If the
  top score over the base country list falls below `--expand-threshold`
  (default 0.35), the photo is re-scored against the *union* of the country
  list and a wider regional pool - `SEA` (Southeast Asia) for `SG`/`TH`,
  `EUROPE` for `DE` - so a species photographed just across a border (or missed
  by the curated table) still gets a chance. The reasoning field records when
  this fired. `--expand-threshold 0` disables it.
- **Sidecars are merged**, never blind-rewritten: existing XMP elements are
  preserved and only the `bird:` fields (+ `dc:subject`, `dc:description`) are
  updated.
- **Forcing a re-run**: a capture counts as *done* iff its sidecar is done, so
  reruns skip it. To re-tag a folder, delete the stale sidecars (+ manifest +
  report) first - originals are never touched.

## Output per photo (XMP sidecar)

One sidecar per capture, `<base>.xmp` (e.g. `IMG_1234.xmp` for
`IMG_1234.CR2` + `IMG_1234.JPG`):

`bird:species` (Seq), `bird:confidence` (Seq), `bird:summary`
("Great Tit 95% | ..."), `bird:topSpecies`, `bird:scientificName`
(e.g. `Parus major`), `bird:topConfidence`,
`bird:status` (done / low-confidence / error), `bird:pipelineVersion`,
`bird:region`, `bird:month`, `bird:datetime`, `bird:reasoning`,
`dc:description` (includes the scientific name), and `dc:subject` keywords
`Birds/<species>`. Register `http://ns.birdpipeline/1.0/` as a custom metadata
panel in IMatch.

Names are normalised against the official eBird taxonomy: the model's answer is
matched case-insensitively, a scientific-name answer is converted back to the
common name, and dropped leading qualifiers are restored (e.g. "Blue Tit" ->
"Eurasian Blue Tit", *Cyanistes caeruleus*). Unmatched names keep a blank
scientific field and a "check spelling" note.

## Report (CSV)

`<folder>/birdid_report.csv` has one row per capture: `capture, file,
viewable, region, month, species, scientific_name, confidence, status, gps,
date_source, alternatives, reasoning, datetime`. Both `species` (common) and
`scientific_name` are always recorded; `gps` / `date_source` say where the
location and date came from (embedded / folder-mean / none and exif / folder /
today).

## Data

- `eBird_Taxonomy_v2023.csv` - full official taxonomy (downloaded from the
  eBird API, no key) used to normalise species names; if absent, the bundled
  Clements v2026 checklist (`eBird-Clements_v2026_integrated-checklist-October-2026.csv`)
  is used instead (same 18,631 common→scientific pairs), so installs with no
  API/network access still get scientific names.
- `regions/DE.csv`, `regions/SG.csv`, `regions/TH.csv` - species-per-month
  country tables (the GPS+time prior). Shipped as curated seeds; `--download`
  regenerates them in place from real eBird observations with an API key.
- `regions/SEA.csv`, `regions/EUROPE.csv` - wider regional pools used by the
  two-tier expansion above (the "neighbouring countries" fallback).
  Species names in every table are the canonical eBird common names.

## Known limits

- Text-image classifiers are good on common species, weaker on look-alikes /
  juveniles / females. Treat results as suggestions and confirm tricky IDs.
- Photos without GPS fall back to `--region`.
