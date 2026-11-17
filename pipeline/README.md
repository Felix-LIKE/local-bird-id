# Bird Photo ID pipeline (offline)

`birdid.py` scans a folder of photos, uses EXIF GPS+time to build a species
candidate list from the offline eBird tables, and asks the local NPU vision
model (Qwen3-VL via FastFlowLM) to identify each bird. It writes IMatch-safe
XMP sidecars and a resumable manifest. **Originals are never modified** - only
`<file>.xmp` sidecars, the manifest, and the report are written. Every result
records both the **common name** and the **scientific name** (from the official
eBird taxonomy).

## One-command launcher (recommended)

`birdid` is a shell wrapper (installed as `~/.local/bin/birdid`) that does
everything for you:

```bash
birdid /path/to/photos            # auto-starts the NPU server if needed
birdid /path/to/photos --max 10   # extra birdid.py args pass straight through
birdid /path/to/photos --dry-run  # plan only, no server needed
birdid --list-candidates          # inspect the region/month tables
```

- If the FLM server isn't already running, the launcher starts
  `flm serve qwen3vl-it:4b`, waits for it, runs the pipeline, then **stops the
  server again only if it was the one that started it** (an already-running
  server is left alone).
- If `flm` is missing or the server fails to start, it prints a clear error,
  tails the server log, and exits 1.
- Env overrides: `BIRDID_MODEL`, `BIRDID_BASE_URL`, `BIRDID_START_WAIT`,
  `BIRDID_LOG`, `BIRDID_PY`.

## Prerequisites (one time)

```bash
flm pull qwen3vl-it:4b          # vision model (~3.9 GB, already pulled)
```

Needs `exiftool` and python3 with `PIL`, `lxml`, `requests` (all present). The
launcher calls `flm serve` itself, so no manual server step is needed.

## Usage (direct)

```bash
# Plan only (no NPU work, no sidecars):
python3 birdid.py /path/to/photos --dry-run

# Process everything (shows a live TUI; q = quit, resume-safe):
python3 birdid.py /path/to/photos

# Plain output, first 10 photos:
python3 birdid.py /path/to/photos --max 10 --no-tui

# Force a region for photos without GPS, and see the per-region tables:
python3 birdid.py /path/to/photos --region DE
python3 birdid.py --list-candidates
```

Defaults are fine: model `qwen3vl-it:4b`, server `127.0.0.1:52625/v1`, data in
`/data/offline-data/bird/ebird`.

## Behaviour (per the project plan)

- **Scan → manifest → worklist → process**: the manifest
  (`<folder>/birdid_manifest.json`) is updated after every capture, so a rerun
  skips anything already tagged and never redoes NPU work.
- **Idempotency**: a file is *done* iff its sidecar has `bird:species` with >=1
  entry; *error* iff `bird:status=error`.
- **RAW+JPEG pairs**: same base name = one capture -> the vision model runs
  once on the viewable file, the result is mirrored to **both** sidecars
  (`IMG_1234.jpg.xmp` + `IMG_1234.CR2.xmp`).
- **RAW-only files**: the embedded JPEG preview is extracted with exiftool for
  the model (temporary, never written over the RAW).
- **Sidecars are merged**, never blind-rewritten: existing XMP elements are
  preserved and only the `bird:` fields (+ `dc:subject`, `dc:description`) are
  updated.

## Output per photo (XMP sidecar)

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
viewable, region, month, species, scientific_name, confidence, status,
alternatives, reasoning, datetime`. Both `species` (common) and
`scientific_name` are always recorded.

## Data

- `eBird_Taxonomy_v2023.csv` - full official taxonomy (downloaded from the
  eBird API, no key) used to normalise species names.
- `regions/DE.csv`, `regions/SG.csv`, `regions/TH.csv` - species-per-month
  tables. Currently curated seed lists; regenerate from real eBird observations
  once an API key is available (see `../ebird/README.md`).

## Known limits

- Small vision models are good on common species, weaker on look-alikes /
  juveniles / females. Treat results as suggestions and confirm tricky IDs.
- Photos without GPS fall back to `--region`.
