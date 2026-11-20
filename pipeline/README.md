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
birdid /path/to/photos --month-grace 0   # strict month, no ±1-month grace
birdid /path/to/photos --analyze-all  # also analyse non-bird subjects (no pre-filter)
birdid --list-candidates          # inspect the region/month tables
birdid --list                     # what's available to download
birdid --list downloadable        # every eBird country (offline reference)
birdid --api [<key>]              # store / show the eBird API key
birdid --download <code>          # write real eBird tables to regions_generated/ (needs --api)
birdid --download country SG --subnational  # SG/<SUB>.csv per subnational1 region
birdid --download region AS-SOUTHEAST      # all 11 country files into AS/AS-SOUTHEAST/
birdid --download taxonomy        # fetch the keyless eBird/Clements taxonomy
birdid --initialize               # one-time: taxonomy + BioCLIP 2 backends
birdid --cache build <region>     # precompute the species-embedding cache (no API key)
birdid --cache clean <region>     # remove that region's data from the cache
birdid --cache update             # sync the cache with the current eBird tables
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

`birdid --download` / `--list` / `--api` / `--initialize` / `--cache` need **no photo
folder**; they manage the eBird data used as the classifier's prior.

- `--list [country|region]` prints the registry: every country and pool, whether
  each table exists locally and from which source (`curated` = committed no-key
  seed, `generated` = API-derived gitignored table), and the API-key status.
  `--list downloadable` shows every eBird country from the bundled ISO
  reference (offline, no key); with the API key,
  `--list downloadable subnational1 <CODE>` (or `subnational2`) lists a
  country's subregions - any listed code works with
  `--download country <CODE>`.
- `--api <key>` stores your eBird API key in
  `~/.config/birdid/ebird_api_key` (chmod 600); bare `--api` shows the status.
- `--download <code>` writes real-eBird tables to the **gitignored**
  `ebird/regions_generated/` **folder tree**. It always writes the **smallest
  division**: a country lands in
  `regions_generated/<CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv` (e.g.
  `AS/AS-SOUTHEAST/SG.csv`), a subcontinent fills its own folder with every
  member country, and a continent creates the subcontinent folders for every
  member country on demand. `--subnational` replaces a country table with a
  `COUNTRY/<SUB>.csv` per subnational1 region (opt-in - fetching subregions for
  every country would be thousands of API calls). A pooled `<REGION>.csv`
  union table is kept alongside each download. For each month it samples dates
  (`--sample-dates 2` by default, `--year 2025`) and unions the species
  actually reported that month. Generated tables are **API-derived and never
  committed** (eBird API data may not be redistributed); the pipeline prefers
  them over the curated seeds while present, and every directory in the tree
  also acts as a queryable union region (so `AS-SOUTHEAST`/`AS` work without a
  pooled file). No key = no generated table; only the curated seeds apply.
- `--download taxonomy` (and any no-key `--download`) fetches the **keyless**
  eBird/Clements taxonomy: it tries the Clements v2026 integrated checklist
  (Cornell) first, then the eBird API v2023 taxonomy (neither needs a key). If
  the machine is fully offline it prints the download link + the filename to
  save + where to put it, so a no-experience user can still complete the
  install manually.
- `--initialize` downloads the eBird taxonomy (no key) and the BioCLIP 2 +
  TreeOfLife backends once, so everything afterwards works offline.
- `--cache build [REGION...]` precomputes the species-embedding cache for the
  given region(s) - the union over all months of the region's candidate species
  **plus every wider pool** (subcontinent, then continent, each ±1-month
  grace-widened) and the coarse labels - so a later run over that region encodes
  nothing on the fly. With no region it covers **all loaded regions** (the
  common "build the whole cache before a trip" case). `--cache clean [REGION...]`
  **removes** the given region(s)' species from the cache and **keeps everything
  else** - for when you stop photographing a region (e.g. `--cache clean SG`
  frees Singapore's data, leaving the rest intact). With no region, `clean`
  instead drops the **stale** species no loaded table can use any more (e.g.
  after an eBird table refresh). `--cache update [REGION...]` **syncs** the
  cache with the current eBird tables: it drops stale species AND encodes the
  species the region(s) now need that aren't cached yet (incremental - only the
  delta) - the "is my cache still current?" command. `clean` needs **no model**
  and runs instantly; `build`/`update` load the shared BioCLIP 2 backbone and
  show a per-species progress bar. Region codes come from the eBird tables
  (`birdid --list`): countries (`DE`, `SG`), subcontinents (`EU-WEST`,
  `AS-SOUTHEAST`) or continents (`EUROPE`, `AS`). Coarse labels are never
  removed.

## Prerequisites (one time)

First-time setup on a fresh machine — full, distro-specific, beginner-friendly
instructions (Ubuntu / Fedora / Arch) are in the top-level `README.md`
(Installation). The short version:

```bash
python3 -m venv --system-site-packages pipeline/.venv
pipeline/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pipeline/.venv/bin/pip install pybioclip lxml requests
pipeline/birdid --initialize
```

Needs `exiftool` and python3 with `PIL`, `lxml`, `requests` (see the top-level
README for the per-distro package names, e.g. `perl-image-exiftool` on Arch,
`libimage-exiftool-perl` on Ubuntu/Debian, `perl-Image-ExifTool` on Fedora).

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

## Folder structure

The pipeline scans the given folder **recursively**; files that belong to one
photo share a base name in the same folder (a RAW+JPG pair = one capture, one
sidecar). Date-prefixed folder names (`YYMMDD-Place`, `YYMM-Event`) are used to
guess dates for photos without EXIF. Per-folder reports, parent indexes,
sidecars and the manifest are all written *next to* the photos; originals are
never touched. See **Photo folder structure** in the top-level `README.md` for
the layout and a worked example.

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
- **Centralized checklist exports** (feature 2): drop your exported eBird
  checklists (one file per checklist, e.g. `S296663962_observations.csv`) into
  the central folder (`~/.config/birdid/checklists/`, or `--checklists-dir` /
  `$BIRDID_CHECKLISTS_DIR`; created by `--initialize`). Each row's
  `Observation Date` (e.g. `Jan 28, 2026`) keys the species by (year, month),
  so a capture whose folder name resolves to that month gets those species
  added to its candidate list - a helper when the export may miss some birds or
  include birds that weren't photographed. A persistent `index.json` (mtime +
  size per file) means reruns only re-read new/changed checklist files. Species
  from the central checklists that are **not** in the GPS region's table are
  flagged in the console and recorded in the report (`central_added` /
  `central_outside` columns) - a signal that the region prior may be too narrow
  and should be expanded (or a wider pool downloaded). `--no-checklist` disables
  both local and central checklists.
- **Staged candidate ladder**: birds don't respect borders (or calendars). The
  base region is **always** scored with the month window already widened by
  `--month-grace` (strict month ∪ ±grace): a species photographed at a season
  boundary - e.g. an Asian Openbill in early January, whose `month_mask`
  starts in February - only appears once the window is widened, so stopping on
  a strict-only softmax can mis-ID a confident-looking wrong species. The
  `--expand-threshold` (default 0.35) therefore gates only widening to the
  **wider region pools** (each likewise grace-widened), **repeating until
  certain** - for a `SG` photo that is `SG ±1mo` → `AS-SOUTHEAST ±1mo` →
  `AS ±1mo` (for `DE`, which has no country seed: `EUROPE ±1mo` directly). The
  reasoning field records which stage won, and the image is encoded only once -
  every stage is a cached-embedding softmax, so walking the ladder is nearly
  free. `--expand-threshold 0` disables widening. Regions without a table fall
  back to their `REGION_POOLS` pool as tier 1 directly.
- **Reusable species-embedding cache + one shared model**: the old code rebuilt
  BioCLIP's per-species text encodings (80 ImageNet templates, ~2.7 s/species)
  and re-loaded the model on every photo and pool expansion - the reason a
  369-photo folder could stall for 26 min on its first photo. Now the whole run
  shares a single compiled backbone, and each species' embedding is computed
  once ever and cached centrally under `~/.cache/birdid/embeddings/` (override
  with `--cache-dir`; kept out of the photo folders and the repo so no derived
  files land in an IMatch-managed tree; keyed by model: `embeddings.npy` +
  `index.json`). Per photo the cost is one image encode + a cached softmax,
  independent of candidate-list size. The first time a region's species are
  seen the one-time per-species encode runs with visible progress; later runs
  (any folder) reuse the cache. The cache is **region-agnostic** (it holds
  every species ever encoded, keyed by name), so `--cache build <region>`
  precomputes exactly the species a run over that region will need, `--cache
  clean <region>` removes that region's species (keeping the rest), and
  `--cache update` syncs the whole store to the current eBird tables - see
  *Data management* above.
- **±1-month grace (`--month-grace 1`, default on)**: a species whose recorded
  presence is off by a month or two still counts as a candidate - the month
  window is widened to `[month-grace, month+grace]`, wrapping across the year.
  The base region is always scored with the grace applied (never gated by the
  confidence threshold), and every wider region tier is grace-widened too, so
  a season-boundary photo is never mis-identified. `--month-grace 0` restores
  a strict single month.
- **Live background feedback**: before the first photo the model is loaded
  explicitly with a "loading BioCLIP 2 model (first load ~2-4 min)..." message
  and a running "working on this step for MM:SS" timer, so a long first-run
  build reads as *working*, not *crashed*. During each capture the stage
  message reports the granular step ("encoding image", "scoring 227 eBird
  candidates (SG ±1mo)", "encoding species embedding 42/85: Asian Openbill
  (one-time)", "expanding to 'AS-SOUTHEAST ±1mo' (1500 candidates)", "writing
  sidecar").
- **Bird pre-filter (default on)**: stage 1 classifies every subject into a
  coarse category (bird / reptile / insect / ...). A non-bird capture is
  recorded with status `non-bird` (species = the coarse category) and the
  expensive whole-tree-of-life stage is **skipped**, so non-bird shots cost
  almost nothing. `--analyze-all` disables the pre-filter: non-bird subjects
  get the full TreeOfLife species analysis, and previously skipped non-bird
  captures are re-analysed on the next run.
- **Low-confidence handling** (`--min-confidence`, default 0.50): a capture
  whose top score stays below the threshold (blurry / distant / ambiguous bird)
  is *not* silently tagged. It still gets a sidecar with its best guess +
  confidence, but with `status = low-confidence` - so it shows in IMatch marked
  as uncertain. On re-runs such captures are **skipped** (they do not count as
  done) and the run summary lists them least-confident-first under
  "low-confidence photos for review". In TUI mode the run then asks whether to
  re-analyse them; a `yes` re-invokes the same command with
  `--include-low-confidence --no-tui` (a plain retag pass, quick now that the
  embedding cache is warm). Use `--include-low-confidence` to put flagged
  captures back in the processing queue, or lower `--min-confidence` to accept
  more marginal shots as `done`.
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
(e.g. `Parus major`), `bird:avibaseId` (e.g. `E15BC0CB`), `bird:avibaseUrl`
(short link `https://avibase.ca/E15BC0CB` → the Avibase species page for the
top guess), `bird:topConfidence`,
`bird:status` (done / low-confidence / non-bird / error), `bird:pipelineVersion`,
`bird:region`, `bird:month`, `bird:datetime`, `bird:reasoning`,
`dc:description` (includes the scientific name, and the Avibase link), and
`dc:subject` keywords
`Birds/<species>`. Register `http://ns.birdpipeline/1.0/` as a custom metadata
panel in IMatch.

Names are normalised against the official eBird taxonomy: the model's answer is
matched case-insensitively, a scientific-name answer is converted back to the
common name, and dropped leading qualifiers are restored (e.g. "Blue Tit" ->
"Eurasian Blue Tit", *Cyanistes caeruleus*). Unmatched names keep a blank
scientific field and a "check spelling" note. The Avibase id comes from the
bundled Clements checklist (`eBird-Clements_Integrated_Checklist.csv`,
'taxon concept ID' column); when that file is missing the Avibase fields are
simply omitted. The same link is printed on the per-capture console line and
stored in the report + manifest.

## Report (CSV)

Every folder that has captures gets its own full-snapshot report
`<folder>/birdid_report.csv` with one row per capture (this run + previously
recorded, merged from the manifest): `capture, file, viewable, region, month,
species, scientific_name, confidence, status, gps, date_source, alternatives,
reasoning, avibase_url, datetime, central_added, central_outside`. Both
`species` (common) and `scientific_name` are always recorded; `gps` /
`date_source` say where the location and date came from (embedded / folder-mean
/ none and exif / folder / today), `avibase_url` is the Avibase page for the top
guess (blank when the bundled Clements file is unavailable), and
`central_added` / `central_outside` list the central-checklist species used and
those missing from the region's table (feature 2).

Every folder that has subfolders-with-captures gets a **parent index**
`<folder>/birdid_index.csv` (feature 1): one row per immediate subfolder with a
pointer to that subfolder's report (or its own index) and summary counts
(`captures, done, error, pending, species`, aggregated over the subtree). The
hierarchy is therefore navigable top-down, new folders added to a parent are
picked up on rerun, and running on a subfolder alone also writes that
subfolder's report (its parent's index is refreshed the next time the parent is
run). `--report <path>` still writes a flat aggregate of the current run to an
explicit path on top of the per-folder files.

## Data

- `eBird_Taxonomy_v2023.csv` - full official taxonomy (downloaded from the
  eBird API, no key) used to normalise species names; if absent, the bundled
  Clements checklist (`eBird-Clements_Integrated_Checklist.csv`) is used
  instead (same 18,631 common→scientific pairs), so installs with no
  API/network access still get scientific names.
- `eBird-Clements_Integrated_Checklist.csv` - the bundled Clements checklist
  (version-agnostic name, so a newer release simply replaces the file). Besides
  being the taxonomy fallback it is the source of the Avibase ids: its
  `taxon concept ID` column (`avibase-XXXX...`) maps every species to its
  Avibase page (`https://avibase.ca/<id>`), surfaced for the highest-confidence
  guess.
- `regions/EUROPE.csv`, `regions/SEA.csv` - **committed curated seeds** for the
  two no-key pools (the GPS+time prior and the two-tier expansion target).
  Species names are the canonical eBird common names. A region without its own
  table (e.g. `DE`) falls back to its pool automatically.
- `regions_generated/` - **gitignored** API-derived tables written by
  `birdid --download <code>` (preferred over the seeds while present; never
  committed - eBird API data may not be redistributed). The smallest division
  is stored in a continent/subcontinent tree:
  `regions_generated/<CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv` (a country with
  subnational tables becomes `<COUNTRY>/<SUB>.csv`), with a pooled
  `<REGION>.csv` kept alongside. Every directory in the tree also acts as a
  queryable union region, so `AS-SOUTHEAST`/`AS` resolve even without a pooled
  file.
- `continents.csv` - **committed** static continent/subcontinent reference
  (250 country codes). Builds the continent + subcontinent pools for
  batch `--download region <code>`, and maps every country to its continent
  pool for the two-tier fallback (see `ebird/README.md`).
- `~/.config/birdid/checklists/` (or `--checklists-dir`) - your centralized
  eBird checklist exports, one file per checklist (feature 2).

## Known limits

- Text-image classifiers are good on common species, weaker on look-alikes /
  juveniles / females. Treat results as suggestions and confirm tricky IDs.
- Photos without GPS fall back to `--region`.
