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
- **Data management built in**: `birdid --list`, `--download`, `--api`,
  `--initialize` and `--cache` turn `birdid` into a small eBird data CLI
  (see below).
- **Common + scientific names**: every result is normalised against the official
  eBird taxonomy (e.g. `Great Tit` / *`Parus major`*), and the top guess links
  to its **Avibase** species page (`https://avibase.ca/<id>`).
- **Your verdict is final**: confirm a capture (`birdid <folder> --confirm
  "Great Tit"`) and birdid writes `bird:confirmed` and **never overwrites it**
  again — not on re-runs, not with `--force` (the future confirm-UI writes the
  same flag; see `pipeline/README.md` → *User confirmation*). A confirmation
  also mirrors the verdict as standard **DarwinCore XMP** (`dwc:*`) with
  `dwc:identificationVerificationStatus = "verified by Person"`, so generic
  photo software can read the verified species without knowing the `bird`
  namespace — and that status itself freezes the record, so even `--force`
  can't rewrite it.
- **Staged candidate ladder** (fast + accurate): the base region is always
  scored with the ±1-month grace window applied (a species at a season
  boundary - e.g. Asian Openbill in early Jan, when its month record starts in
  Feb - only shows up once the window is widened). Only if the top score stays
  below `--expand-threshold` does it widen up the region tree (country →
  subcontinent e.g. `AS-SOUTHEAST` → continent e.g. `AS`), each wider tier
  likewise grace-widened. Confident photos never pay for the big pool;
  low-confidence ones still get the whole ladder.
- **Reusable on-disk embedding cache**: every species' BioCLIP 2 text embedding
  (the 80-ImageNet-template mean) is computed once and cached centrally under
  `~/.cache/birdid/embeddings/` (see `--cache-dir`), shared across runs and
  folders and kept out of the photo folders and this repo. After the first
  photo the per-photo cost is a single image encode + softmax, no matter how
  many species the candidate list holds - so 369-photo folders no longer stall
  on their first photo. The cache is region-agnostic (keyed by species name);
  `birdid --cache build <region>` precomputes it ahead of a trip, `birdid
  --cache clean <region>` removes a region's data when you stop photographing
  it, and `birdid --cache update` syncs it to the current eBird tables.
- **±1-month grace** (`--month-grace 1`, default on): a species whose
  presence is recorded a month earlier or later still counts as a candidate,
  so a photo taken just before/after the typical season isn't mis-identified
  (birds rarely carry a calendar). The base region and every wider tier are
  always scored with the grace applied - never gated by the confidence
  threshold. `--month-grace 0` disables it.
- **Live background feedback**: the TUI shows what is happening right now
  (loading the model, encoding, scoring N candidates, expanding a pool level,
  writing the sidecar) with a "working on this step for MM:SS" timer - a long
  first-run model build (~2-4 min) reads as *working*, not *crashed*.
- **Bird pre-filter**: every photo is first checked for "is this a bird?" with
  a cheap coarse pass. Non-bird shots (landscapes, people, macro, ...) are
  recorded as `non-bird` and skipped instead of running the expensive whole-tree
  analysis - so a big mixed collection is handled fast. `--analyze-all`
  disables the pre-filter and fully analyses every subject (previously skipped
  non-birds are re-analysed on rerun).
- **Low-confidence handling** (`--min-confidence`, default 0.50): a capture
  whose top score stays below the threshold (a blurry, distant, or ambiguous
  bird) still gets a sidecar - with its best guess + confidence kept - but with
  `status = low-confidence`, so it shows up in your photo software clearly
  marked as uncertain. Such photos are **skipped on re-runs and listed for
  review**: at
  the end of a run the summary prints them least-confident first ("low-confidence
  photos for review"), and in TUI mode it asks "re-analyse them now?" - a *yes*
  re-runs them with `--include-low-confidence --no-tui`. To revisit flagged
  photos yourself, run `birdid <folder> --include-low-confidence` (re-analyse)
  or lower `--min-confidence` (accept more marginal shots as done).
- **Burst mode** (`--burst`, opt-in): a rapid series of shots of the same bird
  (holding down the shutter on a Canon 5D Mark IV / R5 fires frames
  tens-to-hundreds of ms apart) is combined into one stronger ID instead of
  tagging each frame in isolation. Captures in the same folder whose **EXIF
  capture time** (`DateTimeOriginal`/`CreateDate` - the real shutter time, never
  the file mtime or the "pulled from the card" timestamp) falls within
  `--burst-gap` (default 3 s) are grouped, then only frames the BioCLIP
  embedding similarity confirms as the **same bird** (cosine ≥ `--burst-sim`,
  default 0.8) have their full-softmax distributions over the shared candidate
  ladder merged by **logit-averaging** (the normalized geometric mean, so N
  looks at the same bird *accumulate* evidence and confidence rises with N);
  the shared result is written to every frame's sidecar.
  Frames that fail the gate (a different bird / a person that walked through)
  keep their own single-frame ID. Without `--burst` the per-capture mode is
  exactly unchanged.
- **Resumable**: the manifest is saved after every capture; reruns skip done
  work and never redo model calls.
- **RAW+JPEG pairs** = one capture → one model call → result mirrored to both
  sidecars; RAW-only files use the embedded preview (no dcraw needed).
- **Live TUI** progress (q = quit, resume-safe) or `--no-tui` for plain output.
- **Completely offline**: the launcher forces `HF_HUB_OFFLINE=1` for folder
  runs and for `--cache` (which only touches already-downloaded data); run
  `birdid --initialize` once to cache everything.

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

## Installation

Two ways to install — pick **one** and follow it to the end. Both finish with
the same result and the same one-time model download. You don't need any
previous experience with git, Python or a terminal.

| | **Path 1 · Autoinstall** | **Path 2 · Manual** |
|---|---|---|
| Commands you run | ~4 | ~8 |
| What happens | one script does everything | you run each step yourself |
| Recommended for | most people | anyone who wants to see every step |

`sudo` in the commands below asks for your password — that is normal (it is the
same password you log in with). If a command says `command not found`, a
previous step has not finished yet — re-run it.

---

### Path 1 — Autoinstall (the easy way)

One script (`bootstrap.sh`) detects your distro, installs the few system
programs (git, Python, exiftool), builds a **private** Python environment and
puts `birdid` on your PATH — it never touches your system Python.

**1. Get the code.** Either way, you end up in a folder that contains a
subfolder called `pipeline/`:

- **If you have `git`** — paste and run:

  ```bash
  git clone https://github.com/Felix-LIKE/local-bird-id.git
  cd local-bird-id
  ```

- **If you don't** — open the GitHub page, click the green **Code** button →
  **Download ZIP**, unpack the archive, then `cd` into the unpacked folder.

  (In case you don't have `git` but want to use it, follow [Step 1 — Install
  the basic tools](#step-1--install-the-basic-tools) of the manual install,
  then come back here and run the `git clone` command above.)

**2. Run the installer** (from inside that folder):

```bash
./bootstrap.sh --dry-run   # optional & safe: only prints what it will do
./bootstrap.sh             # installs the system packages, builds the
                           # environment, links `birdid` into ~/.local/bin
```

It may ask for your sudo password once (for the system packages). Supported:
Arch-based, Debian/Ubuntu, Fedora, openSUSE and macOS (Homebrew).

**3. Reopen the terminal** (or run `source ~/.bashrc`; zsh: `source ~/.zshrc`)
so the new `birdid` command is found.

**4. Download the AI model once (~4 GB):**

```bash
birdid --initialize
```

That's it — `birdid` now works from any folder. See **Quick usage** below.

---

### Path 2 — Manual install (step by step)

The same result as Path 1, but you run each step yourself. Do them in order.

#### Step 1 — Install the basic tools

Open a terminal (most distros: `Ctrl+Alt+T`). Pick the block for your distro
and paste the whole thing in, then press Enter.

<details>
<summary><b>Ubuntu / Debian</b></summary>

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip libimage-exiftool-perl
```
</details>

<details>
<summary><b>Fedora</b></summary>

```bash
sudo dnf install -y git python3 python3-pip perl-Image-ExifTool python3-devel
```
</details>

<details>
<summary><b>Arch, Manjaro, EndeavourOS, CachyOS (and other Arch-based)</b></summary>

```bash
sudo pacman -S --needed git python python-pip perl-image-exiftool
```
</details>

What you just installed:

- `git` — downloads the project and tracks its files;
- `python3` — the language the project is written in (you don't need to learn it);
- `exiftool` (`libimage-exiftool-perl` / `perl-Image-ExifTool` /
  `perl-image-exiftool`) — reads the GPS + date information out of your photos.

Not sure which family your distro belongs to? Run `cat /etc/os-release` and
look at the `ID=` line: `ubuntu`/`debian` → first block, `fedora` → second,
`arch`/`manjaro`/`endeavouros`/`cachyos` → third.

#### Step 2 — Get the code

The project lives on GitHub. Two ways to get it onto your computer.

**Option A — with git (recommended):** needs the `git` you installed in Step 1.
Paste this, then move into the new folder:

```bash
git clone https://github.com/Felix-LIKE/local-bird-id.git
cd local-bird-id
```

**Option B — no git, download a ZIP:** open the GitHub page, click the green
**Code** button → **Download ZIP**, unpack it, then `cd` into the unpacked
folder (the one that contains a subfolder called `pipeline/`).

> **Cloning tips.** If HTTPS clone asks for a username and password, use a
> GitHub **personal access token** as the password (Settings → Developer
> settings → Personal access tokens). With an SSH key set up, use
> `git clone git@github.com:Felix-LIKE/local-bird-id.git` instead.

Run `pwd` afterwards and make sure it ends in `local-bird-id` — the remaining
steps assume you are in that folder.

#### Step 3 — Create the Python environment

This installs the project's Python libraries into a private folder
(`pipeline/.venv`) instead of touching your system Python — so nothing else on
your computer can break. Paste these four lines:

```bash
python3 -m venv pipeline/.venv
pipeline/.venv/bin/pip install --upgrade pip
pipeline/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pipeline/.venv/bin/pip install pybioclip lxml requests
```

This downloads a few hundred MB and can take several minutes.

> The `--index-url .../cpu` part installs the **CPU-only** PyTorch build, which
> keeps the environment small (~1.5–2 GB). Without it, pip installs the default
> build that bundles ~3 GB of NVIDIA GPU libraries — useless unless you have an
> NVIDIA graphics card.

#### Step 4 — Put `birdid` on your PATH (recommended)

To just type `birdid` instead of `pipeline/birdid`, link it into a folder that
is already on your PATH:

```bash
mkdir -p ~/.local/bin
ln -s "$(pwd)/pipeline/birdid" ~/.local/bin/birdid
```

If `birdid` is still not found afterwards, add `~/.local/bin` to your PATH (for
bash):

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Step 5 — Check the install

```bash
birdid --doctor    # if "birdid: command not found", use pipeline/birdid --doctor
birdid --list      # shows the built-in data (no key needed)
```

If you get a `Permission denied` when running `pipeline/birdid`, make it
executable first: `chmod +x pipeline/birdid`.

#### Step 6 — One-time model download (~4 GB)

```bash
birdid --initialize
```

This downloads the **BioCLIP 2** image classifier and the **TreeOfLife**
datafile (~4.2 GB) into `~/.cache/huggingface`, plus the eBird taxonomy. It is
needed exactly once — afterwards, identification runs are **fully offline**.
Nothing here needs an API key.

Done — see **Quick usage** below.

---

### Updating to a newer version

The program improves over time (new flags, fixes, species data). How you update
depends on how you installed it — both ways below assume the original install
worked.

**Path 1 (autoinstall) — one command.** From inside the project folder, run the
installer's update mode; it pulls the latest code and re-syncs your setup in
one go:

```bash
cd local-bird-id
./bootstrap.sh --update
```

**Path 2 (manual) — pull the code, re-sync if needed.** From inside the project
folder:

```bash
cd local-bird-id
git pull
```

That is normally all you need — the Python code updates itself. Only if a
release changes the Python libraries does a re-sync become necessary: re-run
the environment step once (for a manual install, the `pip install` command from
Step 3; the autoinstaller does this automatically as part of `--update`). If
you ever see a `ModuleNotFoundError` when starting birdid after an update, that
is the step you missed.

> **Installed from a ZIP?** There is no git history to pull. Re-download the
> latest ZIP from the GitHub page (green **Code** button → **Download ZIP**),
> unpack it over your existing folder (keeping the `ebird/` data and your
> `pipeline/.venv/`), then re-run the installer once (`./bootstrap.sh`, or the
> Step 3 pip line for a manual install).

To see what changed in the new version, open the project's GitHub page and read
the latest commits / release notes.

---

### Running on macOS or Windows

The core pipeline (`birdid.py`) is pure Python and runs on macOS and Windows —
there is no Linux-only code. Only the small `pipeline/birdid` launcher and the
setup steps above are Linux-flavoured. Data, cache and API key always live in
your user folders (`<repo>/ebird`, `~/.cache/birdid`, `~/.config/birdid`), never
in hardcoded system paths.

**macOS**

1. Install `exiftool` — a hard requirement, it reads the GPS + date out of your
   photos: `brew install exiftool` (install [Homebrew](https://brew.sh) first).
2. Create the Python environment exactly as in manual Step 3 — the venv uses the
   same `bin/python` layout as Linux:

   ```bash
   python3 -m venv pipeline/.venv
   pipeline/.venv/bin/pip install --upgrade pip
   pipeline/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pipeline/.venv/bin/pip install pybioclip lxml requests
   ```
3. `pipeline/birdid --initialize`, then run as usual. The launcher now falls
   back to a portable path resolver when macOS's `readlink` lacks `-f`, so it
   works out of the box. The TUI needs `curses`, which is built into macOS's
   Python.

**Windows**

There is no native bash, so pick one of these:

- **WSL (recommended)** — install WSL, then follow the Linux steps above.
  Everything, including the `pipeline/birdid` launcher and `--doctor`, behaves
  exactly as on Linux.
- **Git Bash / MSYS2** — the launcher mostly works, but the venv sits in
  `pipeline/.venv/Scripts/python.exe` (Windows layout), so set `BIRDID_PY` to
  point at it or just call `python birdid.py ...` directly. Install the
  [ExifTool](https://exiftool.org) Windows binary and add it to your PATH.
- **Native Python (no bash)** — run `python birdid.py /path/to/photos ...`
  directly:
  - create a venv and `pip install torch torchvision pybioclip lxml requests`,
  - install the ExifTool Windows binary and add it to your PATH,
  - the TUI needs `pip install windows-curses` — or simply add `--no-tui`
    (birdid already auto-disables the TUI when stdout isn't a terminal).

### How much disk space do I need?

| Component | Size (approx) |
|---|---|
| The code (git history + working files) | ~5 MB |
| Bundled data in the repo (taxonomy CSV + curated region seeds) | ~15 MB |
| Python environment (torch CPU + pybioclip, in `pipeline/.venv`) | ~1.5–2 GB |
| AI model weights (BioCLIP 2 + TreeOfLife, in `~/.cache/huggingface`) | ~4.2 GB |
| **Total for a working, fully offline install** | **~6 GB** |

The AI model weights dominate — the code itself is tiny. These are the sizes
for the keyless, bundled install described above.

### First-time troubleshooting

- `command not found: git` (or `sudo: apt: command not found` when you are sure
  you are on Ubuntu) → the Step 1 block for your distro did not finish; check
  the password prompt and re-run it.
- `pip ... externally-managed-environment` → you installed into the system
  Python by accident; always use `pipeline/.venv/bin/pip`, never `pip` alone.
- `ModuleNotFoundError: No module named 'torch'` → manual Step 3 did not finish;
  re-run it.
- `exiftool: command not found` → the exiftool package from Step 1 is missing.
- `Permission denied` on `pipeline/birdid` → `chmod +x pipeline/birdid`.
- The clone asks for a username and password → use a GitHub **personal access
  token** as the password, or set up an SSH key (see the cloning tips in Step 2).

## Quick usage

If you haven't installed yet, do **Installation** above first. Then:

```bash
# one time: cache the eBird taxonomy + BioCLIP 2 + TreeOfLife backends
# (needs network once; everything else works offline from here on)
pipeline/birdid --initialize

# run (fully offline) — works from any folder, not just the repo
pipeline/birdid ~/Pictures/birds            # BioCLIP 2, no server
pipeline/birdid ~/Pictures/birds --dry-run  # plan only
pipeline/birdid --list-candidates           # inspect region/month tables
pipeline/birdid --doctor                    # launcher self-diagnostics (if a run
                                            # ever ends in a confusing bash error)
```

## Photo folder structure

Point `birdid` at any folder — it scans all subfolders recursively, so you can
just point it at your whole photo collection. The only rule: **all files that
belong to one photo share the same base name in the same folder.**

A recommended layout (trip → location → photos):

```
~/Pictures/birds/
└── 2601-Singapur/                  ← one trip / event, often one month
    ├── 260128-Botanic_Garden/      ← one location, usually one day
    │   ├── 5M4_0001.CR2
    │   ├── 5M4_0001.JPG            ← RAW+JPG pair = ONE capture
    │   ├── 5M4_0002.JPG
    │   └── S123456789_observations.csv   ← optional eBird checklist export
    └── 260201-NewPlace/            ← new folders are picked up automatically
        └── 5M4_0003.JPG
```

- **Dates from folder names**: a folder named `YYMMDD-Place` (e.g.
  `260128-Botanic_Garden`) or `YYMM-Event` (e.g. `2601-Singapur`) lets the
  program guess the date when a photo has no EXIF timestamp. The `date_source`
  column in the report records where each date came from (`exif` / `folder` /
  `today`).
- **One capture = one base name**: `5M4_0001.CR2` + `5M4_0001.JPG` are treated
  as one capture → one model call → one `<base>.xmp` sidecar (written for both
  files). RAW-only files are classified from the embedded preview.
- **What the program writes** next to your photos — your originals are **never
  modified**:
  - `<base>.xmp` — the result sidecar for every capture (bird data lives in the
    custom namespace `http://ns.birdpipeline/1.0/` — see
    [`docs/birdid-xmp-namespace.md`](docs/birdid-xmp-namespace.md) to make it
    show up as `bird:*` in ExifTool/IMatch);
  - `birdid_manifest.json` — resumable bookkeeping at the folder you pointed at;
  - `birdid_report.csv` — one row per capture, in **every** folder that has any;
  - `birdid_index.csv` — per-subfolder summary counts + links, in **every**
    folder that has subfolders.
  Delete the sidecars (+ the manifest and report/index files) to re-tag a
  folder from scratch.
- **eBird checklists**: a `*_observations.csv` export dropped into a photo
  folder adds that day's species to the candidate list; a **central folder**
  (`~/.config/birdid/checklists/`, see below) does the same for every matching
  month. `--no-checklist` ignores both.

## Layout

| Path | What it is |
|---|---|
| `pipeline/birdid` | one-command launcher (wraps `birdid.py`; offline for folder runs) |
| `pipeline/birdid.py` | the pipeline (scan → manifest → EXIF → region → candidates → classify → sidecar → report) + data-management CLI |
| `pipeline/ebird_data.py` | eBird data management: registry, API key store, `--download` / `--list` / `--initialize` |
| `ebird/regions/*.csv` | **Committed curated seeds** (EUROPE + SEA, the two no-key regions) — the GPS+time prior. |
| `ebird/regions_generated/` | **Gitignored** API-derived tables from `birdid --download <code>` (preferred over the seeds; never committed — eBird API data may not be redistributed). The smallest division is written into a continent/subcontinent tree: `AS/AS-SOUTHEAST/SG.csv` (a country with subnational tables becomes `SG/<SUB>.csv`), with a pooled `<REGION>.csv` kept alongside. |
| `~/.cache/birdid/embeddings/` | **Central per-user cache** (not in this repo; override with `--cache-dir`) — per-species BioCLIP 2 text embeddings (`embeddings.npy` + `index.json`, regenerable on demand) that make the classifier's fine stage a single cached softmax per photo. |
| `ebird/continents.csv` | **Committed** static continent/subcontinent reference (250 codes) — builds the continent/subcontinent pools + country→pool map (feature 3) |
| `ebird/eBird-Clements_Integrated_Checklist.csv` | Bundled Clements checklist (version-agnostic name): offline taxonomy fallback + Avibase ids |
| `ebird/` | eBird data + schema docs |

See `pipeline/README.md` for usage and behaviour, and `ebird/README.md` for the
data schema and how `--download` upgrades the tables with real eBird data.

## Data management (eBird)

### eBird API key — where to get it and how to use it

Everything works **without** a key: the curated EUROPE + SEA seeds, the
taxonomy, and the Avibase links. A key is only needed to download **real,
current per-month species tables for a specific country or region**
(`birdid --download <code>`).

1. **Get a key:** open <https://ebird.org/api/keygen> and click the button.
   It is free and instant — you get a long token like `abc123...`.
2. **Store it on your computer:**
   ```bash
   birdid --api <your-key>
   ```
   It is saved in `~/.config/birdid/ebird_api_key` (readable only by you) and
   never shown again (bare `birdid --api` just shows the status).
3. **Use it:** download a table for the region you photographed, e.g.
   ```bash
   birdid --download SG          # Singapore
   birdid --download region SA   # the whole of South America
   ```
   The table lands in the **gitignored** `ebird/regions_generated/` and is
   preferred over the bundled seeds from then on (it is API-derived and must
   never be committed — see `ebird/README.md`).

```bash
pipeline/birdid --list              # what countries/regions can be downloaded
pipeline/birdid --list country      # countries only
pipeline/birdid --list region       # pools (SEA / EUROPE) only
pipeline/birdid --list downloadable # every eBird country (offline reference)
pipeline/birdid --list downloadable subnational1 DE   # a country's subregions (API)

# optional: bring your own eBird API key to prefer real data
pipeline/birdid --api               # show current key status
pipeline/birdid --api <your-key>    # store it (~/.config/birdid/ebird_api_key, mode 600)

# regenerate region tables from real eBird historic observations (smallest
# division, grouped into a continent/subcontinent folder tree in
# ebird/regions_generated/; a pooled <REGION>.csv is kept alongside)
pipeline/birdid --download SG       # country  (or: --download country SG)
pipeline/birdid --download SEA      # region   (or: --download region SEA)
pipeline/birdid --download region   # default country's subcontinent (DE -> EU-WEST)
pipeline/birdid --download region continent   # widest group: DE -> EUROPE
pipeline/birdid --download region continent SG # specific country: SG -> AS
pipeline/birdid --download country SG --subnational   # SG/<SUB>.csv per subregion

# batch-download a whole continent or subcontinent (from ebird/continents.csv);
# a continent creates subregion folders for every country as needed
pipeline/birdid --download region SA            # South America
pipeline/birdid --download region AS-SOUTHEAST  # fills AS/AS-SOUTHEAST/ with the 11 country files
pipeline/birdid --download region EU-NORTH      # Northern Europe

# no key needed - fetch the keyless eBird/Clements taxonomy
pipeline/birdid --download taxonomy

# species-embedding cache (no photo folder, no API key): precompute it before
# a trip so a run over that region encodes nothing on the fly
pipeline/birdid --cache build DE          # Germany (base + EU-WEST + EUROPE pools)
pipeline/birdid --cache build             # every loaded region (whole cache)
# get rid of a region's data when you stop photographing it - everything else
# stays; runs instantly, needs no model
pipeline/birdid --cache clean SG          # remove Singapore's species from the cache
pipeline/birdid --cache clean             # drop stale species no table uses
# sync the cache with the current eBird tables (drop stale + encode newly
# needed, only the delta) - run this after an eBird --download/--update
pipeline/birdid --cache update
```

- With an API key, `--download <code>` writes a real-eBird table to the
  **gitignored** `ebird/regions_generated/` (preferred over the committed seed,
  but **never committed** — eBird API data may not be redistributed). Without a
  key it keeps the curated seeds and fetches the **keyless taxonomy** instead
  (`--download taxonomy`: Cornell Clements v2026 → eBird API v2023, both
  keyless; fully offline it prints the download link + filename + target path
  so the install can be completed by hand).
- **Continent/subcontinent pools** (feature 3): besides the curated SEA/EUROPE
  seeds, the bundled `ebird/continents.csv` (standard geography, safe to
  commit) provides every continent (NA, SA, AF, AS, OC, AN + EUROPE) and
  subcontinent (EU-NORTH/EU-WEST/EU-SOUTH/EU-EAST; NA-NORTH/NA-CENTRAL/
  NA-CARIBBEAN; SA-NORTH/SA-BRAZIL/SA-SOUTH; AF-NORTH/WEST/CENTRAL/EAST/SOUTH;
  AS-EAST/AS-SOUTHEAST/AS-SOUTH/AS-CENTRAL/AS-WEST/AS-NORTH; OC-AUSTRALASIA/
  MELANESIA/MICRONESIA/POLYNESIA) as `--download region <code>` batches.
  Downloading **always writes the smallest division**: the country files go
  into `regions_generated/<CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv`, a
  continent download creates the subcontinent folders for every country on
  demand, and a pooled `<REGION>.csv` is kept alongside (`AS.csv`,
  `AS/AS-SOUTHEAST/AS-SOUTHEAST.csv`). `SEA` is the curated name of
  `AS-SOUTHEAST` (same 11 countries). `--list region` shows them all with
  member counts. Every country maps to its subcontinent (else continent) pool
  for the two-tier fallback, which walks the same folder tree up to certainty.
- **No-key coverage**: only EUROPE and SEA have committed seeds. A photo that
  resolves to a region without a table (or Germany/Singapore/Thailand, which
  ship no country seed) falls back to its pool (EUROPE / SEA, or the country's
  continent pool) automatically; every other region needs `--download <code>` +
  a key.
- **Per-folder reports + parent index** (feature 1): every folder with captures
  gets its own full-snapshot `birdid_report.csv` (this run + previously
  recorded); every folder with subfolders gets a `birdid_index.csv` pointing to
  each subfolder's report/index with counts. New folders are picked up on
  rerun, running on a subfolder alone works, and `--report <path>` still writes
  a flat aggregate.
- **Centralized eBird checklists** (feature 2): drop exported checklists
  (e.g. `S123456789_observations.csv`) into `~/.config/birdid/checklists/`
  (created by `--initialize`; override with `--checklists-dir` /
  `$BIRDID_CHECKLISTS_DIR`). Species are keyed by their `Observation Date`
  month and boost matching folders' candidate lists; species outside the GPS
  region's table are flagged (`central_outside` in the report) as a sign the
  region prior is too narrow. `--no-checklist` disables them.
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

### `--list` — see what data you have and what you can download

- `birdid --list` — the whole registry: every country and region/pool, whether
  a local table exists and from which source (`curated` = committed no-key
  seed, `generated` = made with your API key).
- `birdid --list country` / `birdid --list region` — filter to just countries
  or just the continent/subcontinent pools (with member counts).
- `birdid --list downloadable` — every eBird country from the bundled offline
  ISO reference (no key needed); with a key,
  `birdid --list downloadable subnational1 DE` lists a country's subregions.
  Any code it prints works with `birdid --download country <CODE>`.

### Central checklists folder

Drop your exported eBird checklists (one file per checklist, e.g.
`S123456789_observations.csv`) into `~/.config/birdid/checklists/` — created by
`--initialize`, or override with `--checklists-dir` / `$BIRDID_CHECKLISTS_DIR`.
Each row's `Observation Date` (e.g. `Jan 28, 2026`) keys its species by month,
so every capture whose folder resolves to that month gets those species added
to its candidate list. A hidden `index.json` caches the files (keyed by size +
modification time), so reruns only re-read new/changed checklists. Species from
these checklists that are **not** in the GPS region's table are flagged on the
console and recorded in the report's `central_outside` column — a sign the
region prior may be too narrow (download a wider pool to cover them).

> Re-tagging a folder: a capture is *done* once its `<base>.xmp` exists, so a
> rerun skips it. For a fresh identification run, use
> `birdid <folder> --force` — every capture is re-identified **in place**: the
> existing `<base>.xmp` is kept (nothing is deleted) and only the bird-ID
> fields are overwritten (`bird:*` always; birdid-owned
> `dc:subject`/`dc:description` only when `--dc-fields` is also given; the
> `dc:subject`/`dc:description` your software wrote are never touched), so it
> is safe on folders managed by your photo software where the sidecars also
> hold metadata written by that software.
> Only if you want to wipe the sidecars
> entirely, delete them (+ `birdid_manifest.json` + the per-folder
> `birdid_report.csv`/`birdid_index.csv` files) — originals are never touched.
> **User-confirmed captures are exempt from `--force`**: a capture with
> `bird:confirmed` set is frozen and never re-written (lift it with
> `birdid <folder> --unconfirm`).

## License

MIT — see [`LICENSE`](LICENSE). The bundled eBird/Clements taxonomy data is
attributed to the Cornell Lab of Ornithology (see `ebird/README.md` →
*Data attribution*); eBird observation data is not redistributed with this
repository.

The **BioCLIP 2** image classifier (`imageomics/bioclip-2`, MIT) and the
**TreeOfLife** datafile (`imageomics/TreeOfLife-200M`, CC0-1.0) are downloaded
from Hugging Face by `birdid --initialize` and are **not bundled** with this
repository; they belong to their respective authors (Allen Institute for AI /
Imageomics).

## Status

Working end-to-end (validated on real photos: Blue Tit, Robin, Great Tit with
100% and correct scientific names; a 369-photo Singapore garden set including
Asian Openbill, Pink-necked Green-Pigeon and Malayan Night Heron, 0 errors).
The repo ships curated **EUROPE** and **SEA** seeds with canonical eBird names;
`birdid --download <code>` (with an API key) writes real eBird tables to the
gitignored `ebird/regions_generated/` for any country/region.
