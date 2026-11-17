# local-bird-id

Fully offline bird-photo identification on a laptop, using a local text-image
classifier (BioCLIP 2, CPU) or an optional NPU vision model, prioritized by
GPS + time-of-year from eBird data.

Point it at a folder of bird photos and it writes an XMP sidecar (common name
**and** scientific name), a resumable manifest, and a CSV report for every
photo. **Originals are never modified** - only sidecars + manifest + report.

```
photo → EXIF (GPS + timestamp) → region + month → eBird candidate list
      → local classifier (BioCLIP 2, or Qwen3-VL on the NPU) → XMP + report
```

## Features

- **One-command launcher** (`birdid <folder>`): defaults to the BioCLIP 2 CPU
  backend (no server needed); auto-starts the FastFlowLM NPU server only if
  `--classifier vlm` is chosen.
- **Common + scientific names**: every result is normalised against the official
  eBird taxonomy (e.g. `Great Tit` / *`Parus major`*).
- **Two-tier neighbour-region expansion**: birds don't respect borders - if the
  country list yields no confident answer, the photo is re-scored against the
  wider regional pool (Southeast Asia / Europe).
- **Resumable**: the manifest is saved after every capture; reruns skip done
  work and never redo vision calls.
- **RAW+JPEG pairs** = one capture → one vision call → result mirrored to both
  sidecars; RAW-only files use the embedded preview (no dcraw needed).
- **Live TUI** progress (q = quit, resume-safe) or `--no-tui` for plain output.
- **Completely offline**: the launcher forces `HF_HUB_OFFLINE=1`; run
  `birdid --download-backends` once to cache everything.

## Layout

| Path | What it is |
|---|---|
| `pipeline/birdid` | one-command launcher (auto-starts server, wraps `birdid.py`) |
| `pipeline/birdid.py` | the pipeline (scan → manifest → worklist → EXIF → region → candidates → classify → sidecar → report) |
| `ebird/regions/*.csv` | curated species-per-month tables for DE / SG / TH (the GPS+time prior) + wider `SEA` / `EUROPE` pools for the two-tier fallback |
| `ebird/` | eBird data + schema docs |

See `pipeline/README.md` for usage and behaviour, and `ebird/README.md` for the
data schema and the eBird API upgrade path.

## Quick start

```bash
# one time: vision model (only needed for --classifier vlm; already pulled)
flm pull qwen3vl-it:4b

# one time: eBird taxonomy (gitignored download, ~2.6 MB, no key needed)
curl -o ebird/eBird_Taxonomy_v2023.csv "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv"

# one time: cache the BioCLIP 2 + TreeOfLife backends (needs network once)
pipeline/birdid --download-backends

# run (offline from here on)
pipeline/birdid ~/Pictures/birds            # BioCLIP 2, no server
pipeline/birdid ~/Pictures/birds --dry-run  # plan only
pipeline/birdid --list-candidates           # inspect region/month tables
```

> Re-tagging a folder: a capture is *done* once its `<base>.xmp` exists, so a
> rerun skips it. Delete the stale sidecars (+ `birdid_manifest.json` +
> `birdid_report.csv`) to force a fresh run - originals are never touched.

## Requirements

Arch Linux, FastFlowLM (`flm`) with a vision model on an AMD NPU, `exiftool`,
and python3 with `PIL`, `lxml`, `requests`.

## Status

Working end-to-end (validated on real photos: Blue Tit, Robin, Great Tit with
100% and correct scientific names; a 369-photo Singapore garden set including
Asian Openbill, Pink-necked Green-Pigeon and Malayan Night Heron). The
`regions/*.csv` tables are curated seeds with canonical eBird v2023 names; the
next upgrade is regenerating them from real eBird historic observation data
once an eBird API key is available.
