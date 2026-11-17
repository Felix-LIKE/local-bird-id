# local-bird-id

Fully offline bird-photo identification on a local NPU, using a vision model
prioritized by GPS + time-of-year from eBird data.

Point it at a folder of bird photos and it writes an XMP sidecar (common name
**and** scientific name), a resumable manifest, and a CSV report for every
photo. **Originals are never modified** - only sidecars + manifest + report.

```
photo → EXIF (GPS + timestamp) → region + month → eBird candidate list
      → local vision model (Qwen3-VL on the NPU) → XMP sidecar + report
```

## Features

- **One-command launcher** (`birdid <folder>`): auto-starts the FastFlowLM NPU
  server if needed, waits for it, runs the pipeline, then stops the server again
  *only if it started it*. Clear error output + server-log tail if it can't.
- **Common + scientific names**: every result is normalised against the official
  eBird taxonomy (e.g. `Great Tit` / *`Parus major`*).
- **Resumable**: the manifest is saved after every capture; reruns skip done
  work and never redo vision calls.
- **RAW+JPEG pairs** = one capture → one vision call → result mirrored to both
  sidecars; RAW-only files use the embedded preview (no dcraw needed).
- **Live TUI** progress (q = quit, resume-safe) or `--no-tui` for plain output.
- **Completely offline** once the model + data are downloaded.

## Layout

| Path | What it is |
|---|---|
| `pipeline/birdid` | one-command launcher (auto-starts server, wraps `birdid.py`) |
| `pipeline/birdid.py` | the pipeline (scan → manifest → worklist → EXIF → region → candidates → vision → sidecar → report) |
| `ebird/regions/*.csv` | curated species-per-month tables for DE / SG / TH (the GPS+time prior) |
| `ebird/` | eBird data + schema docs |

See `pipeline/README.md` for usage and behaviour, and `ebird/README.md` for the
data schema and the eBird API upgrade path.

## Quick start

```bash
# one time: vision model (already pulled on this machine)
flm pull qwen3vl-it:4b

# one time: eBird taxonomy (gitignored download, ~2.6 MB, no key needed)
curl -o ebird/eBird_Taxonomy_v2023.csv "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv"

# run
pipeline/birdid ~/Pictures/birds            # auto-starts the server
pipeline/birdid ~/Pictures/birds --dry-run  # plan only, no server
pipeline/birdid --list-candidates           # inspect region/month tables
```

## Requirements

Arch Linux, FastFlowLM (`flm`) with a vision model on an AMD NPU, `exiftool`,
and python3 with `PIL`, `lxml`, `requests`.

## Status

Working end-to-end (validated on real photos: Blue Tit, Robin, Great Tit with
95% confidence and correct scientific names). The `regions/*.csv` tables are
curated seeds; the next upgrade is regenerating them from real eBird historic
observation data once an eBird API key is available.
