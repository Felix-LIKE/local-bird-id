# Plan: int8 ONNX CPU encoder — validation & make-it-default decision

**Status: decision made (2026-11-26) — keep torch as the default encoder.**
ONNX fast path stays as the documented opt-in for A/B. This document is the
accumulation point: re-run the §5b recipe against each new user-confirmed
folder to grow the comparison dataset before ever revisiting the default.
Last updated: 2026-11-26.

---

## 1. What is being validated

The auto-gated int8 ONNX CPU fast path for the BioCLIP 2 image encoder
(commit `e95116b`, branch `Testing`). It replaces the pybioclip/torch CPU
encode with an int8 ONNX session, gated on: onnxruntime installed **and** the
`.onnx` model present **and** no `--no-onnx-encoder`. NPU (VitisAI EP) remains
separate and de-prioritized.

Guiding constraints (unchanged):
- onnxruntime is **not** a hard dependency — lazy import + graceful fallback to
  torch stays the pattern.
- NPU only when hardware present AND VitisAI EP available; otherwise CPU.
- New work stays on `Testing`; `main` is frozen at `90957c4`.

Two things must both hold before this is "good":
1. **Speedup** on real photos (the point of the feature).
2. **Result equivalence** vs torch — especially no systematic **confidence
   inflation** and no new coarse-gate (bird/non-bird) failures.

## 2. Test A — 480-photo A/B (completed)

- Harness: `/tmp/opencode/validate_encoder.py`
  - Two-pass (untimed pass 1 warms species-embedding cache + encoders; pass 2
    records). Eager torch reference: `--no-onnx-encoder` + `_orig_mod` unwrap
    (torch.compile is numerically identical to eager but ~30 min cold compile).
  - 12 SG/TH folders × 40 photos, seed 42. Region from folder GPS, month from
    folder prefix. Records only top-1 species/conf/status.
- Results (`/tmp/opencode/encoder_ab_full.csv`):
  - **Speedup: 2.01× end-to-end** (torch 649 ms → onnx 322 ms per photo;
    model-only ~3×). classify/ladder ~1 ms both — unaffected.
  - **Agreement: 393/480 identical (82%)**; 63 species diffs; 24 coarse-gate flips.
  - **Confidence inflation: not confirmed.** Median `onnx_conf − torch_conf`
    across all 480 = **−0.005** (slightly *negative*; ONNX a hair less
    confident, not more). The earlier single-photo "0.28 → 0.64" was an
    outlier, not a pattern.

### Ground-truth checklist (completed by user)
`/tmp/opencode/ground_truth_checklist.csv` — 39 curated rows (24 GATE + 15
high-confidence SPECIES; the 48 low-conf lookalike drifts excluded). Filename
collisions resolved by re-classifying each folder copy and matching results
(74 extra disambiguations).

| Scope | ONNX better | torch better |
|---|---|---|
| All 39 diffs | 15 | 24 |
| Bird folders only (excl. `Reise` + `Singapur_Abend`) | **12** | **10** |

Key finding: the two **non-bird folders** (airplane window + moon) produced 17
of 39 diff rows, and torch's wins were dominated by 14 near-identical moon
photos where ONNX hallucinated "Savanna Nightjar". On real bird content int8 is
**at parity / slightly ahead**.

## 3. Test B — full-folder coarse-gate test (completed)

Reason: recheck the coarse-gate instability on non-bird night content without
the sampling/weighting artifacts of Test A. Ran **both** encoders over the
**full** contents of both known-non-bird folders (142 photos) and judged how
often each *correctly* said non-bird.

`/tmp/opencode/night_gate_full.csv` — columns: folder, photo, torch_cat,
torch_cat_conf, torch_status, torch_sp, torch_conf, onnx_* .

| Folder (all non-bird GT) | torch non-bird | onnx non-bird | bird FPs torch / onnx |
|---|---|---|---|
| `260120-Reise` (92, plane window) | 90/92 | **91/92** | 2 / 1 (all fine-conf < 0.35) |
| `260202-Singapur_Abend` (50, moon) | **25/50** | 10/50 | 25 / **40** |
| **Combined** | **115/142 (81%)** | 101/142 (71%) | 27 / 41 |

**Verdict of this test: torch gates more correctly as non-bird**, driven
entirely by the moon folder (25 vs 10). Airplane windows are a tie (~perfect).

### Moon-folder confusion matrix (50 photos)
- both say **bird**: 23
- torch-only bird: 2
- **onnx-only bird: 17**  ← the int8 over-trigger
- both non-bird: 8

All 17 onnx-only FPs are **Savanna Nightjar**, coarse `bird` 0.50–0.82, fine
0.27–0.56; **~10 of 17 clear the 0.35 fine-conf bar** (auto-confirmable).

### Mechanism
Not uniform inflation. When int8 calls something "bird" its coarse confidence
is **lower** than torch's (median 0.71 vs 0.91) but it makes **more** such
calls — the int8 coarse boundary has shifted toward "bird" on dark/aerial
content: *more, weaker* bird calls.

### Crucial nuance
**Both encoders are bad on full-frame moon.** torch itself calls 25/50 moon
photos "bird" (16 clear 0.35 — White-nest Swiftlet / Large-tailed Nightjar at
coarse 0.83–0.97). This is a model-level weakness (bright disc on black sky
maps toward aerial/night bird categories), **not int8-specific — int8 amplifies
it** (+16 net FPs). The moon folder is 50 of 7214 photos (~0.7%).

## 3b. Test C — known-positive same-species accuracy (completed)

Reason: check accuracy + false positives against a large set of real photos of
the **same known species** (real birds, bulk), for both encoders. User-supplied
ground truth.

- `260128-Botanic_Garden` `5M4_0152..0328.JPG` = **177 Malayan Night Herons**
- `260130-BishanPark` `5M4_0001..0415.JPG` minus `0319/0320/0321` = **412 Black
  Bitterns** (the excluded ones + anything after 0415 are NOT bitterns)

Both GT species verified present in the SG base ladder (a correct ID is
possible). Results: `/tmp/opencode/species_gt_full.csv`.

### Per-species (top-1 species accuracy, gate %)

| Set | torch species-acc | onnx species-acc | torch gate=bird | onnx gate=bird |
|---|---|---|---|---|
| Malayan Night Heron (177) | **81.4%** (144) | 70.1% (124) | 90.4% | 87.0% |
| Black Bittern (412) | 60.4% (249) | **64.1%** (264) | 80.3% | **86.2%** |

### Combined (589 known positives)

| | torch | onnx |
|---|---|---|
| gate=bird (no false-negative) | 491 (83.4%) | **509 (86.4%)** |
| species-correct top-1 | 393 (66.7%) | 388 (65.9%) |
| correct & conf ≥ 0.35 / 0.5 | 391 / 378 | 382 / 367 |
| wrong-ID (≥0.35 / ≥0.5) | 98 (70 / 35) | 121 (73 / 40) |
| non-bird gate FNs | 98 | **80** |

### Findings
- **Species accuracy: a wash overall** (66.7 vs 65.9%) but split by species —
  torch clearly better on Malayan Night Heron (81 vs 70, +11 pts), onnx modestly
  better on Black Bittern (64 vs 60). No systematic species-level winner.
- **Gate is a double-edged sword.** The int8 coarse boundary shifts toward
  "bird": on these known positives that **recovers real birds** (86.4 vs 83.4%
  gate, −18 gate FNs, mostly "fish"-gated Black Bitterns in water), but on the
  night/moon known-negatives it **adds false positives** (+14). Net count is
  roughly a wash; direction favors onnx (recovering real birds is worth more
  than moon noise).
- **Both encoders lose ~10–20% of real birds at the coarse gate** (Botanic:
  17/23 to "reptile" — juvenile herons; Bishan: 81/57 to "fish" — bitterns in
  water). This is a pre-existing pipeline weakness, not int8-specific; int8
  shifts it (helps waterbirds, slightly hurts herons).

## 3c. Test D — 776-photo mixed DE set, user-confirmed ground truth (completed)

Reason: a large, realistic mixed set (16 bird species + non-birds, varying
quality) with **full ground truth** — the user confirmed every photo in the
XMP sidecars (`bird:confirmed`/`topSpecies`), including where they overrode the
pipeline. Folder `260803-Neustrelitz` (DE, Aug).

Ground truth from sidecars: **732 birds (16 species) + 44 non-bird** (39 Human,
5 "No living thing"). GT species verified reachable in the DE/Europe ladder.
Results: `/tmp/opencode/de803_full.csv`, GT `/tmp/opencode/de803_gt.csv`.

### Aggregates (torch vs int8 ONNX)

| | torch | onnx |
|---|---|---|
| gate bird-FP /44 (GT non-bird) | 11 | 12 |
| gate bird-FN /732 (GT bird lost) | **15** | 20 |
| species-correct top-1 /732 | **441 (56.8%)** | 418 (53.9%) |
| correct & conf ≥ 0.5 | **368** | 344 |
| wrong-ID (≥0.5 conf) | 104 | 109 |
| correct-call conf median | 0.841 | 0.840 |

Species accuracy is low for both (~54–57%) — varying-quality photos + huge
Europe pool — but the relative comparison is clean (same ladder, same text
embeddings; only the image feature differs).

### Disagreement analysis (the sharpest signal)
Of 776 photos the two encoders **disagree on 156**:

| disagreement type | n | torch right | onnx right |
|---|---|---|---|
| gate-only | 20 | **10** | 6 |
| species-only | 136 | **38** | 18 |
| **total** | 156 | **48** | **24** |

When int8 disagrees with torch, it is wrong **2× as often** (48 vs 24; 80 both
wrong). torch's species edge is concentrated in Common Raven (+10), Gray Heron
(+12), Barn Swallow (+5); onnx only edges Common Swift (+2) and Common Buzzard
(+1).

## 4. Where the decision stands

| | int8 (ONNX) |
|---|---|
| Speed | **2× end-to-end** |
| Real bird content (480 A/B, curated checklist) | parity, slightly better (12 vs 10) |
| Known-positive same-species (589 SG) | species-acc tied (65.9 vs 66.7); gates more waterbirds in |
| **Mixed DE, full GT (776)** | **torch modestly but consistently better** (species 56.8 vs 53.9; disagreements 48 vs 24 for torch) |
| Non-bird night/moon content | **worse at the coarse gate** (+14 net FPs / 142) |

Net read after four tests: **the case for int8 being "at parity" no longer
holds at scale.** The small curated checklist (12 vs 10) and the Black Bittern
recovery masked a broader pattern: on the largest and most realistic GT set
(mixed content, full human confirmation), **torch is more accurate** — modestly
on aggregates (+3 pts species), clearly on disagreements (2:1). int8's wins are
concentrated (Black Bittern / waterbirds) and content-specific; torch's are
broad. int8's weak spots: night/moon coarse-gate FPs and Malayan Night Heron
species accuracy. The 2× speedup is real; no systematic confidence inflation
(correct-call conf identical: 0.841 vs 0.840).

Open question: **make int8 ONNX the default image encoder?**
→ **Decided 2026-11-26: no — keep torch default.** Revisit only if new
confirmed-data A/B (see §5b) shows int8 closing the disagreement gap.

Candidate paths considered:
1. **Keep torch default; ONNX opt-in.** The accuracy evidence (2:1 on
   disagreements, species +3 pts, night-gate FPs) outweighs the 2× speedup for
   a default. `--no-onnx-encoder` becomes the default stance; ONNX available for
   fast bulk passes where the user accepts slightly lower accuracy.
2. **Mitigate gate + re-test before deciding** (was the recommendation before
   Test D; the disagreement gap is mostly species-level, so a gate-only fix
   would NOT close it — only worth it if the night/moon FP is the concern).
3. Make ONNX default as-is (2× speed, accept ~2:1-wrong on disagreements).
4. Investigate the coarse boundary shift first (per-category logit deltas,
   threshold sweep) before deciding.

## 5. Next test

Tests A–D are done. The remaining question is whether any mitigation can close
int8's species-level gap (not just the gate):
- A gate-only fix would NOT close the disagreement gap (136 of 156
  disagreements are species-level, 38 vs 18 for torch). To even consider ONNX
  as default you'd need evidence a mitigation fixes species accuracy too — e.g.
  a different int8 model/calibration, or a hybrid (ONNX image encode + torch
  logits) — none of which are quick.
- More useful next step: decide whether the 2× speedup justifies ~2:1-wrong-on-
  disagreements as the default, or keep torch default and treat ONNX as the
  opt-in fast pass. Test design (if pursued): re-run ONNX on the 776 DE set
  after any encoder change and re-check the disagreement split.

Test plan / notes (fill in as decided):
- [ ]

## 5b. Reusable A/B recipe against confirmed XMP data

Any folder the user has confirmed in the viewer (XMP sidecars carry
`bird:confirmed` / `bird:topSpecies` per photo) can be turned into a clean A/B
run, no manual checklist needed. Test D did this; repeat it as confirmed data
accumulates:

1. Parse every `*.xmp` with a regex for `<bird:topSpecies>…` and
   `<bird:status>…`. Ground truth = `topSpecies`; `gt_bird` = 1 unless it is a
   non-bird label (`Human`, `No living thing`, …). Region/month come from
   `bird:region` / `bird:month` (or folder prefix).
2. Run both encoders over the matching `.JPG`s (recipe in §6 gotchas: two
   classifier instances, eager `_orig_mod` unwrap, `onnx_enc.encode_image()`
   for the ONNX side; coarse gate = top-1 `COARSE_LABELS` softmax). Columns:
   `photo, torch_{cat,cat_conf,status,sp,conf}, onnx_*`.
3. Report the three numbers that matter:
   - species-correct /732-style (top-1 == GT, plus &≥0.35 / ≥0.5),
   - gate bird-FP (non-bird called bird) and bird-FN (real bird gated out),
   - **disagreement split** (photos where torch≠onnx → torch-right vs onnx-right).
   The disagreement split is the sharpest signal (Test D: 48 vs 24).
4. Append a row to a cumulative results table so the decision re-uses every
   confirmed folder instead of one-off tests.

Note: with onnxruntime uninstalled (torch default), re-enable ONNX for a test
run with `./.venv/bin/pip install onnxruntime`, run the A/B, then
`pip uninstall -y onnxruntime` to restore the torch default.

## 6. Artifacts

- `/tmp/opencode/validate_encoder.py` — 480-photo harness (seed 42, two-pass).
- `/tmp/opencode/encoder_ab_full.csv` / `.log` — 480-photo A/B (Test A).
- `/tmp/opencode/ground_truth_checklist.csv` — 39-row user-judged checklist.
- `/tmp/opencode/night_gate_full.csv` — 142-photo full-folder gate test (Test B).
- `/tmp/opencode/species_gt_full.csv` — 589 known-positive SG test (Test C).
- `/tmp/opencode/de803_full.csv` + `/tmp/opencode/de803_gt.csv` — 776 mixed DE
  test (Test D), GT extracted from user-confirmed XMP.
- Photos: `/run/media/felix/Bilder-Backup_2/2601-Singapur/<folder>/` (Tests
  A–C), `/home/felix/Pictures/Felix-Bilder/2608-Mueritz/260803-Neustrelitz/`
  (Test D, 776 confirmed).

Reproducibility gotchas worth remembering:
- onnxruntime is **not** installed in the pipeline venv (torch is the default
  encoder). To test the ONNX fast path: `pip install onnxruntime` (README
  documents this as the opt-in), run the A/B, then uninstall again.
- `_cpu_onnx_encoder()` memoizes its tried/disabled state — flipping
  `args.no_onnx_encoder` after the first call is too late; use two classifier
  instances (one `no_onnx_encoder=True`, one `False`) and call
  `onnx_enc.encode_image(path)` directly for the ONNX side.
- `_load_coarse()` returns a backbone **shared** across instances; unwrap guard
  `hasattr(bb.model, "_orig_mod")`.
- Ladder stage-0 sizes: SG m1 = 246, TH m1 = 682; species text-embeds ~0.8 s
  each (why a cold run warms the cache first).

## 7. Decision log

- 2026-11-26: Test A (480) + user checklist done → int8 parity/slightly better
  on birds, no systematic inflation; non-bird folders skewed torch's total.
- 2026-11-26: Test B (142 full night/moon folders) done → torch gates more
  correctly as non-bird (81% vs 71%), all from the moon folder; both encoders
  bad on full-frame moon; int8 amplifies.
- 2026-11-26: Test C (589 known-positive: 177 Malayan Night Heron + 412 Black
  Bittern) done → species accuracy a wash (onnx 65.9 vs torch 66.7, split by
  species: torch better on herons, onnx on bitterns); onnx recovers more real
  birds at the gate (86.4 vs 83.4%).
- 2026-11-26: Test D (776 mixed DE, user-confirmed XMP ground truth) done →
  **torch more accurate**: species 56.8 vs 53.9, gate FNs 15 vs 20, and on
  156 disagreements torch right 48 vs onnx 24 (2:1). No confidence inflation
  (correct-call conf 0.841 vs 0.840). Evidence flips toward keeping torch
  default.
- 2026-11-26: **DECISION — keep torch as the default encoder.** Uninstalled
  onnxruntime 1.32.0 from the pipeline venv (restoring pybioclip/torch as the
  active encoder; verified `_cpu_onnx_encoder()` → None). ONNX fast path stays
  available as the documented opt-in (`pip install onnxruntime`) for A/B runs.
  This document is kept as the accumulation point: re-run §5b against each new
  user-confirmed folder to grow the dataset before revisiting the default.
