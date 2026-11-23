# birdid XMP namespace — make it visible in your photo software

Every bird-ID result lives in the custom XMP namespace

```
http://ns.birdpipeline/1.0/
```

That is deliberate: it can never collide with tags written by your photo
software (IMatch, Lightroom, …). The trade-off is that generic photo apps
don't know the namespace, so by default they either ignore the tags or show
them with an ugly auto-generated prefix (`XMP-ns0`, `ns0:`…).

This page shows how to register the namespace so the tags get the friendly
`bird:` prefix / `XMP-bird` group, and what to expect from photo-administration
software (Lightroom, IMatch, …).

The tags birdid writes (see `pipeline/README.md` → "Output per photo"):

| Element | Content |
|---|---|
| `bird:species` | ranked list (Seq) of candidate species |
| `bird:confidence` | matching confidence list (Seq) |
| `bird:summary` | "Swinhoe's White-eye 33% \| …" one-liner |
| `bird:topSpecies` | best guess (common name) |
| `bird:scientificName` | best guess (e.g. *Zosterops simplex*) |
| `bird:avibaseId` / `bird:avibaseUrl` | Avibase reference for the top guess |
| `bird:topConfidence` | score of the best guess |
| `bird:confirmed` | user-confirmed species (e.g. `Great Tit`) — once set, the record is **frozen**: birdid never overwrites it |
| `bird:status` | `done` / `low-confidence` / `non-bird` / `error` |
| `bird:pipelineVersion`, `bird:region`, `bird:month`, `bird:datetime`, `bird:reasoning` | provenance of the ID |
| `bird:dcOwned`, `bird:error` | housekeeping |

> **Standard XMP fields are opt-in.** By default birdid writes **only** its
> `bird:*` fields and never touches the standard XMP fields (`dc:description`,
> `dc:subject`, …) — so it never fights your DAM over who owns the caption and
> keywords. If you want birdid to maintain them too (so even a viewer that
> knows nothing about the `bird` namespace shows a caption and keywords), pass
> **`--dc-fields`** on the command line — see [the section
> below](#standard-xmp-fields-dcdescription--dcsubject-are-opt-in).

---

## Standard XMP fields (`dc:description` / `dc:subject`) are opt-in

Run with **`--dc-fields`** to have birdid also write the standard, widely
readable XMP fields on the sidecars it created:

```bash
birdid ~/Pictures/birds --dc-fields
```

With the flag, on **birdid-owned** sidecars (ones birdid itself created,
marked `bird:dcOwned`) it additionally writes:

- **`dc:description`** — a human-readable one-liner: top species, scientific
  name, confidence, alternatives and the Avibase link;
- **`dc:subject`** — keywords `Birds/<species>` (or `Non-bird/<category>`),
  up to the top five candidates — the fields XnViewMP, Windows Explorer and
  most simple viewers show as caption / keywords.

Two guarantees hold in **both** modes (flag on or off):

- **Foreign sidecars are never touched.** A sidecar that pre-existed from other
  software (no `bird:dcOwned` marker, no `Birds/…` keyword) always gets only
  the `bird:*` fields — with or without `--dc-fields` — so captions and
  keywords written by your photo software are preserved.
- **Nothing is ever deleted.** Without the flag birdid simply stops writing
  `dc:*`; it does **not** remove `dc:description`/`dc:subject` that are already
  in a sidecar (they just go stale if the species later changes). To fully
  clear them, delete the sidecars and re-run.

Without `--dc-fields`, `dc:*` in a birdid-owned sidecar behaves like any other
metadata written by your photo software: birdid's re-runs leave it untouched.

## DarwinCore mirror (written with every `--confirm`)

Since a confirmation is exactly the "this identification was verified" fact that
DarwinCore exists to express, `--confirm` additionally mirrors `bird:confirmed`
as a standard **DarwinCore XMP** block in the namespace ExifTool/Exiv2/IMatch's
bundled ExifTool all declare for DarwinCore XMP:

```
http://rs.tdwg.org/dwc/index.htm
```

(The canonical `http://rs.tdwg.org/dwc/terms/` was deliberately *not* used:
ExifTool only labels `terms/` as `XMP-dwc` if registered in a config, and then
warns about a namespace-prefix conflict on every run. The legacy URI reads as
clean `XMP-dwc:*` with zero config in the whole photo toolchain.)

What `--confirm` writes (see `pipeline/README.md` → *User confirmation*):

| dwc element | content | source |
|---|---|---|
| `dwc:Occurrence` → `dwc:recordedBy` | the photographer | photo's `Artist` tag — only when actually present |
| `dwc:Identification` → `dwc:dateIdentified` | confirmation date | today (when you ran `--confirm`) |
| `…→ dwc:identificationVerificationStatus` | `verified using recorded media` | fixed value (camtrap convention for human-verified-from-photo IDs) |
| `…→ dwc:verbatimIdentification` | the species string you typed | your `--confirm` argument |
| `dwc:Taxon` → `dwc:scientificName` / `dwc:vernacularName` | *Parus major* / Great Tit | eBird taxonomy |
| `…→ dwc:kingdom`…`dwc:class` | Animalia / Chordata / Aves | constant for birds |
| `…→ dwc:order` / `dwc:family` | e.g. Passeriformes / Paridae | Clements v2026 checklist (omitted if unavailable) |
| `…→ dwc:genus` / `dwc:specificEpithet` | Parus / major | from the scientific name |

`dwc:identifiedBy` is **deliberately not written**: birdid has no record of who
assigned the taxon, so it never invents an identifier — that element only
appears if you add it yourself. `dwc:recordedBy` is likewise only written when
the photo actually carries an `Artist`.

This mirror does **not** change the freeze guarantee: `bird:confirmed` remains
the authority, and the dwc block is written/refreshed only by `--confirm` and
removed by `--unconfirm`. ExifTool shows it out of the box:

```bash
exiftool -G1 -XMP-dwc:all IMG_1234.xmp
# [XMP-dwc]  Identification Verification Status: verified using recorded media
# [XMP-dwc]  Taxon Vernacular Name           : Great Tit
# ...
```

## User confirmation (`bird:confirmed`)

birdid never deletes or second-guesses an identification a human has verified.
To mark a capture's record as **user-confirmed**, set the `bird:confirmed`
element to the species (e.g. `bird:confirmed = "Great Tit"`). Once present,
birdid treats the sidecar as **frozen**:

- It is never re-written — not on a re-run, not with `--force`, not by a review
  pass or burst mode. The identification you confirmed is the final one.
- The pipeline's CLI sets it for you (see `pipeline/README.md` → *User
  confirmation*): `birdid --confirm "Great Tit" <folder>`, or per photo with
  `--photo`. `birdid --unconfirm <folder>` lifts the freeze if you want birdid
  to re-identify later.
- Your photo software can simply *show* the flag; it never has to write it. Any
  tool that can edit the sidecar can set/clear `bird:confirmed` (a future
  birdid viewer does this through the same CLI).

> The freeze is on purpose: it is the guarantee that your manual verdict will
> never be clobbered by a later automated run. `bird:confirmed` carries the
> confirmed species, so the record stays self-describing ("the user verified
> *this* species").

---

## 1. ExifTool (any OS)

Copy [`birdid.config`](birdid.config) to your ExifTool config file:

- **Linux/macOS:** `~/.ExifTool_config`
- **Windows:** `C:\Users\<you>\.ExifTool_config`

ExifTool auto-loads it, so no flags are needed afterwards:

```bash
exiftool -G1 -bird:all IMG_1234.xmp
# [XMP-bird]  TopSpecies  : Swinhoe's White-eye
# [XMP-bird]  ScientificName : Zosterops simplex
# ...
```

Without the file ExifTool shows the same tags as `[XMP-ns0]` (its generic
name for an unregistered namespace) — the data is identical, only the label
and group name change.

## 2. IMatch (Windows)

IMatch reads and writes metadata **through its bundled ExifTool**, so the same
config file makes IMatch show the `bird:` tags with friendly names. From the
official IMatch help (*Technical Information → ExifTool*):

> Since version 5.4.6 IMatch also checks the last modified timestamp of the
> `imatch_et.config` file. If the timestamp of this file is newer than
> recorded in the database, IMatch re-imports all ExifTool tags and groups.
> This allows users which want to support custom XMP metadata schemas to force
> a re-load of modified tags by simply 'touching' the `imatch_et.config` file.

So:

1. Locate IMatch's ExifTool config file **`imatch_et.config`** (it lives with
   IMatch's bundled ExifTool in the IMatch program folder; the IMatch help
   topic above is the authoritative reference).
2. Append the two hash blocks from [`birdid.config`](birdid.config)
   (`%Image::ExifTool::UserDefined` and
   `%Image::ExifTool::UserDefined::bird`, plus the final `1;`).
3. "Touch" the file so IMatch re-imports the tags into its database. IMatch
   then shows the fields in its **Metadata Browser**, where you can add them to
   a **custom Metadata panel**.
   - **Restart needed for the Tag Selector:** touching the config only updates
     the metadata database. To see the group in the **Tag Selector / Tag
     Manager** with its proper names, fully restart IMatch — a machine reboot
     also works (observed behaviour).
   - The tags then live under the tag group **`UserDefined::bird`** — that's
     where they appear in the Tag Selector, and how they show up in panel
     variables (`{File.MD.UserDefined::bird\…}`).
4. Because IMatch stores imported metadata in its database, you can also use
   these tags for **searching, sorting, collections and data-driven
   categories**.

## 3. Adobe Lightroom / Photoshop

Adobe's programs read the standard XMP namespaces, and when they rewrite a
sidecar they keep a custom namespace such as `bird` **untouched**. They have no
user-extensible namespace registry, though, so the bird fields are **not shown**
in Lightroom's metadata panel. Practically that means:

- The tags survive a Lightroom save. Verify once after the first round-trip
  that the block is still there: `exiftool -G1 -bird:all IMG_1234.xmp`.
- To *see* the values, use ExifTool, IMatch, or open the `.xmp` in a text/XML
  editor — Lightroom itself will not display them.
- The tags survive because birdid keeps the namespace self-contained; by
  default it never touches the fields Lightroom manages (`dc:subject`,
  `dc:description`, … — see the `--dc-fields` section above), so a Lightroom
  save cannot clobber the identification results either.

## 4. Other DAMs (Photo Mechanic, Digikam, XnViewMP …)

The general rule: **any** program that lets you register a custom XMP namespace
(IMatch, Photo Mechanic, Digikam…) can be pointed at
`http://ns.birdpipeline/1.0/` — that makes the tags first-class citizens with a
stable `bird:` prefix. Programs *without* a registry (XnViewMP, many simple
viewers) simply ignore unknown namespaces or list them under an auto-generated
prefix (`ns0:`); the data is still there and visible on an XMP tab or via
ExifTool.

One caution: birdid always writes the namespace under the stable `bird:`
prefix, but a tool that re-serialises the sidecar through a bare ExifTool
*without* the `birdid.config` (Section 1) will rename the prefix to an
auto-generated `ns0:`/`ns1:`… — the data is unchanged, only the label moves.
If you ever see `XMP-ns0`/`ns0:` where `XMP-bird`/`bird:` used to be, a tool
re-serialised the file without the config loaded. birdid's own re-runs restore
the `bird:` prefix automatically.

For plain XML viewing, open the `.xmp` in any text/XML editor and look for the
`xmlns:ns0="http://ns.birdpipeline/1.0/"` elements.
