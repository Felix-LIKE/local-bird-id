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
`bird:` prefix / `XMP-bird` group, for the two tools most likely to matter to
you (ExifTool and IMatch), plus what to expect in XnViewMP.

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
| `bird:status` | `done` / `low-confidence` / `non-bird` / `error` |
| `bird:pipelineVersion`, `bird:region`, `bird:month`, `bird:datetime`, `bird:reasoning` | provenance of the ID |
| `bird:dcOwned`, `bird:error` | housekeeping |

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
3. "Touch" the file (or restart IMatch) so IMatch re-imports the tags into
   its database. IMatch then shows the fields in its **Metadata Browser** and
   you can add them to a **custom Metadata panel**.
4. Because IMatch stores imported metadata in its database, you can also use
   these tags for **searching, sorting, collections and data-driven
   categories**.

## 3. XnViewMP

XnViewMP bundles its own ExifTool and can display the `bird:` fields in
several places:

- **XMP tab of the Info panel** — lists *all* XMP tags, including the custom
  `bird:` namespace (this works out of the box; the tags are shown under their
  raw namespace label, e.g. `ns0:`).
- **Info panel → Favorites tab** — right-click any bird field on the XMP tab
  and choose **Add to Favorites panel**; the fields then stay visible in a
  persistent Favorites tab (works for any metadata, standard or custom).
- **Info Overlay** (the panel you get with **`I`** in the viewer) — the
  overlay text is a template (in `xnview.ini` under `[Preview] >
  infoTemplate`, or the Info settings dialog). It accepts
  `{exiftool:group:tag}` placeholders, so you can add a bird block such as:

  ```
  Bird: {exiftool:xmp-bird:topSpecies} ({exiftool:xmp-bird:scientificName}) {exiftool:xmp-bird:topConfidence}
  {exiftool:xmp-bird:summary}
  Status: {exiftool:xmp-bird:status}
  ```

  For `{exiftool:xmp-bird:...}` to resolve, ExifTool must know the namespace
  prefix `bird`. XnViewMP's bundled ExifTool reads the user config
  (`~/.ExifTool_config` on Linux/macOS, `C:\Users\<you>\.ExifTool_config` on
  Windows) just like the standalone tool, so installing
  [`birdid.config`](birdid.config) there gives XnViewMP the `xmp-bird` group
  name too. Without it the group is `xmp-ns0`
  (e.g. `{exiftool:xmp-ns0:topSpecies}`).

  If you enable the full ExifTool metadata import in
  `Tools > Settings > Metadata > Import > Exiftool` and run **Update
  metadata**, these overlay fields are served from XnViewMP's database instead
  of a per-image ExifTool call (faster on large folders).

For plain XML viewing, open the `.xmp` in any text/XML editor and look for the
`xmlns:ns0="http://ns.birdpipeline/1.0/"` elements.

---

## Registering the namespace as a *custom* namespace (the general option)

Any DAM that lets you register a custom XMP namespace (IMatch, Photo Mechanic,
Digikam… ) can be pointed at `http://ns.birdpipeline/1.0/`. That is the
"proper" way to make the tags first-class citizens with a stable prefix; the
ExifTool config above is the zero-config shortcut that works everywhere
ExifTool is involved.
