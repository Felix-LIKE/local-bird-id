# eBird data (offline, bird project)

What's here, where it comes from, and what needs an API key.

## Two kinds of data (and what needs a key)

The pipeline uses **two completely different** eBird data products:

1. **Region tables** (`region,species,status,month_mask`) - the "which bird can
   I see here, in this month" prior that narrows the classifier's candidate
   list. Two sources:
   - **curated seeds** (committed) - hand-written from field guides, no key;
   - **generated tables** (gitignored) - real observations from the keyed eBird
     API via `birdid --download <code>`, preferred over the seeds.
2. **Taxonomy** (the global common↔scientific name dictionary + Avibase ids) -
   a static reference, **never needs a key**.

| Data | No key | With a free API key |
|---|---|---|
| Taxonomy (names / scientific names / Avibase) | ✓ bundled & auto-download | ✓ |
| Curated region seeds (EUROPE, SEA) | ✓ | ✓ |
| Real region table for a specific country | ✗ | ✓ `--download <code>` |

**eBird API data may not be publicly redistributed.** Generated tables are
therefore written to the **gitignored** `regions_generated/` directory and are
never committed - the committed repo contains only the curated, non-API seeds.

## Files

| File | What it is | Source |
|---|---|---|
| `regions/EUROPE.csv`, `regions/SEA.csv` | **Committed curated seeds** - the only two no-key regions. Species-per-month pools (Europe / Southeast Asia) used as the GPS+time prior; a region without its own table falls back to its pool. | Curated from field guides (Svensson / Collins; Robson / Mackinnon & Phillipps / Wells). No eBird API data. |
| `regions_generated/` | **Gitignored** API-derived tables from `birdid --download <code>` - created locally, preferred over the seeds, never committed. The smallest division is stored in a continent/subcontinent tree: `regions_generated/<CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv` (a country with subnational tables becomes `<COUNTRY>/<SUB>.csv`), with a pooled `<REGION>.csv` kept alongside. | Real eBird API observations. |
| `eBird_Taxonomy_v2023.csv` | Full eBird/Clements taxonomy (~18.6k rows): canonical names + codes, used to normalise species names. **Not committed** (gitignored, re-fetchable). `curl -o eBird_Taxonomy_v2023.csv "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv"` (no key). | eBird API (keyless). |
| `eBird-Clements_Integrated_Checklist.csv` | **Committed** bundled Clements checklist (version-agnostic name). Offline taxonomy fallback (same 18,631 common→scientific pairs as v2023) and the source of the **Avibase ids** (`taxon concept ID` column, `https://avibase.ca/<id>`) for every top guess. To use a newer release, replace this file (see "Updating the taxonomy" below). | Cornell Clements checklist download. |
| `countries.csv` | Bundled ISO country reference (250 codes) for `--list downloadable` (offline). | ISO 3166. |
| `continents.csv` | **Committed** static continent/subcontinent reference (250 codes): `code,continent,subcontinent`. Builds the continent + subcontinent **pools** (NA/SA/AF/AS/OC, EU-NORTH, AS-SOUTHEAST, ...) for `--download region <code>` batch downloads and maps every country to its continent pool for the two-tier fallback. Pure standard geography - **not** eBird API data. | ISO 3166 + standard geographic groupings. |

## Region CSV schema

```
region,species,status,month_mask
SEA,Asian Openbill,resident,111111111111
EUROPE,Great Tit,resident,111111111111
```

- `month_mask`: 12 characters, Jan..Dec, `1` = species is plausibly present that month.
- `status`: `resident | summer | winter | passage` on curated tables;
  `resident | seasonal` on generated tables (informative only).
- Lines starting with `#` are comments (also `# --- section ---` separators).
- `region` and `species` use the **canonical eBird common names** so
  scientific-name lookup works for every row.

## Without an API key: EUROPE + SEA only

The repo ships exactly two curated seeds: **EUROPE** and **SEA**. No-key runs:

- a photo in Europe / Southeast Asia gets its pool as the candidate prior
  (Germany → EUROPE, Singapore/Thailand → SEA automatically);
- a photo anywhere else has **no prior** - generate a table for that country
  with `birdid --download <code>` (needs a key). The pipeline falls back to the
  default region's pool otherwise.

## Real data for any region (`birdid --download`)

1. Get a free eBird API key: `https://ebird.org/api/keygen` (one click).
2. Import it: `birdid --api <your-key>` (stored in `~/.config/birdid/ebird_api_key`, mode 600).
3. Download a table: `birdid --download SG` (or `--download country SG`,
   `--download region SEA`, `--download FR`, ...). Subnational codes work too
   (`--download country DE-BY`). `--download country SG --subnational` fetches
   Singapore's subnational1 regions instead and stores them as
   `regions_generated/AS/AS-SOUTHEAST/SG/<SUB>.csv`.

`--download` is **incremental**: a table that is already downloaded is skipped,
only the missing parts are fetched. So re-running `birdid --download region EU-WEST`
after a partial run only downloads the countries you don't have yet.

To refresh tables whose **eBird data is newer** (not just "filled in"), use
`birdid --update <code>`: every generated CSV records the date of the eBird
data it was built from (`# data_date:` - the newest observation date sampled,
NOT the date it was downloaded), and `--update` compares that against the
newest data the API serves (default target: the current year, so a table
sampled from 2025 data is refreshed with 2026 observations). Months not yet
complete in the target year fall back to the previous year's observations, so
updating mid-year never leaves December (or any future month) empty. Missing
tables are fetched too; `--year` pins a different target year.

After a `--download`/`--update` changes the species set, the central
species-embedding cache can be re-aligned with the same regions:
`birdid --cache clean <region>` prunes it down to what the refreshed tables can
need (default: all loaded regions - drop every species no table uses), or
`birdid --cache build <region>` precomputes the new region's species so the
next run encodes nothing on the fly. Both accept the same region codes as
`--download` (countries, subcontinents like `EU-WEST`/`AS-SOUTHEAST`, or
continents like `EUROPE`); `clean` needs no model or API key.

Batch-download a whole continent or subcontinent at once (from the bundled
`continents.csv` reference). **The smallest division is always written**: the
country files land in a continent/subcontinent folder tree, and a continent
download creates the subcontinent folders for every country on demand, with a
pooled `<REGION>.csv` kept alongside each download:

```bash
birdid --download region SA            # South America -> regions_generated/SA/<SUB>/<country>.csv
birdid --download region NA            # North America
birdid --download region AF            # Africa
birdid --download region AS            # Asia
birdid --download region OC            # Oceania
birdid --download region AS-SOUTHEAST  # 11 country files into regions_generated/AS/AS-SOUTHEAST/
birdid --download region EU-NORTH      # Northern Europe subcontinent
birdid --download region NA-CARIBBEAN  # Caribbean subcontinent
birdid --download region SA-BRAZIL     # Brazil (1 country)
```

Bare `birdid --download region` resolves to the **default country's
subcontinent** (DE → `EU-WEST` - the tightest pool); `--download region
continent` to its **continent** (DE → `EUROPE` - the widest). For a specific
country use `--download region subcontinent SG` (→ `SEA`) or
`--download region continent SG` (→ `AS`).

So e.g. `regions_generated/AS/AS-SOUTHEAST/` ends up containing `SG.csv`,
`MY.csv`, ... plus a pooled `AS-SOUTHEAST.csv`; downloading `region AS` later
fills `AS/AS-EAST/`, `AS/AS-WEST/`, ... without touching the existing folders.

`birdid --list region` shows every continent + subcontinent pool, how many
member countries each has, and which have a local table. `SEA` is the curated
name of `AS-SOUTHEAST` (same 11 countries); every other subcontinent
(e.g. `EU-NORTH`, `NA-CARIBBEAN`, `AS-EAST`, `AF-WEST`, `OC-POLYNESIA`) is its
own pool. Invalid member codes (ISO codes the eBird API doesn't know, or
territories without an eBird region) are skipped gracefully during the download.

How the real data is fetched:

- for each month it calls the **historic observations** endpoint
  (`GET /v2/data/obs/{regionCode}/historic/{y}/{m}/{d}` - the date is three path
  segments) on `--sample-dates` (default 2) dates per month of `--year`
  (default 2025; `--update` defaults to the **current year**), and
- if the target year is still in progress, any month that has not fully passed
  uses the **previous year's** observations instead (so December is never
  empty when you update in, say, November), and
- aggregates which species were actually reported in each month → writes
  `regions_generated/<CONTINENT>/<SUBCONTINENT>/<COUNTRY>.csv` (status
  `resident` if present all 12 sampled months, else `seasonal`).
- A country downloads that country; a pool (SEA / EUROPE / a subcontinent /
  a continent) unions the monthly presence of all its member countries and
  keeps the union as a pooled `<REGION>.csv` alongside the country files.
- Rate-limit guard: `--api-delay` seconds between calls (default 1.5).
- **Resilient to slow dates**: a single date the eBird API doesn't answer for
  (the historic endpoint can hang server-side on very high-volume dates, e.g.
  around Global Big Day in early May) is skipped after a few short retries with
  a warning, instead of aborting the whole download - the table is built from
  the dates that did respond.

Generated tables are **API-derived and gitignored** - they are preferred over
the seeds while present, but must never be committed (eBird API data may not be
redistributed). Remove one with `rm ebird/regions_generated/AS/AS-SOUTHEAST/SG.csv`.

`birdid --list downloadable` shows every eBird country (from the bundled ISO
reference) with its **pool hierarchy** `country -> subcontinent -> continent`
(e.g. `DE -> EU-WEST -> EUROPE`), so you can grep it to pick the subcontinent
to download (`birdid --download region EU-WEST` gets DE and its region) or the
whole continent (`region EUROPE`). With the API key it can also list a
country's subregions (`--list downloadable subnational1 DE`); any listed code
works with `birdid --download country <CODE>`. Every country belongs to a
subcontinent pool, and every subcontinent to a continent pool (from the
bundled `continents.csv`; `AS-SOUTHEAST` is shown as its curated alias `SEA`).

## Updating the taxonomy (newer Clements release)

The bundled `eBird-Clements_Integrated_Checklist.csv` is a snapshot of the
Clements/eBird integrated checklist (Oct 2026). Taxonomy changes every year -
for example the **2025** update split the Striated Heron (*Butorides striata*)
into Striated Heron, **Little Heron** (*B. atricapilla*) and Lava Heron
(*B. sundevalli*), so older checklists lump them as one species.

To use a newer release:

1. Get the latest **integrated checklist CSV** from
   `https://www.birds.cornell.edu/clementschecklist/download/`
   (the file is named e.g. `eBird-Clements_v2027_integrated-checklist-...csv`).
2. Save it **under the same standard name**
   `eBird-Clements_Integrated_Checklist.csv` in the data dir (`ebird/`).
3. That's it - the pipeline picks it up automatically (same name = no code
   change) for both scientific names and Avibase ids.

`birdid --download taxonomy` (keyless) does this for the current version
automatically: it tries Cornell, then the eBird API taxonomy; if the machine is
fully offline it prints the download link + the filename to save + where to put
it.

Note: `api.ebird.org/v2/ref/taxonomy/ebird` works **without** a key and serves
the older v2023 taxonomy (which the API's observation data is aligned to); most
other endpoints (recent/historic observations, region refs) return `403` without
one - `birdid --download` surfaces that as a clear "run `birdid --api
<your-key>`" message.
