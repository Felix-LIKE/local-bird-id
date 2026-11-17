# eBird data (offline, bird project)

What's here and where it comes from.

## Files

| File | What it is | Source |
|---|---|---|
| `eBird_Taxonomy_v2023.csv` | The full eBird/Clements taxonomy (~18.6k rows): canonical species names + eBird codes, used to normalise species names in the pipeline. **Not committed to git** (gitignored) - it is a downloaded data artifact. Regenerate with: `curl -o eBird_Taxonomy_v2023.csv "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv"` (no key needed). | Official eBird API. |
| `regions/DE.csv`, `regions/SG.csv`, `regions/TH.csv` | "Which species are present in which month" for each region — the GPS+time prior that narrows the vision model's candidate list. | Curated seed lists (field-guide knowledge). To be replaced/upgraded with real eBird observation data when an API key is available (see below). |

## Region CSV schema

```
region,species,status,month_mask
DE,Great Tit,resident,111111111111
DE,Barn Swallow,summer,001111111000
```

- `month_mask`: 12 characters, Jan..Dec, `1` = species is plausibly present that month.
- `status`: `resident | summer | winter | passage` (informative only).
- Lines starting with `#` are comments.

## Upgrade to real eBird data (recommended)

The curated lists cover only common species. To replace them with real data:

1. Get a free eBird API key while logged in: `https://ebird.org/api/keygen` (one click).
2. Give the key to the pipeline / an agent session. It will:
   - pull **historic observations** for each region on sample dates per month
     (`GET /v2/data/obs/{regionCode}/historic/{date}`), and
   - aggregate which species were actually reported in each month → regenerate
     the `regions/*.csv` tables from real observations.

Note: `api.ebird.org/v2/ref/taxonomy/ebird` works **without** a key; most other
endpoints (recent/historic observations, region refs) return `403` without one.
