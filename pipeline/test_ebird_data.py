#!/usr/bin/env python3
"""Regression tests for the offline-safe, pure parts of ebird_data.py.

No network is touched: every tested function works off the bundled reference
data (ebird/continents.csv, ebird/countries.csv) or pure math (sampling dates,
month masks). Anything that calls the eBird API (http_get_json, historic_obs,
sample_months) is deliberately NOT tested here.

Pins the contracts --download and --update rely on:
  * country/continent/subcontinent resolution (region/continent levels)
  * resolve_spec token forms (1, 2 and 3 values)
  * sampling-date math + the in-progress-year fallback
  * the 12-char month mask and the region-table header shape
"""
import sys
import tempfile
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import ebird_data  # noqa: E402  (sys.path must be set first)
import birdid  # noqa: E402  (POOLS etc. live on the birdid side too)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS: {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL: {name} {detail}")


def run():
    _HERE_DATA = _HERE.parent / "ebird"
    have_ref = (_HERE_DATA / "continents.csv").is_file()
    if not have_ref:
        print("SKIP test_ebird_data.py: bundled ebird/continents.csv not present")
        return 0

    countries = ebird_data.load_countries(_HERE_DATA)
    check("load_countries covers 2-letter ISO codes",
          len(countries) > 100 and "DE" in countries and "SG" in countries
          and "US" in countries, f"n={len(countries)}")
    check("country names parsed",
          countries["DE"] == "Germany" and countries["SG"] == "Singapore")

    print("\n== E1: country -> continent/subcontinent resolution ==")
    c2c = ebird_data.country_continent_map()
    c2s = ebird_data.country_subcontinent_map()
    check("DE -> EUROPE (continent)",
          c2c.get("DE") == "EUROPE", f"got {c2c.get('DE')}")
    check("DE -> EU-WEST (subcontinent)",
          c2s.get("DE") == "EU-WEST", f"got {c2s.get('DE')}")
    check("every mapped subcontinent has a continent",
          all(ebird_data.subcontinent_continent_map().get(s)
              for s in set(c2s.values())), str(set(c2s.values())))
    code, label = ebird_data._region_level_for("DE", "SUBCONTINENT", countries)
    check("region level DE subcontinent resolves to EU-WEST",
          code == "EU-WEST" and label == ebird_data.SUB_NAMES["EU-WEST"],
          f"{code} {label}")
    code, label = ebird_data._region_level_for("DE", "CONTINENT", countries)
    check("region level DE continent resolves to EUROPE",
          code == "EUROPE", f"{code} {label}")
    try:
        ebird_data._region_level_for("XX", "CONTINENT", countries)
        check("unknown country raises ValueError", False, "no exception")
    except ValueError:
        check("unknown country raises ValueError", True)

    print("\n== E2: resolve_spec token forms ==")
    # 1 value
    kind, code, label = ebird_data.resolve_spec(["SG"], countries)
    check("bare code SG -> country",
          kind == "country" and code == "SG" and label == "Singapore",
          f"{kind} {code} {label}")
    kind, code, label = ebird_data.resolve_spec(["SEA"], countries)
    check("pool code SEA -> region",
          kind == "region" and code == "SEA", f"{kind} {code}")
    kind, code, label = ebird_data.resolve_spec(["AS-SOUTHEAST"], countries)
    check("subcontinent alias AS-SOUTHEAST -> SEA pool",
          kind == "region" and code == "SEA", f"{kind} {code}")
    kind, code, label = ebird_data.resolve_spec(["TAXONOMY"], countries)
    check("TAXONOMY -> taxonomy",
          kind == "taxonomy" and code == "", f"{kind} {code}")
    kind, code, label = ebird_data.resolve_spec(["COUNTRY"], countries)
    check("bare COUNTRY -> default region DE",
          kind == "country" and code == ebird_data.DEFAULT_REGION
          and label == ebird_data.COUNTRIES[code]["name"], f"{kind} {code}")
    kind, code, label = ebird_data.resolve_spec(["REGION"], countries)
    check("bare REGION -> default country's subcontinent pool",
          kind == "region" and code == "EU-WEST", f"{kind} {code}")
    # 2 values
    kind, code, label = ebird_data.resolve_spec(["country", "DE"], countries)
    check("country DE", kind == "country" and code == "DE"
          and label == ebird_data.COUNTRIES["DE"]["name"])
    kind, code, label = ebird_data.resolve_spec(["region", "continent"], countries)
    check("region continent (default country's continent)",
          kind == "region" and code == "EUROPE", f"{kind} {code}")
    kind, code, label = ebird_data.resolve_spec(["region", "SEA"], countries)
    check("region SEA", kind == "region" and code == "SEA")
    # 3 values
    kind, code, label = ebird_data.resolve_spec(["region", "subcontinent", "US"], countries)
    check("region subcontinent US -> its subcontinent pool",
          kind == "region" and code == c2s.get("US"), f"{kind} {code}")
    kind, code, label = ebird_data.resolve_spec(["region", "continent", "SG"], countries)
    check("region continent SG -> its continent pool",
          kind == "region" and code == c2c.get("SG"), f"{kind} {code}")
    # errors
    for bad in (["ZZZZ"], ["frobnicate", "SG"], ["region", "nonsense"],
                ["country", "DE", "extra"]):
        try:
            ebird_data.resolve_spec(bad, countries)
            check(f"resolve_spec({bad}) raises", False, "no exception")
        except ValueError:
            check(f"resolve_spec({bad}) raises", True)

    print("\n== E3: sampling-date math ==")
    # formula: d = round((i+1)*(last+1)/(n+1)), clamped to the month's last day
    days = ebird_data.sample_dates_for_month(2026, 11, 2)
    check("2 dates in November spread through the month",
          days == [(2026, 11, 10), (2026, 11, 21)], f"{days}")
    feb = ebird_data.sample_dates_for_month(2026, 2, 2)
    check("February 2026 respects 28 days", feb[-1][2] <= 28, f"{feb}")
    leap = ebird_data.sample_dates_for_month(2024, 2, 2)
    check("February 2024 (leap) respects 29 days", leap[-1][2] <= 29, f"{leap}")
    n = ebird_data.sample_dates_for_month(2026, 1, 1)
    check("single date in January", n == [(2026, 1, 16)], f"{n}")

    print("\n== E4: in-progress-year fallback (sampling_year_for_month) ==")
    # On 2026-11-20: Jan-Oct 2026 are complete -> use 2026; Nov-Dec not yet ->
    # fall back to 2025. Mirrors the documented example.
    today = date(2026, 11, 20)
    for m, want in [(1, 2026), (10, 2026), (11, 2025), (12, 2025)]:
        got = ebird_data.sampling_year_for_month(2026, m, today)
        check(f"month {m} on {today} samples {want}",
              got == want, f"got {got}")
    check("a fully-past year uses the target year everywhere",
          all(ebird_data.sampling_year_for_month(2025, m, date(2026, 1, 2)) == 2025
              for m in range(1, 13)))

    print("\n== E5: month mask + region-table header shape ==")
    check("_mask empty = 12 zeros",
          ebird_data._mask(set()) == "000000000000")
    check("_mask all = 12 ones",
          ebird_data._mask(set(range(1, 13))) == "111111111111")
    check("_mask sparse (Jan, Dec)",
          ebird_data._mask({1, 12}) == "100000000001")
    check("_mask ignores out-of-range",
          ebird_data._mask({0, 13, 5}) == "000010000000")
    # write_region_csv writes a parseable CSV with the data_date + mask headers.
    with tempfile.TemporaryDirectory(prefix="birdid-ebird-") as tmp:
        out = Path(tmp) / "DE.csv"
        ebird_data.write_region_csv(out, "DE", {"Coal Tit": {1, 12}},
                                    "Germany", 2026, 2, members=["DE"],
                                    data_date="2026-11-20")
        text = out.read_text()
        check("region CSV has the data_date header",
              "# data_date: 2026-11-20" in text, text.splitlines()[0] if text else "")
        check("region CSV has a species row with a 12-char mask",
              "DE,Coal Tit,seasonal,100000000001" in text,
              "\n".join(text.splitlines()[-3:]))
        check("region CSV has a resident status for 12-month species",
              "# month_mask: 12 chars" in text, "header missing month_mask")
        # load_countries caches per data_dir (idempotent second call).
        again = ebird_data.load_countries(_HERE_DATA)
        check("load_countries is cached per data_dir",
              again is countries)

    print("\n============================================")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
