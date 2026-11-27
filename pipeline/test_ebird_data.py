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

    print("\n== E?: progress streaming (piped vs terminal) ==")
    # _sample_with_progress must emit one NEWLINE-terminated line per sample
    # when stdout is a pipe (the viewer's line-by-line reader would otherwise
    # hold \r-only progress invisible until the region finishes), and keep the
    # in-place \r counter on a real terminal.
    from io import StringIO
    from contextlib import redirect_stdout

    def fake_sample_months(region_code, key, year, sample_dates, progress, today=None):
        for done in range(1, 5):            # 4 samples, 0..100%
            progress(region_code, done, 4, date(year, 1, done))
        return {"s": 1}, date(year, 1, 1)

    orig_sample_months = ebird_data.sample_months
    ebird_data.sample_months = fake_sample_months
    try:
        piped = StringIO()                  # .isatty() -> False, like a viewer pipe
        with redirect_stdout(piped):
            ebird_data._sample_with_progress("DE", "k", 2026, 1)
        v = piped.getvalue()
        check("piped stdout: 4 newline-terminated lines (no \\r, no blank)",
              v.count("\n") == 4 and v.count("\r") == 0 and v.endswith("\n"),
              repr(v))
        check("piped stdout: progress lines show count + percent",
              "DE:   4/4 (100%)" in v and "DE:   1/4 ( 25%)" in v, repr(v))

        class TTYLike(StringIO):
            def isatty(self):
                return True
        term = TTYLike()
        with redirect_stdout(term):
            ebird_data._sample_with_progress("DE", "k", 2026, 1)
        tv = term.getvalue()
        check("tty stdout: in-place \\r counter (4 \\r, single trailing \\n)",
              tv.count("\r") == 4 and tv.count("\n") == 1, repr(tv))
    finally:
        ebird_data.sample_months = orig_sample_months

    # TUI.progress() (cache builds, embeddings, ...) must split the same way:
    # newline-terminated on a pipe, in-place \r on a terminal.
    tui = birdid.TUI(False)
    pbuf = StringIO()
    with redirect_stdout(pbuf):
        tui.progress("cache", 1, 4)
        tui.progress("cache", 4, 4)
    pv = pbuf.getvalue()
    check("TUI.progress piped: newline-terminated lines, no \\r",
          pv.count("\n") == 2 and "\r" not in pv, repr(pv))
    class TTYLike2(StringIO):
        def isatty(self):
            return True
    tv2 = TTYLike2()
    with redirect_stdout(tv2):
        tui.progress("cache", 1, 4)
        tui.progress("cache", 4, 4)
    check("TUI.progress tty: in-place \\r, newline only at completion",
          tv2.getvalue().count("\r") == 2 and tv2.getvalue().count("\n") == 1,
          repr(tv2.getvalue()))

    # The launcher must stream the pipeline unbuffered, or every print sits in
    # python's ~8 KB pipe buffer and the viewer sees nothing until a run ends.
    launcher = (_HERE / "birdid").read_text(encoding="utf-8")
    py_inv = launcher.find('"$PY" "$PIPELINE"')
    check("launcher exports PYTHONUNBUFFERED before running the pipeline",
          py_inv > 0 and "PYTHONUNBUFFERED" in launcher[:py_inv],
          "PYTHONUNBUFFERED not exported before the python invocation")

    print("\n== E6: continent-pool fallback for countries with no table ==")
    # A fresh install ships only the curated EUROPE + SEA seeds. A country with
    # no table of its own must fall back to its continent pool (DE -> EUROPE,
    # SG -> SEA) instead of being flagged no-data; a country with no pool
    # ancestor (e.g. BR) stays no-data. Once the country table IS downloaded,
    # the tighter tree chain (DE -> EU-WEST -> EUROPE) must be preserved.
    import shutil
    with tempfile.TemporaryDirectory(prefix="birdid-pool-") as tmp:
        d = Path(tmp)
        regs = d / "regions"; regs.mkdir()
        (d / "regions_generated").mkdir()
        for seed in ("EUROPE.csv", "SEA.csv"):
            shutil.copy(_HERE_DATA / "regions" / seed, regs / seed)
        fresh = birdid.SpeciesDB(regs, d / "regions_generated")
        fresh.load()
        check("fresh install loads only the curated pools",
              sorted(fresh.regions()) == ["EUROPE", "SEA"], str(fresh.regions()))
        check("DE falls back to the EUROPE pool",
              fresh.tier_codes("DE") == ["DE", "EUROPE"],
              str(fresh.tier_codes("DE")))
        check("SG falls back to the SEA pool",
              fresh.tier_codes("SG") == ["SG", "SEA"],
              str(fresh.tier_codes("SG")))
        check("BR (no pool ancestor) stays no-data",
              fresh.tier_codes("BR") == ["BR"], str(fresh.tier_codes("BR")))
        ladder = birdid._build_ladder(fresh, "DE", 5, 0, [], [])
        check("DE ladder is non-empty via the pool (not no-data)",
              sum(len(st) for _, st in ladder) > 0
              and [lbl for lbl, _ in ladder] == ["DE", "EUROPE"],
              str([(l, len(s)) for l, s in ladder]))
        ladder_br = birdid._build_ladder(fresh, "BR", 5, 0, [], [])
        check("BR ladder stays empty (still no-data)",
              sum(len(st) for _, st in ladder_br) == 0)

    # Full install: the downloaded tree (EU-WEST/DE.csv under EUROPE) keeps
    # the tighter chain; the pool chain must not displace existing ancestors.
    full = birdid.SpeciesDB(_HERE_DATA / "regions", _HERE_DATA / "regions_generated")
    full.load()
    check("full install keeps DE -> EU-WEST -> EUROPE",
          full.tier_codes("DE") == ["DE", "EU-WEST", "EUROPE"],
          str(full.tier_codes("DE")))
    check("full install keeps SG -> AS-SOUTHEAST -> AS",
          full.tier_codes("SG") == ["SG", "AS-SOUTHEAST", "AS"],
          str(full.tier_codes("SG")))
    # _wider_pool_chain: subcontinent, then continent (deduped).
    check("wider pool chain DE = subcontinent then continent",
          birdid._wider_pool_chain("DE") == ["EU-WEST", "EUROPE"],
          str(birdid._wider_pool_chain("DE")))

    print("\n== E7: installed_tables() excludes derived dir-union scaffolds ==")
    # 'installed tables' = what the panel shows and 'update all' refreshes.
    # Downloading DE writes only <CONT>/<SUB>/DE.csv; the EU-WEST / EUROPE
    # dirs are pure tree scaffolding and must NOT be reported as installed
    # (the old --status listed them, so 'update all' queued --update EU-WEST /
    # --update EUROPE and re-downloaded a whole subcontinent/continent).
    def write_table(path: Path, region: str, mask: str = "111111111111"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"region,species,status,month_mask\n"
            f"{region},Test Sparrow,resident,{mask}\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="birdid-inst-") as tmp:
        d = Path(tmp)
        regs = d / "regions"; regs.mkdir()
        gen = d / "regions_generated"; gen.mkdir()
        for seed in ("EUROPE.csv", "SEA.csv"):
            shutil.copy(_HERE_DATA / "regions" / seed, regs / seed)
        db = birdid.SpeciesDB(regs, gen); db.load()
        check("fresh install installed = curated seeds only",
              db.installed_tables() == ["EUROPE", "SEA"],
              str(db.installed_tables()))

        # download DE: one file, but the loader derives EU-WEST + EUROPE dirs
        write_table(gen / "EUROPE" / "EU-WEST" / "DE.csv", "DE")
        db2 = birdid.SpeciesDB(regs, gen); db2.load()
        check("DE.csv creates EU-WEST/EUROPE as queryable dir-unions",
              "EU-WEST" in db2.regions() and "EUROPE" in db2.regions(),
              str(db2.regions()))
        check("but installed_tables() = DE + curated seeds only (no scaffolds)",
              db2.installed_tables() == ["DE", "EUROPE", "SEA"],
              str(db2.installed_tables()))
        check("EU-WEST scaffold is NOT an installed table",
              "EU-WEST" not in db2.installed_tables())

        # a country split into subnational tables (CLI --subnational) still
        # counts as installed: <CONT>/<SUB>/<COUNTRY>/<SUB>.csv
        (gen / "EUROPE" / "EU-WEST" / "DE.csv").unlink()
        write_table(gen / "EUROPE" / "EU-WEST" / "DE" / "DE-BE.csv", "DE-BE")
        db3 = birdid.SpeciesDB(regs, gen); db3.load()
        check("subnational country dir counts as installed",
              "DE" in db3.installed_tables(), str(db3.installed_tables()))
        check("subnational country dir does not leak scaffolds",
              sorted(x for x in db3.installed_tables()
                     if x not in ("EUROPE", "SEA")) == ["DE", "DE-BE"],
              str(db3.installed_tables()))

        # an explicitly downloaded pool (pooled <SUB>.csv) IS installed
        write_table(gen / "EUROPE" / "EU-WEST" / "EU-WEST.csv", "EU-WEST")
        db4 = birdid.SpeciesDB(regs, gen); db4.load()
        check("explicit pooled subcontinent table counts as installed",
              "EU-WEST" in db4.installed_tables(),
              str(db4.installed_tables()))

        # AS collision: the Asia continent scaffold must NOT be reported as
        # installed even though 'AS' is also a country (American Samoa).
        write_table(gen / "AS" / "AS-SOUTHEAST" / "TH.csv", "TH")
        db5 = birdid.SpeciesDB(regs, gen); db5.load()
        inst5 = db5.installed_tables()
        check("Asia continent scaffold AS is NOT installed",
              "AS" not in inst5, str(inst5))
        check("Asia subcontinent scaffold AS-SOUTHEAST is NOT installed",
              "AS-SOUTHEAST" not in inst5, str(inst5))
        check("the TH country table IS installed",
              "TH" in inst5, str(inst5))

    print("\n== E8: --update passes the freshness target to country downloads ==")
    # Regression: --update re-sampled EVERY country on every call because
    # _run_download never supplied target_date to _download_country, so the
    # skip-if-current branch in _sample_or_skip never ran ("update all"
    # re-downloaded what it had just updated 30 minutes earlier). The target
    # must be computed and handed to country downloads (pools compute their
    # own internally; countries receive it from _run_download).
    calls = {}

    def _capture_country(generated_dir, code, label, year, sample_dates, key,
                         subnational, today=None, update=False, target_date=""):
        calls["args"] = dict(code=code, update=update, target_date=target_date)

    old_country = ebird_data._download_country
    ebird_data._download_country = _capture_country
    try:
        ebird_data._run_download(_HERE_DATA, ["DE"], None, 2, "KEY", False,
                                 update=True)
        got = calls.get("args") or {}
        check("--update country resolves to the country path",
              got.get("code") == "DE", str(got))
        check("--update country gets a non-empty freshness target",
              bool(got.get("target_date")),
              f"target_date={got.get('target_date')!r} (was '' -> always refreshed)")
        calls.clear()
        ebird_data._run_download(_HERE_DATA, ["DE"], None, 2, "KEY", False,
                                 update=False)
        got = calls.get("args") or {}
        check("--download (non-update) sends no target -> full sampling",
              got.get("target_date") == "",
              f"target_date={got.get('target_date')!r}")
    finally:
        ebird_data._download_country = old_country

    print("\n============================================")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
