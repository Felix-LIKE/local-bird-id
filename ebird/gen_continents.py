#!/usr/bin/env python3
"""Generate ebird/continents.csv: country code -> continent, subcontinent.

Standard geographic grouping (country codes are ISO-3166 alpha-2 as used in
the bundled ebird/countries.csv). Pure geography - NOT eBird API data, safe to
commit. Subcontinent codes with a dash are our own pool codes; 'SEA' and
'EUROPE' reuse the existing top-level pool codes.
"""
import csv
import sys
from pathlib import Path

# code -> (continent, subcontinent)
MAP = {
    # ---------------- Europe (continent pool EUROPE) ----------------
    # Northern Europe
    "NO": ("EUROPE", "EU-NORTH"), "SE": ("EUROPE", "EU-NORTH"),
    "FI": ("EUROPE", "EU-NORTH"), "DK": ("EUROPE", "EU-NORTH"),
    "IS": ("EUROPE", "EU-NORTH"), "FO": ("EUROPE", "EU-NORTH"),
    "EE": ("EUROPE", "EU-NORTH"), "LV": ("EUROPE", "EU-NORTH"),
    "LT": ("EUROPE", "EU-NORTH"), "AX": ("EUROPE", "EU-NORTH"),
    "SJ": ("EUROPE", "EU-NORTH"),
    # Western / Central Europe
    "GB": ("EUROPE", "EU-WEST"), "IE": ("EUROPE", "EU-WEST"),
    "FR": ("EUROPE", "EU-WEST"), "BE": ("EUROPE", "EU-WEST"),
    "NL": ("EUROPE", "EU-WEST"), "LU": ("EUROPE", "EU-WEST"),
    "DE": ("EUROPE", "EU-WEST"), "AT": ("EUROPE", "EU-WEST"),
    "CH": ("EUROPE", "EU-WEST"), "LI": ("EUROPE", "EU-WEST"),
    "MC": ("EUROPE", "EU-WEST"), "PT": ("EUROPE", "EU-WEST"),
    "ES": ("EUROPE", "EU-WEST"), "AD": ("EUROPE", "EU-WEST"),
    "GI": ("EUROPE", "EU-WEST"), "IM": ("EUROPE", "EU-WEST"),
    "GG": ("EUROPE", "EU-WEST"), "JE": ("EUROPE", "EU-WEST"),
    # Southern / Southeast Europe
    "IT": ("EUROPE", "EU-SOUTH"), "GR": ("EUROPE", "EU-SOUTH"),
    "CY": ("EUROPE", "EU-SOUTH"), "MT": ("EUROPE", "EU-SOUTH"),
    "SM": ("EUROPE", "EU-SOUTH"), "VA": ("EUROPE", "EU-SOUTH"),
    "SI": ("EUROPE", "EU-SOUTH"), "HR": ("EUROPE", "EU-SOUTH"),
    "BA": ("EUROPE", "EU-SOUTH"), "ME": ("EUROPE", "EU-SOUTH"),
    "MK": ("EUROPE", "EU-SOUTH"), "AL": ("EUROPE", "EU-SOUTH"),
    "RS": ("EUROPE", "EU-SOUTH"), "XK": ("EUROPE", "EU-SOUTH"),
    # Eastern Europe
    "PL": ("EUROPE", "EU-EAST"), "CZ": ("EUROPE", "EU-EAST"),
    "SK": ("EUROPE", "EU-EAST"), "HU": ("EUROPE", "EU-EAST"),
    "RO": ("EUROPE", "EU-EAST"), "BG": ("EUROPE", "EU-EAST"),
    "MD": ("EUROPE", "EU-EAST"), "UA": ("EUROPE", "EU-EAST"),
    "BY": ("EUROPE", "EU-EAST"),
    # ---------------- North America ----------------
    "CA": ("NA", "NA-NORTH"), "US": ("NA", "NA-NORTH"),
    "MX": ("NA", "NA-NORTH"), "GL": ("NA", "NA-NORTH"),
    "BM": ("NA", "NA-NORTH"), "PM": ("NA", "NA-NORTH"),
    "GT": ("NA", "NA-CENTRAL"), "BZ": ("NA", "NA-CENTRAL"),
    "SV": ("NA", "NA-CENTRAL"), "HN": ("NA", "NA-CENTRAL"),
    "NI": ("NA", "NA-CENTRAL"), "CR": ("NA", "NA-CENTRAL"),
    "PA": ("NA", "NA-CENTRAL"),
    "AG": ("NA", "NA-CARIBBEAN"), "AI": ("NA", "NA-CARIBBEAN"),
    "AW": ("NA", "NA-CARIBBEAN"), "BB": ("NA", "NA-CARIBBEAN"),
    "BL": ("NA", "NA-CARIBBEAN"), "BQ": ("NA", "NA-CARIBBEAN"),
    "BS": ("NA", "NA-CARIBBEAN"), "CU": ("NA", "NA-CARIBBEAN"),
    "CW": ("NA", "NA-CARIBBEAN"), "DM": ("NA", "NA-CARIBBEAN"),
    "DO": ("NA", "NA-CARIBBEAN"), "GD": ("NA", "NA-CARIBBEAN"),
    "GP": ("NA", "NA-CARIBBEAN"), "HT": ("NA", "NA-CARIBBEAN"),
    "JM": ("NA", "NA-CARIBBEAN"), "KN": ("NA", "NA-CARIBBEAN"),
    "KY": ("NA", "NA-CARIBBEAN"), "LC": ("NA", "NA-CARIBBEAN"),
    "MF": ("NA", "NA-CARIBBEAN"), "MQ": ("NA", "NA-CARIBBEAN"),
    "MS": ("NA", "NA-CARIBBEAN"), "PR": ("NA", "NA-CARIBBEAN"),
    "SX": ("NA", "NA-CARIBBEAN"), "TC": ("NA", "NA-CARIBBEAN"),
    "TT": ("NA", "NA-CARIBBEAN"), "VC": ("NA", "NA-CARIBBEAN"),
    "VG": ("NA", "NA-CARIBBEAN"), "VI": ("NA", "NA-CARIBBEAN"),
    # ---------------- South America ----------------
    "CO": ("SA", "SA-NORTH"), "VE": ("SA", "SA-NORTH"),
    "EC": ("SA", "SA-NORTH"), "PE": ("SA", "SA-NORTH"),
    "GY": ("SA", "SA-NORTH"), "SR": ("SA", "SA-NORTH"),
    "GF": ("SA", "SA-NORTH"),
    "BR": ("SA", "SA-BRAZIL"),
    "AR": ("SA", "SA-SOUTH"), "BO": ("SA", "SA-SOUTH"),
    "CL": ("SA", "SA-SOUTH"), "PY": ("SA", "SA-SOUTH"),
    "UY": ("SA", "SA-SOUTH"), "FK": ("SA", "SA-SOUTH"),
    # ---------------- Africa ----------------
    "DZ": ("AF", "AF-NORTH"), "EG": ("AF", "AF-NORTH"),
    "EH": ("AF", "AF-NORTH"), "LY": ("AF", "AF-NORTH"),
    "MA": ("AF", "AF-NORTH"), "TN": ("AF", "AF-NORTH"),
    "SD": ("AF", "AF-NORTH"),
    "BJ": ("AF", "AF-WEST"), "BF": ("AF", "AF-WEST"),
    "CV": ("AF", "AF-WEST"), "CI": ("AF", "AF-WEST"),
    "GH": ("AF", "AF-WEST"),
    "GM": ("AF", "AF-WEST"), "GN": ("AF", "AF-WEST"),
    "GW": ("AF", "AF-WEST"), "LR": ("AF", "AF-WEST"),
    "ML": ("AF", "AF-WEST"), "MR": ("AF", "AF-WEST"),
    "NE": ("AF", "AF-WEST"), "NG": ("AF", "AF-WEST"),
    "SN": ("AF", "AF-WEST"), "SL": ("AF", "AF-WEST"),
    "TG": ("AF", "AF-WEST"),
    "CM": ("AF", "AF-CENTRAL"), "CF": ("AF", "AF-CENTRAL"),
    "TD": ("AF", "AF-CENTRAL"), "CG": ("AF", "AF-CENTRAL"),
    "CD": ("AF", "AF-CENTRAL"), "GA": ("AF", "AF-CENTRAL"),
    "GQ": ("AF", "AF-CENTRAL"), "ST": ("AF", "AF-CENTRAL"),
    "BI": ("AF", "AF-EAST"), "DJ": ("AF", "AF-EAST"),
    "ER": ("AF", "AF-EAST"), "ET": ("AF", "AF-EAST"),
    "KE": ("AF", "AF-EAST"), "RW": ("AF", "AF-EAST"),
    "SO": ("AF", "AF-EAST"), "SS": ("AF", "AF-EAST"),
    "TZ": ("AF", "AF-EAST"), "UG": ("AF", "AF-EAST"),
    "KM": ("AF", "AF-EAST"), "MG": ("AF", "AF-EAST"),
    "MU": ("AF", "AF-EAST"), "RE": ("AF", "AF-EAST"),
    "SC": ("AF", "AF-EAST"), "YT": ("AF", "AF-EAST"),
    "AO": ("AF", "AF-SOUTH"), "BW": ("AF", "AF-SOUTH"),
    "LS": ("AF", "AF-SOUTH"), "MW": ("AF", "AF-SOUTH"),
    "MZ": ("AF", "AF-SOUTH"), "NA": ("AF", "AF-SOUTH"),
    "SZ": ("AF", "AF-SOUTH"), "ZA": ("AF", "AF-SOUTH"),
    "ZM": ("AF", "AF-SOUTH"), "ZW": ("AF", "AF-SOUTH"),
    "SH": ("AF", "AF-SOUTH"),
    # ---------------- Asia ----------------
    "CN": ("AS", "AS-EAST"), "JP": ("AS", "AS-EAST"),
    "KP": ("AS", "AS-EAST"), "KR": ("AS", "AS-EAST"),
    "MN": ("AS", "AS-EAST"), "TW": ("AS", "AS-EAST"),
    "HK": ("AS", "AS-EAST"), "MO": ("AS", "AS-EAST"),
    "SG": ("AS", "AS-SOUTHEAST"), "MY": ("AS", "AS-SOUTHEAST"),
    "ID": ("AS", "AS-SOUTHEAST"), "TH": ("AS", "AS-SOUTHEAST"),
    "VN": ("AS", "AS-SOUTHEAST"), "PH": ("AS", "AS-SOUTHEAST"),
    "KH": ("AS", "AS-SOUTHEAST"), "LA": ("AS", "AS-SOUTHEAST"),
    "MM": ("AS", "AS-SOUTHEAST"), "BN": ("AS", "AS-SOUTHEAST"),
    "TL": ("AS", "AS-SOUTHEAST"),
    "IN": ("AS", "AS-SOUTH"), "PK": ("AS", "AS-SOUTH"),
    "BD": ("AS", "AS-SOUTH"), "LK": ("AS", "AS-SOUTH"),
    "NP": ("AS", "AS-SOUTH"), "BT": ("AS", "AS-SOUTH"),
    "MV": ("AS", "AS-SOUTH"), "AF": ("AS", "AS-SOUTH"),
    "IO": ("AS", "AS-SOUTH"),
    "KZ": ("AS", "AS-CENTRAL"), "KG": ("AS", "AS-CENTRAL"),
    "TJ": ("AS", "AS-CENTRAL"), "TM": ("AS", "AS-CENTRAL"),
    "UZ": ("AS", "AS-CENTRAL"),
    "TR": ("AS", "AS-WEST"), "GE": ("AS", "AS-WEST"),
    "AM": ("AS", "AS-WEST"), "AZ": ("AS", "AS-WEST"),
    "IR": ("AS", "AS-WEST"), "IQ": ("AS", "AS-WEST"),
    "SY": ("AS", "AS-WEST"), "LB": ("AS", "AS-WEST"),
    "IL": ("AS", "AS-WEST"), "PS": ("AS", "AS-WEST"),
    "JO": ("AS", "AS-WEST"), "SA": ("AS", "AS-WEST"),
    "YE": ("AS", "AS-WEST"), "OM": ("AS", "AS-WEST"),
    "AE": ("AS", "AS-WEST"), "QA": ("AS", "AS-WEST"),
    "KW": ("AS", "AS-WEST"), "BH": ("AS", "AS-WEST"),
    "RU": ("AS", "AS-NORTH"),
    # ---------------- Oceania ----------------
    "AU": ("OC", "OC-AUSTRALASIA"), "NZ": ("OC", "OC-AUSTRALASIA"),
    "CC": ("OC", "OC-AUSTRALASIA"), "CX": ("OC", "OC-AUSTRALASIA"),
    "NF": ("OC", "OC-AUSTRALASIA"),
    "PG": ("OC", "OC-MELANESIA"), "FJ": ("OC", "OC-MELANESIA"),
    "SB": ("OC", "OC-MELANESIA"), "VU": ("OC", "OC-MELANESIA"),
    "NC": ("OC", "OC-MELANESIA"),
    "FM": ("OC", "OC-MICRONESIA"), "MH": ("OC", "OC-MICRONESIA"),
    "PW": ("OC", "OC-MICRONESIA"), "NR": ("OC", "OC-MICRONESIA"),
    "KI": ("OC", "OC-MICRONESIA"), "GU": ("OC", "OC-MICRONESIA"),
    "MP": ("OC", "OC-MICRONESIA"), "UM": ("OC", "OC-MICRONESIA"),
    "WS": ("OC", "OC-POLYNESIA"), "TO": ("OC", "OC-POLYNESIA"),
    "PF": ("OC", "OC-POLYNESIA"), "CK": ("OC", "OC-POLYNESIA"),
    "NU": ("OC", "OC-POLYNESIA"), "TV": ("OC", "OC-POLYNESIA"),
    "TK": ("OC", "OC-POLYNESIA"), "WF": ("OC", "OC-POLYNESIA"),
    "AS": ("OC", "OC-POLYNESIA"), "PN": ("OC", "OC-POLYNESIA"),
    # ---------------- Antarctica / sub-Antarctic ----------------
    "AQ": ("AN", "AN"), "GS": ("AN", "AN"), "BV": ("AN", "AN"),
    "TF": ("AN", "AN"), "HM": ("AN", "AN"),
}

def main():
    countries = {}
    data_dir = Path(__file__).resolve().parent
    with open(data_dir / "countries.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip().upper()
            if code:
                countries[code] = row.get("name") or code
    missing = sorted(set(countries) - set(MAP))
    extra = sorted(set(MAP) - set(countries))
    if missing:
        print("MISSING from MAP:", missing, file=sys.stderr)
    if extra:
        print("EXTRA in MAP (not in countries.csv):", extra, file=sys.stderr)
    if missing or extra:
        sys.exit(1)
    with open(data_dir / "continents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "continent", "subcontinent"])
        for code in sorted(MAP):
            cont, sub = MAP[code]
            w.writerow([code, cont, sub])
    # summary
    from collections import Counter
    cc = Counter(v[0] for v in MAP.values())
    sc = Counter(v[1] for v in MAP.values())
    print("continents:", dict(cc))
    print("subcontinents:", dict(sc))
    print("total codes:", len(MAP))

if __name__ == "__main__":
    main()
