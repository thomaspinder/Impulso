"""Fetch the supply-chain SVAR dataset from FRED and the NY Fed.

Series:
    gscpi   - NY Fed Global Supply Chain Pressure Index (monthly)
    oil     - WTISPLC, WTI spot crude ($/bbl, monthly avg)
    output  - INDPRO, industrial production index
    ppi     - PPIACO, producer price index (all commodities)
    prices  - CPIAUCSL, consumer price index (all urban)
    rate    - DGS2, 2-year US Treasury constant maturity yield (%)

Requires ``FRED_API_KEY`` in the environment for the FRED series. GSCPI is
fetched directly from the NY Fed's public xlsx download; FRED does not mirror
it.

Writes ``docs/tutorials/data/supply_chain.csv`` and a provenance record.

The NY Fed serves GSCPI in legacy .xls (OLE2) format, so ``xlrd`` is needed
for parsing. ``xlrd`` is not in the project's locked dependencies, so run
this script with an ephemeral install:

    uv run --with xlrd python scripts/build_supply_chain_data.py
"""

from __future__ import annotations

import os
import pathlib
from datetime import date
from io import BytesIO

import pandas as pd
import requests

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = {
    "oil": "WTISPLC",
    "output": "INDPRO",
    "ppi": "PPIACO",
    "prices": "CPIAUCSL",
    "rate": "DGS2",
}
GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
START = "1997-09-01"
OUT_CSV = pathlib.Path("docs/tutorials/data/supply_chain.csv")
OUT_PROV = pathlib.Path("docs/tutorials/data/supply_chain_provenance.md")


def fetch_fred(code: str, api_key: str) -> pd.Series:
    r = requests.get(
        FRED_URL,
        params={
            "series_id": code,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": START,
        },
        timeout=30,
    )
    r.raise_for_status()
    obs = pd.DataFrame(r.json()["observations"])
    obs["date"] = pd.to_datetime(obs["date"])
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    return obs.set_index("date")["value"].sort_index().resample("MS").mean()


def fetch_gscpi() -> pd.Series:
    r = requests.get(
        GSCPI_URL,
        headers={
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36"),
            "Referer": "https://www.newyorkfed.org/research/policy/gscpi",
        },
        timeout=30,
    )
    r.raise_for_status()
    df = pd.read_excel(
        BytesIO(r.content),
        sheet_name="GSCPI Monthly Data",
        engine="xlrd",
        header=None,
        usecols=[0, 1],
        skiprows=5,
        names=["date", "gscpi"],
    ).dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["gscpi"].sort_index().resample("MS").mean()


def main() -> None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit(
            "FRED_API_KEY is not set. Obtain a free key from https://fredaccount.stlouisfed.org and export it."
        )

    series = {name: fetch_fred(code, api_key) for name, code in FRED_SERIES.items()}
    series["gscpi"] = fetch_gscpi()
    df = pd.concat(series, axis=1)[["gscpi", "oil", "output", "ppi", "prices", "rate"]].dropna().loc["1998-01-01":]
    df.index.name = "date"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, float_format="%.6f")

    OUT_PROV.write_text(
        "# Supply-chain VAR data - provenance\n\n"
        f"Pulled: {date.today().isoformat()}\n"
        f"Sample: {df.index.min().date()} - {df.index.max().date()}\n"
        f"Rows: {len(df)}\n\n"
        "## Sources\n\n"
        "| Column | Source | Code / URL |\n"
        "|--------|--------|------------|\n"
        f"| gscpi  | NY Fed direct download | {GSCPI_URL} |\n"
        "| oil    | FRED API | WTISPLC |\n"
        "| output | FRED API | INDPRO |\n"
        "| ppi    | FRED API | PPIACO |\n"
        "| prices | FRED API | CPIAUCSL |\n"
        "| rate   | FRED API | DGS2 |\n\n"
        "- FRED API fetches require a free API key in `FRED_API_KEY` env var\n"
        "  (https://fredaccount.stlouisfed.org).\n"
        "- GSCPI is not mirrored on FRED; fetched directly from the NY Fed's public xlsx.\n"
        "- Monthly alignment at month-start. DGS2 daily values averaged to month.\n"
        "- GSCPI source of record: https://www.newyorkfed.org/research/policy/gscpi\n"
    )


if __name__ == "__main__":
    main()
