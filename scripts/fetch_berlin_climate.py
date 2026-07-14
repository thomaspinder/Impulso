"""Fetch the Berlin energy-climate VAR dataset from the Open-Meteo ERA5 archive.

Produces ``docs/tutorials/data/berlin_climate.csv``: monthly means (1980-2024) of
four daily ERA5 variables at Berlin (52.52 N, 13.41 E), consumed by the
``conjugate-var`` tutorial.

Data source: ERA5 reanalysis (Copernicus Climate Change Service / ECMWF), served by
Open-Meteo <https://open-meteo.com/>. Non-commercial use is free and needs no key.
This script uses the commercial ``customer-archive-api`` endpoint when
``OPEN_METEO_API_KEY`` is set in the environment, and otherwise falls back to the
free ``archive-api`` endpoint -- so it reproduces the committed CSV with or without
a key.

Columns (monthly mean of the daily aggregate):

  temperature    temperature_2m_mean      (deg C)       heating / cooling demand
  radiation      shortwave_radiation_sum  (MJ/m^2/day)  solar-PV potential
  wind           wind_speed_10m_mean      (km/h)        wind-power potential
  precipitation  precipitation_sum        (mm/day)      hydro / runoff

Re-run:  uv run python scripts/fetch_berlin_climate.py
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

LATITUDE, LONGITUDE = 52.52, 13.41
START_DATE, END_DATE = "1980-01-01", "2024-12-31"
DAILY_VARS = [
    "temperature_2m_mean",
    "shortwave_radiation_sum",
    "wind_speed_10m_mean",
    "precipitation_sum",
]
COLUMNS = {
    "temperature_2m_mean": "temperature",
    "shortwave_radiation_sum": "radiation",
    "wind_speed_10m_mean": "wind",
    "precipitation_sum": "precipitation",
}
FREE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
CUSTOMER_ENDPOINT = "https://customer-archive-api.open-meteo.com/v1/archive"
OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "data" / "berlin_climate.csv"


def _request_url() -> str:
    """Build the archive request, preferring the keyed endpoint when a key is set."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    }
    query = urllib.parse.urlencode(params)
    key = os.environ.get("OPEN_METEO_API_KEY")
    if key:
        return f"{CUSTOMER_ENDPOINT}?{query}&apikey={key}"
    return f"{FREE_ENDPOINT}?{query}"


def main() -> None:
    """Fetch daily ERA5 data, aggregate to monthly means, and write the CSV."""
    with urllib.request.urlopen(_request_url(), timeout=180) as response:  # noqa: S310
        payload = json.load(response)

    daily = pd.DataFrame(payload["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.set_index("time").astype(float)

    monthly = daily.resample("MS").mean().rename(columns=COLUMNS)
    monthly.index.name = "date"
    monthly = monthly[list(COLUMNS.values())]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    monthly.round(4).to_csv(OUTPUT)
    print(f"wrote {len(monthly)} monthly rows ({START_DATE}..{END_DATE}) -> {OUTPUT}")


if __name__ == "__main__":
    main()
