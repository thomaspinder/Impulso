# Supply-chain VAR data - provenance

Pulled: 2026-04-23
Sample: 1998-01-01 - 2026-03-01
Rows: 338

## Sources

| Column | Source | Code / URL |
|--------|--------|------------|
| gscpi  | NY Fed direct download | https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx |
| oil    | FRED API | WTISPLC |
| output | FRED API | INDPRO |
| ppi    | FRED API | PPIACO |
| prices | FRED API | CPIAUCSL |
| rate   | FRED API | DGS2 |

- FRED API fetches require a free API key in `FRED_API_KEY` env var
  (https://fredaccount.stlouisfed.org).
- GSCPI is not mirrored on FRED; fetched directly from the NY Fed's public xlsx.
- Monthly alignment at month-start. DGS2 daily values averaged to month.
- GSCPI source of record: https://www.newyorkfed.org/research/policy/gscpi
