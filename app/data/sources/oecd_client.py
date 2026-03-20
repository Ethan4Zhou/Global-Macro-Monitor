"""OECD API client for optional macro proxy series."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import os

import pandas as pd
import requests

OECD_DEFAULT_API_BASE = "https://sdmx.oecd.org/public/rest/data"
NORMALIZED_COLUMNS = [
    "date",
    "value",
    "series_id",
    "country",
    "source",
    "frequency",
    "release_date",
    "ingested_at",
]


def _ingested_at() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def fetch_oecd_series(
    source_series_id: str,
    country: str,
    frequency: str,
    api_base: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch one OECD series and return a normalized DataFrame."""
    base = api_base or os.getenv("OECD_API_BASE") or OECD_DEFAULT_API_BASE
    url = f"{base.rstrip('/')}/{source_series_id.lstrip('/')}"
    response = requests.get(url, params={"format": "csvfile"}, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    date_column = "TIME_PERIOD" if "TIME_PERIOD" in frame.columns else "date"
    value_column = "OBS_VALUE" if "OBS_VALUE" in frame.columns else "value"
    if date_column not in frame.columns or value_column not in frame.columns:
        raise ValueError("OECD response missing required date/value columns.")
    normalized = frame.rename(columns={date_column: "date", value_column: "value"}).loc[:, ["date", "value"]]
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized["series_id"] = source_series_id
    normalized["country"] = country
    normalized["source"] = "oecd"
    normalized["frequency"] = frequency
    normalized["release_date"] = normalized["date"]
    normalized["ingested_at"] = _ingested_at()
    normalized = normalized.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return normalized.loc[:, NORMALIZED_COLUMNS]
