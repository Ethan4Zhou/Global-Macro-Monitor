"""Eurostat API client for Eurozone macro series."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import pandas as pd
import requests

EUROSTAT_DEFAULT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
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


def _normalize_time_label(label: str) -> str:
    """Convert Eurostat time labels into parseable dates."""
    if "M" in label and "-" not in label:
        year, month = label.split("M", maxsplit=1)
        return f"{year}-{month}-01"
    if "Q" in label:
        year, quarter = label.split("Q", maxsplit=1)
        month = {"1": "01", "2": "04", "3": "07", "4": "10"}.get(quarter, "01")
        return f"{year}-{month}-01"
    return label


def _parse_eurostat_json(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a simplified Eurostat JSON-stat response."""
    dimension = payload.get("dimension", {})
    time_dimension = dimension.get("time", {}).get("category", {})
    time_index = time_dimension.get("index", {})
    values = payload.get("value", {})
    if not isinstance(time_index, dict) or not isinstance(values, dict):
        return pd.DataFrame(columns=["date", "value"])
    reverse_time = {int(index): key for key, index in time_index.items()}
    rows = [
        {"date": _normalize_time_label(str(reverse_time[int(index)])), "value": value}
        for index, value in values.items()
        if int(index) in reverse_time
    ]
    return pd.DataFrame(rows, columns=["date", "value"])


def fetch_eurostat_series(
    source_series_id: str,
    country: str,
    frequency: str,
    api_base: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch one Eurostat dataset slice and return a normalized DataFrame."""
    base = api_base or os.getenv("EUROSTAT_API_BASE") or EUROSTAT_DEFAULT_API_BASE
    url = f"{base.rstrip('/')}/{source_series_id.lstrip('/')}"
    response = requests.get(url, params={"format": "JSON"}, timeout=timeout)
    response.raise_for_status()
    frame = _parse_eurostat_json(response.json())
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = source_series_id
    frame["country"] = country
    frame["source"] = "eurostat"
    frame["frequency"] = frequency
    frame["release_date"] = frame["date"]
    frame["ingested_at"] = _ingested_at()
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame.loc[:, NORMALIZED_COLUMNS]
