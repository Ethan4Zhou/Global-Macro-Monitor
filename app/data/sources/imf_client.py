"""Generic IMF adapter for macro fallback series."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import pandas as pd
import requests

IMF_DEFAULT_API_BASE = "https://api.imf.org/external/sdmx/2.1/data"
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


def _parse_imf_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a simplified IMF payload into date/value rows."""
    rows = payload.get("observations") or payload.get("data") or payload.get("values") or []
    if not isinstance(rows, list):
        return pd.DataFrame(columns=["date", "value"])

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        date_value = row.get("date") or row.get("period") or row.get("time_period")
        value = row.get("value") or row.get("obs_value")
        if date_value is None:
            continue
        normalized_rows.append({"date": date_value, "value": value})
    return pd.DataFrame(normalized_rows, columns=["date", "value"])


def fetch_imf_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
    api_base: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch one IMF series and normalize the result."""
    base = api_base or os.getenv("IMF_API_BASE") or IMF_DEFAULT_API_BASE
    url = f"{base.rstrip('/')}/{source_series_id.lstrip('/')}"
    response = requests.get(url, params={"source_hint": source_hint or source_series_id}, timeout=timeout)
    response.raise_for_status()
    frame = _parse_imf_payload(response.json())
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = source_series_id
    frame["country"] = country
    frame["source"] = "imf"
    frame["frequency"] = frequency
    frame["release_date"] = pd.NaT
    frame["ingested_at"] = _ingested_at()
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame.loc[:, NORMALIZED_COLUMNS]
