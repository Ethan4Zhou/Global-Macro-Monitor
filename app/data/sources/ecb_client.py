"""ECB API client for Eurozone macro series."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import os
from typing import Any

import pandas as pd
import requests

ECB_DEFAULT_API_BASE = "https://data-api.ecb.europa.eu/service/data"
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


def _empty_frame() -> pd.DataFrame:
    """Return an empty normalized frame."""
    return pd.DataFrame(columns=NORMALIZED_COLUMNS)


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    series_id: str,
    country: str,
    frequency: str,
) -> pd.DataFrame:
    """Normalize an ECB frame to the shared long-format schema."""
    if frame.empty:
        return _empty_frame()
    normalized = frame.loc[:, ["date", "value"]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized["series_id"] = series_id
    normalized["country"] = country
    normalized["source"] = "ecb"
    normalized["frequency"] = frequency
    normalized["release_date"] = normalized["date"]
    normalized["ingested_at"] = _ingested_at()
    normalized = normalized.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return normalized.loc[:, NORMALIZED_COLUMNS]


def _parse_ecb_csv(text: str) -> pd.DataFrame:
    """Parse a CSV-style ECB response."""
    frame = pd.read_csv(StringIO(text))
    for date_column in ["TIME_PERIOD", "DATE", "date"]:
        if date_column in frame.columns:
            break
    else:
        raise ValueError("ECB CSV response missing a time column.")
    for value_column in ["OBS_VALUE", "value", "obs_value"]:
        if value_column in frame.columns:
            break
    else:
        raise ValueError("ECB CSV response missing a value column.")
    return frame.rename(columns={date_column: "date", value_column: "value"}).loc[:, ["date", "value"]]


def _parse_ecb_json(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a minimal SDMX-JSON ECB payload."""
    time_values = (
        payload.get("structure", {})
        .get("dimensions", {})
        .get("observation", [{}])[0]
        .get("values", [])
    )
    observations = (
        payload.get("dataSets", [{}])[0]
        .get("series", {})
    )
    if not observations:
        return pd.DataFrame(columns=["date", "value"])
    first_series = next(iter(observations.values()))
    obs_map = first_series.get("observations", {})
    rows: list[dict[str, object]] = []
    for key, value in obs_map.items():
        index = int(str(key).split(":")[0])
        if index >= len(time_values):
            continue
        date_value = time_values[index].get("id") or time_values[index].get("name")
        rows.append({"date": date_value, "value": value[0] if isinstance(value, list) else value})
    return pd.DataFrame(rows, columns=["date", "value"])


def fetch_ecb_series(
    series_id: str,
    country: str,
    frequency: str,
    api_base: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch one ECB series and return a normalized DataFrame."""
    base = api_base or os.getenv("ECB_API_BASE") or ECB_DEFAULT_API_BASE
    url = f"{base.rstrip('/')}/{series_id.lstrip('/')}"
    response = requests.get(
        url,
        params={"format": "jsondata", "detail": "dataonly"},
        headers={"Accept": "application/json, text/csv"},
        timeout=timeout,
    )
    response.raise_for_status()

    content_type = str(response.headers.get("content-type", "")).lower()
    text_payload = response.text if isinstance(response.text, str) else ""
    if "json" in content_type:
        frame = _parse_ecb_json(response.json())
    elif "csv" in content_type or text_payload.startswith("TIME_PERIOD"):
        frame = _parse_ecb_csv(text_payload)
    else:
        frame = _parse_ecb_json(response.json())
    return _normalize_frame(frame, series_id=series_id, country=country, frequency=frequency)
