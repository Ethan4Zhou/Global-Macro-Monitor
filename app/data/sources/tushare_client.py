"""Tushare Pro API client for China macro series."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import pandas as pd
import requests

TUSHARE_API_URL = "https://api.tushare.pro"
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

TUSHARE_SERIES_CONFIG: dict[str, dict[str, Any]] = {
    "cn_cpi": {"api_name": "cn_cpi", "date_field": "month", "value_field": "nt_yoy"},
    "cn_pmi": {"api_name": "cn_pmi", "date_field": "month", "value_field": "pmi010000"},
    "cn_m": {"api_name": "cn_m", "date_field": "month", "value_field": "m2_yoy"},
    "shibor_lpr": {"api_name": "shibor_lpr", "date_field": "trade_date", "value_field": "1y"},
    "yc_cb_10y": {
        "api_name": "yc_cb",
        "date_field": "workTime",
        "value_field": "yield",
        "params": {"curve_type": "0", "curve_term": "10"},
    },
}


def _ingested_at() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _normalize_period_label(label: str) -> str:
    """Convert Tushare period labels into parseable dates."""
    text = str(label)
    if len(text) == 6 and text.isdigit():
        return f"{text[:4]}-{text[4:]}-01"
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _parse_tushare_payload(payload: dict[str, Any], source_series_id: str) -> pd.DataFrame:
    """Parse a Tushare response into a simple date/value frame."""
    data = payload.get("data", {})
    fields = data.get("fields", [])
    items = data.get("items", [])
    config = TUSHARE_SERIES_CONFIG[source_series_id]
    if not fields or not items:
        return pd.DataFrame(columns=["date", "value"])
    frame = pd.DataFrame(items, columns=fields)
    date_field = config["date_field"]
    value_field = config["value_field"]
    if date_field not in frame.columns or value_field not in frame.columns:
        raise ValueError(f"Tushare response missing {date_field} or {value_field}.")
    normalized = frame.rename(columns={date_field: "date", value_field: "value"}).loc[:, ["date", "value"]]
    normalized["date"] = normalized["date"].map(_normalize_period_label)
    return normalized


def fetch_tushare_series(
    source_series_id: str,
    country: str,
    frequency: str,
    token: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch one Tushare series and return a normalized DataFrame."""
    if source_series_id not in TUSHARE_SERIES_CONFIG:
        raise ValueError(f"Unsupported Tushare series config: {source_series_id}")
    api_token = token or os.getenv("TUSHARE_TOKEN")
    if not api_token:
        raise ValueError("TUSHARE_TOKEN is required to fetch China API data.")

    config = TUSHARE_SERIES_CONFIG[source_series_id]
    body = {
        "api_name": config["api_name"],
        "token": api_token,
        "params": config.get("params", {}),
        "fields": ",".join([config["date_field"], config["value_field"]]),
    }
    response = requests.post(TUSHARE_API_URL, json=body, timeout=timeout)
    response.raise_for_status()
    frame = _parse_tushare_payload(response.json(), source_series_id=source_series_id)
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = source_series_id
    frame["country"] = country
    frame["source"] = "tushare"
    frame["frequency"] = frequency
    frame["release_date"] = frame["date"]
    frame["ingested_at"] = _ingested_at()
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame.loc[:, NORMALIZED_COLUMNS]
