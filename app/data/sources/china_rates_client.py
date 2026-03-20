"""China rates adapter for yield and short-rate proxies."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import os

import pandas as pd
import requests

CHINA_RATES_DEFAULT_API_BASE = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-currency"
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


def _parse_rates_payload(text: str) -> pd.DataFrame:
    """Parse a CSV-like rates payload into date/value rows."""
    frame = pd.read_csv(StringIO(text))
    for date_column in ["date", "trade_date", "workTime", "time"]:
        if date_column in frame.columns:
            break
    else:
        raise ValueError("China rates response missing a date column.")

    for value_column in ["value", "yield", "rate", "close"]:
        if value_column in frame.columns:
            break
    else:
        raise ValueError("China rates response missing a value column.")

    return frame.rename(columns={date_column: "date", value_column: "value"}).loc[:, ["date", "value"]]


def fetch_china_rates_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
    api_base: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch one China rates series from a public rates endpoint."""
    base = api_base or os.getenv("CHINA_RATES_API_BASE") or CHINA_RATES_DEFAULT_API_BASE
    response = requests.get(
        base,
        params={"series_id": source_series_id, "source_hint": source_hint or source_series_id},
        timeout=timeout,
    )
    response.raise_for_status()
    frame = _parse_rates_payload(response.text)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = source_series_id
    frame["country"] = country
    frame["source"] = "china_rates"
    frame["frequency"] = frequency
    frame["release_date"] = pd.NaT
    frame["ingested_at"] = _ingested_at()
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame.loc[:, NORMALIZED_COLUMNS]
