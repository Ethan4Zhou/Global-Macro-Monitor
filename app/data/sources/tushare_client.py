"""Tushare Pro API client for China macro and valuation series."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

DEFAULT_TUSHARE_API_URL = "http://lianghua.nanyangqiankun.top"
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

TUSHARE_SERIES_ALIASES = {
    "cn_cpi": "cpi",
    "cn_pmi": "pmi",
    "cn_m": "m2",
    "shibor_lpr": "policy_rate",
    "yc_cb_10y": "yield_10y",
}

TUSHARE_SERIES_CONFIG: dict[str, dict[str, Any]] = {
    "cpi": {"api_name": "cn_cpi", "date_field": "month", "value_field": "nt_yoy"},
    "pmi": {"api_name": "cn_pmi", "date_field": "month", "value_field": "pmi010000"},
    "m2": {"api_name": "cn_m", "date_field": "month", "value_field": "m2_yoy"},
    "policy_rate": {"api_name": "shibor_lpr", "date_field": "date", "value_field": "1y"},
    "yield_10y": {
        "api_name": "yc_cb",
        "date_field": "trade_date",
        "value_field": "yield",
        "params": {"curve_type": "0", "curve_term": "10"},
    },
    "hs300_pe_proxy": {
        "api_name": "index_dailybasic",
        "date_field": "trade_date",
        "value_field": "pe_ttm",
        "params": {"ts_code": "000300.SH", "start_date": "20050101"},
    },
    "hs300_pb_proxy": {
        "api_name": "index_dailybasic",
        "date_field": "trade_date",
        "value_field": "pb",
        "params": {"ts_code": "000300.SH", "start_date": "20050101"},
    },
    "ppi": {"api_name": "cn_ppi", "date_field": "month", "value_field": "ppi_yoy"},
}


def _ingested_at() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _tushare_api_url() -> str:
    """Return the configured Tushare API URL."""
    return os.getenv("TUSHARE_API_URL", DEFAULT_TUSHARE_API_URL)


def _canonical_series_id(source_series_id: str) -> str:
    """Map legacy Tushare ids to canonical China series ids."""
    return TUSHARE_SERIES_ALIASES.get(source_series_id, source_series_id)


def _normalize_period_label(label: str) -> str:
    """Convert Tushare period labels into parseable dates."""
    text = str(label)
    if len(text) == 6 and text.isdigit():
        return f"{text[:4]}-{text[4:]}-01"
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _parse_tushare_payload(payload: dict[str, Any], series_id: str) -> pd.DataFrame:
    """Parse a Tushare response into a simple date/value frame."""
    code = payload.get("code")
    if code not in (None, 0):
        raise ValueError(f"Tushare API returned code={code}: {payload.get('msg', '')}")

    data = payload.get("data", {})
    fields = data.get("fields", [])
    items = data.get("items", [])
    config = TUSHARE_SERIES_CONFIG[series_id]
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
    source_hint: str | None = None,
) -> pd.DataFrame:
    """Fetch one Tushare series and return a normalized DataFrame."""
    del source_hint
    series_id = _canonical_series_id(source_series_id)
    if series_id not in TUSHARE_SERIES_CONFIG:
        raise ValueError(f"Unsupported Tushare series config: {source_series_id}")
    api_token = token or os.getenv("TUSHARE_TOKEN")
    if not api_token:
        raise ValueError("TUSHARE_TOKEN is required to fetch China API data.")

    config = TUSHARE_SERIES_CONFIG[series_id]
    body = {
        "api_name": config["api_name"],
        "token": api_token,
        "params": config.get("params", {}),
        "fields": ",".join([config["date_field"], config["value_field"]]),
    }
    response = requests.post(_tushare_api_url(), json=body, timeout=timeout)
    response.raise_for_status()
    frame = _parse_tushare_payload(response.json(), series_id=series_id)
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = series_id
    frame["country"] = country
    frame["source"] = "tushare"
    frame["frequency"] = frequency
    frame["release_date"] = frame["date"]
    frame["ingested_at"] = _ingested_at()
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame.loc[:, NORMALIZED_COLUMNS]
