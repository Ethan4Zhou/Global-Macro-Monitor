"""China NBS / National Data adapter for macro series."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
from typing import Any

import pandas as pd
import requests

NBS_DEFAULT_API_BASE = "https://data.stats.gov.cn/easyquery.htm"
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


def _normalize_date(value: object) -> str:
    """Convert common period labels into parseable dates."""
    text = str(value)
    if len(text) == 6 and text.isdigit():
        return f"{text[:4]}-{text[4:]}-01"
    if len(text) == 7 and "-" in text:
        return f"{text}-01"
    return text


def _parse_rows(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a simplified NBS payload into date/value rows."""
    rows = payload.get("data") or payload.get("rows") or payload.get("observations") or []
    if not isinstance(rows, list):
        return pd.DataFrame(columns=["date", "value"])

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        date_value = (
            row.get("date")
            or row.get("period")
            or row.get("time")
            or row.get("stat_month")
            or row.get("month")
        )
        value = row.get("value") or row.get("data") or row.get("obs_value")
        if date_value is None:
            continue
        normalized_rows.append({"date": _normalize_date(date_value), "value": value})
    return pd.DataFrame(normalized_rows, columns=["date", "value"])


def _month_name_to_number(text: str) -> int | None:
    """Convert an English month name into its numeric representation."""
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    return month_map.get(text.strip().lower())


def _extract_release_month(text: str) -> pd.Timestamp:
    """Extract the release month from an NBS English monthly release."""
    patterns = [
        re.compile(r"in\s+(?P<month>[A-Za-z]+)\s+(?P<year>20\d{2})", re.IGNORECASE),
        re.compile(r"(?P<month>[A-Za-z]+)\s+(?P<year>20\d{2})", re.IGNORECASE),
        re.compile(r"(?P<year>20\d{2})[-/](?P<month>\d{1,2})"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        year = int(match.group("year"))
        month_text = match.group("month")
        month = int(month_text) if month_text.isdigit() else _month_name_to_number(month_text)
        if month:
            return pd.Timestamp(year=year, month=month, day=1)
    return pd.NaT


def _extract_series_value(text: str, patterns: list[str]) -> float | None:
    """Extract the first matching numeric value from a release text."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group("value"))
    return None


def extract_core_cpi_from_release_text(text: str) -> pd.DataFrame:
    """Extract China core CPI from a monthly NBS English release."""
    date = _extract_release_month(text)
    value = _extract_series_value(
        text,
        patterns=[
            r"core CPI[^.]*?(?:rose|increased|was up|went up)\s+(?P<value>-?\d+(?:\.\d+)?)\s*percent",
            r"consumer price index excluding food and energy[^.]*?(?:rose|increased|was up|went up)\s+(?P<value>-?\d+(?:\.\d+)?)\s*percent",
        ],
    )
    if pd.isna(date) or value is None:
        return pd.DataFrame(columns=["date", "value"])
    return pd.DataFrame({"date": [date], "value": [value]})


def extract_unrate_from_release_text(text: str) -> pd.DataFrame:
    """Extract China urban surveyed unemployment from a monthly NBS English release."""
    date = _extract_release_month(text)
    value = _extract_series_value(
        text,
        patterns=[
            r"surveyed urban unemployment rate[^.]*?(?:was|stood at|came in at)\s+(?P<value>-?\d+(?:\.\d+)?)\s*percent",
            r"urban surveyed unemployment rate[^.]*?(?:was|stood at|came in at)\s+(?P<value>-?\d+(?:\.\d+)?)\s*percent",
        ],
    )
    if pd.isna(date) or value is None:
        return pd.DataFrame(columns=["date", "value"])
    return pd.DataFrame({"date": [date], "value": [value]})


def _parse_release_text(text: str, series_id: str) -> pd.DataFrame:
    """Parse NBS English release text for selected China enrichment series."""
    if series_id == "core_cpi":
        return extract_core_cpi_from_release_text(text)
    if series_id == "unrate":
        return extract_unrate_from_release_text(text)
    return pd.DataFrame(columns=["date", "value"])


def fetch_china_nbs_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
    api_base: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch one China macro series from a National Data style endpoint."""
    base = api_base or os.getenv("CHINA_NBS_API_BASE") or NBS_DEFAULT_API_BASE
    response = requests.get(
        base,
        params={"series_id": source_series_id, "source_hint": source_hint or source_series_id},
        timeout=timeout,
    )
    response.raise_for_status()
    frame = pd.DataFrame(columns=["date", "value"])
    try:
        frame = _parse_rows(response.json())
    except ValueError:
        frame = pd.DataFrame(columns=["date", "value"])
    if frame.empty and hasattr(response, "text"):
        frame = _parse_release_text(response.text, source_series_id)
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = source_series_id
    frame["country"] = country
    frame["source"] = "china_nbs"
    frame["frequency"] = frequency
    frame["release_date"] = pd.NaT
    frame["ingested_at"] = _ingested_at()
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame.loc[:, NORMALIZED_COLUMNS]
