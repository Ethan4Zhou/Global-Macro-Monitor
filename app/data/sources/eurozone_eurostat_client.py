"""Eurozone Eurostat adapter wrappers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
import re

import pandas as pd
import requests

from app.data.sources.eurostat_client import fetch_eurostat_series

FLASH_URL_TEMPLATE = "https://ec.europa.eu/eurostat/web/products-euro-indicators/w/2-{date_code}-ap"
FLASH_LOOKBACK_DAYS = 21


def _clean_flash_value(value: object) -> float | None:
    """Convert Eurostat flash table cells like `1.9e` into floats."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    cleaned = re.sub(r"[^0-9.+-]", "", text)
    return float(cleaned) if cleaned else None


def _parse_flash_month(label: str) -> pd.Timestamp | None:
    """Parse Eurostat flash table month labels such as `Feb 26`."""
    text = str(label).strip()
    if not text:
        return None
    try:
        return pd.to_datetime(f"01 {text}", format="%d %b %y", errors="raise")
    except ValueError:
        return None


def _flatten_columns(frame: pd.DataFrame) -> list[str]:
    """Flatten a possibly multi-level Eurostat table header."""
    flattened: list[str] = []
    for column in frame.columns:
        if isinstance(column, tuple):
            parts = [str(part).strip() for part in column if str(part).strip() and not str(part).startswith("Unnamed")]
            flattened.append(" | ".join(parts))
        else:
            flattened.append(str(column).strip())
    return flattened


def _find_flash_row(table: pd.DataFrame, series_id: str) -> pd.Series | None:
    """Pick the relevant flash-inflation row for CPI or core CPI."""
    first_column = table.columns[0]
    labels = table[first_column].astype(str).str.lower()
    if series_id == "cpi":
        matched = table.loc[labels.str.contains("all-items hicp", na=False)]
        return None if matched.empty else matched.iloc[0]

    best_index: int | None = None
    best_score = 0
    keywords = ["excluding", "energy", "food", "alcohol", "tobacco"]
    for index, label in labels.items():
        score = sum(keyword in label for keyword in keywords)
        if score > best_score:
            best_score = score
            best_index = index
    if best_index is None or best_score < 3:
        return None
    return table.loc[best_index]


def _parse_flash_table(html: str, series_id: str) -> pd.DataFrame:
    """Parse one Eurostat flash-inflation page into a normalized row."""
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()
    table = tables[0].copy()
    table.columns = _flatten_columns(table)
    row = _find_flash_row(table, series_id)
    if row is None:
        return pd.DataFrame()

    annual_rate_columns = [column for column in table.columns if "Annual rate" in column]
    if not annual_rate_columns:
        return pd.DataFrame()
    latest_column = annual_rate_columns[-1]
    month_label = latest_column.split("|")[-1].strip()
    date_value = _parse_flash_month(month_label)
    value = _clean_flash_value(row[latest_column])
    if date_value is None or value is None:
        return pd.DataFrame()
    release_date = None
    match = re.search(r"/2-(\d{8})-ap", html)
    if match:
        release_date = pd.to_datetime(match.group(1), format="%d%m%Y", errors="coerce")
    return pd.DataFrame(
        {
            "date": [date_value],
            "value": [value],
            "series_id": [series_id],
            "country": ["eurozone"],
            "source": ["eurostat_flash"],
            "frequency": ["monthly"],
            "release_date": [release_date],
            "ingested_at": [pd.Timestamp(datetime.now(timezone.utc))],
        }
    )


def _fetch_flash_inflation_row(series_id: str, timeout: int = 8) -> pd.DataFrame:
    """Fetch the latest available Eurostat flash-inflation row."""
    today = datetime.now(timezone.utc).date()
    for offset in range(FLASH_LOOKBACK_DAYS):
        day = today - timedelta(days=offset)
        url = FLASH_URL_TEMPLATE.format(date_code=day.strftime("%d%m%Y"))
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code != 200:
                continue
            if "annual inflation" not in response.text.lower():
                continue
            parsed = _parse_flash_table(response.text, series_id)
            if not parsed.empty:
                return parsed
        except requests.RequestException:
            continue
    return pd.DataFrame()


def _append_flash_inflation_if_newer(frame: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Append a flash inflation row when it is newer than the API dataset."""
    if series_id not in {"cpi", "core_cpi"}:
        return frame
    flash_frame = _fetch_flash_inflation_row(series_id)
    if flash_frame.empty:
        return frame
    if frame.empty:
        return flash_frame
    latest_regular = pd.to_datetime(frame["date"], errors="coerce").max()
    latest_flash = pd.to_datetime(flash_frame["date"], errors="coerce").max()
    if pd.isna(latest_flash) or (pd.notna(latest_regular) and latest_flash <= latest_regular):
        return frame
    combined = pd.concat([frame, flash_frame], ignore_index=True)
    return combined.drop_duplicates(subset=["series_id", "date"], keep="last").sort_values("date").reset_index(drop=True)


def fetch_eurozone_eurostat_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
) -> pd.DataFrame:
    """Fetch and normalize one Eurozone Eurostat series."""
    series_id = str(source_hint or source_series_id)
    frame = fetch_eurostat_series(
        source_series_id=source_series_id,
        country=country,
        frequency=frequency,
    )
    return _append_flash_inflation_if_newer(frame, series_id=series_id)
