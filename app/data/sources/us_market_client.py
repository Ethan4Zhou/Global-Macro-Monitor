"""Public US equity valuation adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import re

import pandas as pd
import requests

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

MULTPL_BASE_URL = "https://www.multpl.com"
MULTPL_SERIES_CONFIG = {
    "equity_pe_proxy": {
        "path": "s-p-500-pe-ratio/table/by-month",
        "frequency": "monthly",
    },
    "shiller_pe_proxy": {
        "path": "shiller-pe/table/by-month",
        "frequency": "monthly",
    },
    "equity_pb_proxy": {
        "path": "s-p-500-price-to-book/table/by-quarter",
        "frequency": "quarterly",
    },
    "earnings_yield_proxy": {
        "path": "s-p-500-earnings-yield/table/by-month",
        "frequency": "monthly",
    },
}


def _ingested_at() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _clean_multpl_value(value: object) -> float:
    """Convert a Multpl string value into a float."""
    text = str(value or "").replace("†", "").replace("%", "").replace(",", "").strip()
    if not text:
        return float("nan")
    return float(text)


def _parse_multpl_table(html: str) -> pd.DataFrame:
    """Parse a Multpl table page into a two-column DataFrame."""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        tables = []
    for table in tables:
        if len(table.columns) < 2:
            continue
        frame = table.iloc[:, :2].copy()
        frame.columns = ["release_date", "value"]
        return frame

    rows = re.findall(r"([A-Z][a-z]{2} \d{1,2}, \d{4})\s+([0-9.,]+%?)", html)
    if not rows:
        raise ValueError("Unable to parse Multpl table rows.")
    return pd.DataFrame(rows, columns=["release_date", "value"])


def _expand_to_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand lower-frequency valuation observations into a monthly as-of series."""
    cleaned = frame.copy()
    cleaned["release_date"] = pd.to_datetime(cleaned["release_date"], errors="coerce")
    cleaned["value"] = cleaned["value"].map(_clean_multpl_value)
    cleaned = cleaned.dropna(subset=["release_date", "value"]).sort_values("release_date")
    if cleaned.empty:
        return cleaned

    cleaned["date"] = cleaned["release_date"].dt.to_period("M").dt.to_timestamp()
    cleaned = cleaned.sort_values(["date", "release_date"]).drop_duplicates(subset=["date"], keep="last")
    monthly_index = pd.date_range(cleaned["date"].min(), cleaned["date"].max(), freq="MS")
    expanded = (
        cleaned.set_index("date")
        .reindex(monthly_index)
        .sort_index()
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )
    return expanded.loc[:, ["date", "value", "release_date"]]


def fetch_us_market_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
) -> pd.DataFrame:
    """Fetch one US market valuation series from Multpl."""
    del source_hint
    if source_series_id not in MULTPL_SERIES_CONFIG:
        raise ValueError(f"Unsupported US market valuation series: {source_series_id}")

    config = MULTPL_SERIES_CONFIG[source_series_id]
    url = f"{MULTPL_BASE_URL}/{config['path']}"
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "global-macro-monitor/1.0"},
    )
    response.raise_for_status()

    parsed = _parse_multpl_table(response.text)
    expanded = _expand_to_monthly(parsed)
    if expanded.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    normalized = expanded.copy()
    normalized["series_id"] = source_series_id
    normalized["country"] = country
    normalized["source"] = "multpl"
    normalized["frequency"] = frequency
    normalized["ingested_at"] = _ingested_at()
    return normalized.loc[:, NORMALIZED_COLUMNS]


def save_us_market_series(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist a US market valuation frame to CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
