"""Public international equity valuation adapters."""

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

SIBLIS_BASE_URL = "https://siblisresearch.com"
SIBLIS_SERIES_CONFIG = {
    "china_shiller_pe_proxy": {
        "path": "data/china-shanghai-pe-cape-ratio/",
        "country": "china",
        "frequency": "monthly",
        "metric_candidates": ["cape ratio", "cape"],
        "output_series_id": "shiller_pe_proxy",
    },
    "eurozone_equity_pe_proxy": {
        "path": "data/europe-pe-ratio/",
        "country": "eurozone",
        "frequency": "monthly",
        "metric_candidates": ["pe ratio", "p/e ratio"],
        "output_series_id": "equity_pe_proxy",
    },
    "eurozone_shiller_pe_proxy": {
        "path": "data/europe-pe-ratio/",
        "country": "eurozone",
        "frequency": "monthly",
        "metric_candidates": ["cape ratio", "cape"],
        "output_series_id": "shiller_pe_proxy",
    },
}


def _ingested_at() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _normalize_column_name(value: object) -> str:
    """Normalize column labels for robust table matching."""
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _clean_numeric(value: object) -> float:
    """Convert a scraped string value into a float."""
    text = str(value or "").replace(",", "").replace("%", "").strip()
    if not text:
        return float("nan")
    return float(text)


def _find_siblis_table(html: str) -> pd.DataFrame:
    """Extract the historical valuation table from a Siblis page."""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        tables = []

    for table in tables:
        normalized_columns = [_normalize_column_name(column) for column in table.columns]
        if "date" in normalized_columns and any(
            candidate in normalized_columns for candidate in ["pe ratio", "p/e ratio", "cape ratio", "cape"]
        ):
            frame = table.copy()
            frame.columns = normalized_columns
            return frame

    raise ValueError("Unable to locate a Siblis valuation table.")


def _extract_metric_series(frame: pd.DataFrame, metric_candidates: list[str]) -> pd.DataFrame:
    """Select one metric column from the parsed Siblis table."""
    normalized_candidates = [_normalize_column_name(item) for item in metric_candidates]
    metric_column = next(
        (
            column
            for column in frame.columns
            if any(candidate in column or column in candidate for candidate in normalized_candidates)
        ),
        None,
    )
    if metric_column is None:
        raise ValueError(f"Unable to locate any of the requested metrics: {metric_candidates}")

    cleaned = frame.loc[:, ["date", metric_column]].copy()
    cleaned.columns = ["release_date", "value"]
    cleaned["release_date"] = pd.to_datetime(cleaned["release_date"], errors="coerce")
    cleaned["value"] = cleaned["value"].map(_clean_numeric)
    cleaned = cleaned.dropna(subset=["release_date", "value"]).sort_values("release_date")
    return cleaned.reset_index(drop=True)


def _expand_to_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand lower-frequency valuation observations into a monthly as-of series."""
    if frame.empty:
        return frame

    expanded = frame.copy()
    expanded["date"] = expanded["release_date"].dt.to_period("M").dt.to_timestamp()
    expanded = expanded.sort_values(["date", "release_date"]).drop_duplicates(subset=["date"], keep="last")
    monthly_index = pd.date_range(expanded["date"].min(), expanded["date"].max(), freq="MS")
    expanded = (
        expanded.set_index("date")
        .reindex(monthly_index)
        .sort_index()
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )
    return expanded.loc[:, ["date", "value", "release_date"]]


def fetch_international_market_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
) -> pd.DataFrame:
    """Fetch one international equity valuation proxy from Siblis Research."""
    del source_hint
    if source_series_id not in SIBLIS_SERIES_CONFIG:
        raise ValueError(f"Unsupported international valuation series: {source_series_id}")

    config = SIBLIS_SERIES_CONFIG[source_series_id]
    url = f"{SIBLIS_BASE_URL}/{config['path']}"
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "global-macro-monitor/1.0"},
    )
    response.raise_for_status()

    table = _find_siblis_table(response.text)
    metric_series = _extract_metric_series(table, config["metric_candidates"])
    monthly = _expand_to_monthly(metric_series)
    if monthly.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    normalized = monthly.copy()
    normalized["series_id"] = config["output_series_id"]
    normalized["country"] = country
    normalized["source"] = "siblis"
    normalized["frequency"] = frequency or config["frequency"]
    normalized["ingested_at"] = _ingested_at()
    return normalized.loc[:, NORMALIZED_COLUMNS]


def save_international_market_series(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist an international market valuation frame to CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
