"""Fetch and normalize shared market-overlay series for the nowcast layer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.data.fetchers import fetch_fred_series
from app.data.sources.public_site_client import fetch_public_site_series
from app.utils.logging import get_logger

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

MARKET_OVERLAY_FRED_SERIES = {
    "dxy_proxy": {"fred_id": "DTWEXBGS", "country": "global"},
    "credit_spread_proxy": {"fred_id": "BAMLC0A0CM", "country": "global"},
    "vix_proxy": {"fred_id": "VIXCLS", "country": "global"},
    "gold_proxy": {"fred_id": "NASDAQXAU", "country": "global"},
    "oil_proxy": {"fred_id": "DCOILWTICO", "country": "global"},
    "copper_proxy": {"fred_id": "PCOPPUSDM", "country": "global"},
    "sp500_proxy": {"fred_id": "SP500", "country": "us"},
    "eurostoxx50_proxy": {"fred_id": "NASDAQNQEUROZ50T", "country": "eurozone"},
    "china_equity_proxy": {"fred_id": "NASDAQNQCNAT", "country": "china"},
}
MARKET_OVERLAY_PUBLIC_SERIES = {
    "gold_proxy": {"source_series_id": "gold_proxy_public", "country": "global"},
    "oil_proxy": {"source_series_id": "oil_proxy_public", "country": "global"},
    "copper_proxy": {"source_series_id": "copper_proxy_public", "country": "global"},
    "sp500_proxy": {"source_series_id": "sp500_proxy_public", "country": "us"},
    "china_equity_proxy": {"source_series_id": "china_equity_proxy_public", "country": "china"},
    "eurostoxx50_proxy": {"source_series_id": "eurozone_equity_proxy_public", "country": "eurozone"},
}

logger = get_logger(__name__)


def _ingested_at() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def normalize_market_overlay_frame(
    frame: pd.DataFrame,
    *,
    series_id: str,
    country: str,
    source: str = "fred",
    frequency: str = "daily",
) -> pd.DataFrame:
    """Normalize one fetched market series into the shared schema."""
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    if normalized.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    normalized["series_id"] = series_id
    normalized["country"] = country
    normalized["source"] = source
    normalized["frequency"] = frequency
    normalized["release_date"] = normalized["date"]
    normalized["ingested_at"] = _ingested_at()
    return normalized.loc[:, NORMALIZED_COLUMNS]


def fetch_market_overlay_bundle(api_key: str) -> dict[str, pd.DataFrame]:
    """Fetch a compact shared market-overlay bundle from FRED."""
    bundle: dict[str, pd.DataFrame] = {}
    for series_id, metadata in MARKET_OVERLAY_FRED_SERIES.items():
        try:
            raw = fetch_fred_series(metadata["fred_id"], api_key=api_key)
            bundle[series_id] = normalize_market_overlay_frame(
                raw,
                series_id=series_id,
                country=str(metadata["country"]),
                source="fred",
                frequency="daily",
            )
        except Exception as exc:  # pragma: no cover - exercised by live fetch failures
            logger.warning("Failed to fetch market overlay series %s (%s): %s", series_id, metadata["fred_id"], exc)
            bundle[series_id] = pd.DataFrame(columns=NORMALIZED_COLUMNS)
        if bundle[series_id].empty and series_id in MARKET_OVERLAY_PUBLIC_SERIES:
            public_metadata = MARKET_OVERLAY_PUBLIC_SERIES[series_id]
            try:
                bundle[series_id] = fetch_public_site_series(
                    source_series_id=str(public_metadata["source_series_id"]),
                    country=str(public_metadata["country"]),
                    frequency="daily",
                )
            except Exception as exc:  # pragma: no cover - exercised by live fetch failures
                logger.warning(
                    "Failed to fetch public market overlay series %s (%s): %s",
                    series_id,
                    public_metadata["source_series_id"],
                    exc,
                )
                bundle[series_id] = pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return bundle


def save_market_overlay_series(
    frame: pd.DataFrame,
    *,
    series_id: str,
    output_dir: str = "data/raw/api/global_markets/normalized",
) -> Path:
    """Persist one normalized market-overlay series to CSV."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{series_id}.csv"
    frame.to_csv(output_path, index=False)
    return output_path
