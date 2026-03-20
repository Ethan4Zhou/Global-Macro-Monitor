"""Data fetching utilities for macro time series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests

from app.data.sources.ecb_client import fetch_ecb_series
from app.data.sources.eurostat_client import fetch_eurostat_series
from app.data.sources.oecd_client import fetch_oecd_series
from app.data.sources.tushare_client import fetch_tushare_series
from app.utils.config import get_country_indicators
from app.utils.logging import get_logger

logger = get_logger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
US_FRED_SERIES = {
    "CPIAUCSL": "headline_cpi",
    "CPILFESL": "core_cpi",
    "UNRATE": "unemployment_rate",
    "FEDFUNDS": "fed_funds_rate",
    "GS10": "yield_10y",
    "M2SL": "m2_money_stock",
}
API_SOURCE_FETCHERS = {
    "ecb": fetch_ecb_series,
    "eurostat": fetch_eurostat_series,
    "oecd": fetch_oecd_series,
    "tushare": fetch_tushare_series,
}
API_SOURCE_ALIASES = {
    "eurozone_ecb": "ecb",
    "eurozone_eurostat": "eurostat",
    "eurozone_oecd": "oecd",
}


def _build_fred_params(
    series_id: str,
    api_key: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> dict[str, str]:
    """Build request parameters for the FRED observations endpoint."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end
    return params


def _parse_fred_observations(
    observations: list[dict[str, Any]],
    series_id: str,
) -> pd.DataFrame:
    """Convert FRED observations into a clean DataFrame."""
    frame = pd.DataFrame(observations)
    if frame.empty:
        return pd.DataFrame(columns=["date", "value", "series_id"])

    required_columns = {"date", "value"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing expected FRED fields: {sorted(missing)}")

    cleaned = frame.loc[:, ["date", "value"]].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
    cleaned["series_id"] = series_id
    cleaned = cleaned.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return cleaned.loc[:, ["date", "value", "series_id"]]


def fetch_fred_series(
    series_id: str,
    api_key: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> pd.DataFrame:
    """Fetch a single FRED series and return a cleaned DataFrame."""
    params = _build_fred_params(
        series_id=series_id,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=observation_end,
    )
    logger.info("Fetching FRED series %s", series_id)
    response = requests.get(FRED_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object from FRED.")

    observations = payload.get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("Expected 'observations' to be a list.")

    return _parse_fred_observations(observations, series_id=series_id)


def fetch_us_macro_bundle(api_key: str) -> dict[str, pd.DataFrame]:
    """Fetch the V1 US macro bundle from FRED."""
    return {
        series_id: fetch_fred_series(series_id=series_id, api_key=api_key)
        for series_id in US_FRED_SERIES
    }


def save_series_to_csv(frame: pd.DataFrame, series_id: str, output_dir: str = "data/raw/fred") -> Path:
    """Save a series DataFrame to a CSV file under the raw FRED directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{series_id}.csv"
    frame.to_csv(output_path, index=False)
    logger.info("Saved %s rows to %s", len(frame), output_path)
    return output_path


def save_api_series_to_csv(
    frame: pd.DataFrame,
    country: str,
    indicator_key: str,
    output_dir: str = "data/raw/api",
) -> Path:
    """Save a normalized API series under the country raw API directory."""
    destination = Path(output_dir) / country
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{indicator_key}.csv"
    frame.to_csv(output_path, index=False)
    logger.info("Saved %s rows to %s", len(frame), output_path)
    return output_path


def fetch_country_api_bundle(country: str) -> dict[str, pd.DataFrame]:
    """Fetch all configured API-backed macro series for one country."""
    bundle: dict[str, pd.DataFrame] = {}
    for indicator in get_country_indicators(country, "macro"):
        source = str(indicator.get("source", ""))
        lookup_source = API_SOURCE_ALIASES.get(source, source)
        if lookup_source not in API_SOURCE_FETCHERS:
            continue
        fetcher = API_SOURCE_FETCHERS[lookup_source]
        source_series_id = str(indicator.get("source_series_id") or indicator.get("series_id") or indicator.get("key"))
        try:
            frame = fetcher(
                source_series_id=source_series_id,
                country=country,
                frequency=str(indicator.get("frequency", "monthly")),
            ) if lookup_source in {"eurostat", "oecd", "tushare"} else fetcher(
                series_id=source_series_id,
                country=country,
                frequency=str(indicator.get("frequency", "monthly")),
            )
        except Exception:
            if indicator.get("fallback_source"):
                logger.warning(
                    "API fetch failed for %s/%s, downstream loader can fall back to %s.",
                    country,
                    indicator.get("key"),
                    indicator.get("fallback_source"),
                )
                continue
            raise
        bundle[str(indicator["key"])] = frame
    return bundle


@dataclass(slots=True)
class DataFetcher:
    """Simple data fetcher that can normalize and persist tabular macro data."""

    duckdb_path: str = "data/processed/global_macro.duckdb"
    timeout: int = 10

    def fetch_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch JSON from a remote endpoint."""
        logger.info("Fetching data from %s", url)
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object response.")
        return payload

    def normalize_records(
        self,
        records: list[dict[str, Any]],
        country: str,
        indicator: str,
    ) -> pd.DataFrame:
        """Convert a list of records into a standard long-format DataFrame."""
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(
                columns=["date", "country", "indicator", "value", "updated_at"]
            )

        if "date" not in frame.columns or "value" not in frame.columns:
            raise ValueError("Records must contain 'date' and 'value' fields.")

        frame = frame.loc[:, ["date", "value"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["country"] = country
        frame["indicator"] = indicator
        frame["updated_at"] = datetime.now(timezone.utc)
        return frame

    def save_to_duckdb(self, frame: pd.DataFrame, table_name: str = "macro_data") -> None:
        """Persist a DataFrame into DuckDB."""
        Path(self.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(self.duckdb_path) as conn:
            conn.register("incoming_frame", frame)
            conn.execute(
                f"""
                create table if not exists {table_name} as
                select * from incoming_frame where 1 = 0
                """
            )
            conn.execute(f"insert into {table_name} select * from incoming_frame")
        logger.info("Saved %s rows into %s", len(frame), table_name)
