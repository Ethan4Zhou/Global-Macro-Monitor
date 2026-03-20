"""Eurozone ECB adapter wrappers."""

from __future__ import annotations

import pandas as pd

from app.data.sources.ecb_client import fetch_ecb_series


def fetch_eurozone_ecb_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
) -> pd.DataFrame:
    """Fetch and normalize one Eurozone ECB series."""
    del source_hint
    return fetch_ecb_series(series_id=source_series_id, country=country, frequency=frequency)
