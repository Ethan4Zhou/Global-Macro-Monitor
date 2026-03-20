"""Eurozone OECD adapter wrappers."""

from __future__ import annotations

import pandas as pd

from app.data.sources.oecd_client import fetch_oecd_series


def fetch_eurozone_oecd_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
) -> pd.DataFrame:
    """Fetch and normalize one Eurozone OECD fallback series."""
    del source_hint
    return fetch_oecd_series(
        source_series_id=source_series_id,
        country=country,
        frequency=frequency,
    )
