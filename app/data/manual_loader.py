"""Helpers for loading and validating manual country CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MINIMUM_MANUAL_SERIES = ["cpi", "pmi", "policy_rate", "yield_10y"]
REQUIRED_COLUMNS = {"date", "value", "series_id"}


def load_manual_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate one manual CSV file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing manual CSV: {file_path}")

    frame = pd.read_csv(file_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Manual CSV is empty: {file_path}")

    cleaned = frame.loc[:, ["date", "value", "series_id"]].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")

    if cleaned["date"].isna().any():
        raise ValueError(f"Invalid date values in {file_path}")
    if cleaned["value"].isna().any():
        raise ValueError(f"Invalid numeric values in {file_path}")

    return cleaned.sort_values("date").reset_index(drop=True)


def load_country_manual_series(
    country: str,
    base_dir: str = "data/raw/manual",
) -> pd.DataFrame:
    """Load all valid manual CSV files for one country into a normalized frame."""
    country_dir = Path(base_dir) / country
    if not country_dir.exists():
        return pd.DataFrame(columns=["date", "value", "series_id"])

    frames: list[pd.DataFrame] = []
    for file_path in sorted(country_dir.glob("*.csv")):
        frames.append(load_manual_csv(file_path))

    if not frames:
        return pd.DataFrame(columns=["date", "value", "series_id"])

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )


def assess_manual_country_readiness(
    country: str,
    base_dir: str = "data/raw/manual",
) -> dict[str, object]:
    """Check whether a country has the minimum manual series for V1."""
    normalized = load_country_manual_series(country=country, base_dir=base_dir)
    available_series = sorted(normalized["series_id"].dropna().astype(str).unique().tolist())
    missing_series = [series for series in MINIMUM_MANUAL_SERIES if series not in available_series]
    return {
        "country": country,
        "available_series": available_series,
        "missing_series": missing_series,
        "ready": len(missing_series) == 0,
    }
