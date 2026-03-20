"""Helpers for lightweight nowcast-style date overlays in the dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

US_FRED_SERIES_MAP = {
    "CPIAUCSL": "cpi",
    "CPILFESL": "core_cpi",
    "UNRATE": "unrate",
    "FEDFUNDS": "policy_rate",
    "GS10": "yield_10y",
    "M2SL": "m2",
}

COUNTRY_INPUT_PRIORITY = {
    "us": [Path("data/raw/fred")],
    "china": [Path("data/raw/api/china/normalized"), Path("data/raw/manual/china")],
    "eurozone": [Path("data/raw/api/eurozone/normalized"), Path("data/raw/manual/eurozone")],
}

MARKET_SENSITIVE_SERIES = {
    "us": {"policy_rate", "yield_10y"},
    "china": {"policy_rate", "yield_10y", "hs300_pe_proxy", "hs300_pb_proxy"},
    "eurozone": {"policy_rate", "yield_10y"},
}

COUNTRY_PROCESSED_FILES = {
    "us": [
        Path("data/processed/us_macro_features.csv"),
        Path("data/processed/us_macro_regimes.csv"),
        Path("data/processed/us_valuation_features.csv"),
        Path("data/processed/us_asset_preferences.csv"),
    ],
    "china": [
        Path("data/processed/china_macro_features.csv"),
        Path("data/processed/china_macro_regimes.csv"),
        Path("data/processed/china_valuation_features.csv"),
        Path("data/processed/china_asset_preferences.csv"),
    ],
    "eurozone": [
        Path("data/processed/eurozone_macro_features.csv"),
        Path("data/processed/eurozone_macro_regimes.csv"),
        Path("data/processed/eurozone_valuation_features.csv"),
        Path("data/processed/eurozone_asset_preferences.csv"),
    ],
}

GLOBAL_PROCESSED_FILES = [
    Path("data/processed/global_macro_summary.csv"),
    Path("data/processed/global_allocation_map.csv"),
    Path("data/processed/global_summary_history.csv"),
    Path("data/processed/global_allocation_history.csv"),
]


def _read_series_file(path: Path, country: str, source_label: str) -> dict[str, object] | None:
    """Read one raw series file and return a compact status row."""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty or "date" not in frame.columns:
        return None

    dates = pd.to_datetime(frame["date"], errors="coerce")
    latest_date = dates.max()
    if pd.isna(latest_date):
        return None

    if country == "us":
        series_id = US_FRED_SERIES_MAP.get(path.stem, path.stem.lower())
    else:
        series_id = path.stem

    source_used = source_label
    if "source" in frame.columns:
        sources = frame["source"].dropna().astype(str).unique().tolist()
        if sources:
            source_used = sources[0]

    return {
        "series_id": series_id,
        "latest_date": latest_date,
        "row_count": int(len(frame)),
        "source_used": source_used,
    }


def _first_existing_input_folder(country: str) -> tuple[Path | None, str]:
    """Return the preferred existing input folder for a country."""
    for folder in COUNTRY_INPUT_PRIORITY[country]:
        if not folder.exists():
            continue
        files = [path for path in folder.glob("*.csv") if path.stem != "_summary"]
        if files:
            if "normalized" in folder.parts:
                return folder, "normalized_api"
            if "fred" in folder.parts:
                return folder, "fred"
            return folder, "manual_fallback"
    return None, "missing"


def collect_country_input_status(country: str) -> pd.DataFrame:
    """Collect actual loaded input status for a country from the preferred source path."""
    folder, source_label = _first_existing_input_folder(country)
    if folder is None:
        return pd.DataFrame(columns=["series_id", "latest_date", "row_count", "source_used"])

    rows: list[dict[str, object]] = []
    for path in sorted(folder.glob("*.csv")):
        if path.stem == "_summary":
            continue
        row = _read_series_file(path, country=country, source_label=source_label)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["series_id", "latest_date", "row_count", "source_used"])
    return pd.DataFrame(rows).sort_values(["latest_date", "series_id"]).reset_index(drop=True)


def latest_processed_timestamp(country: str | None = None) -> pd.Timestamp | pd.NaT:
    """Return the latest processed-file modification timestamp for a country or the global stack."""
    paths = GLOBAL_PROCESSED_FILES if country is None else COUNTRY_PROCESSED_FILES.get(country, [])
    timestamps = [
        pd.Timestamp(path.stat().st_mtime, unit="s")
        for path in paths
        if path.exists()
    ]
    if not timestamps:
        return pd.NaT
    return max(timestamps)


def build_country_nowcast_overlay(country: str, regime_date: pd.Timestamp | pd.NaT) -> dict[str, object]:
    """Build a lightweight country-level overlay using the freshest available raw inputs."""
    status = collect_country_input_status(country)
    freshest_input_date = status["latest_date"].max() if not status.empty else pd.NaT
    market_status = status.loc[status["series_id"].isin(MARKET_SENSITIVE_SERIES.get(country, set()))].copy()
    freshest_market_date = market_status["latest_date"].max() if not market_status.empty else pd.NaT
    freshest_market_series = (
        sorted(
            market_status.loc[market_status["latest_date"] == freshest_market_date, "series_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if pd.notna(freshest_market_date)
        else []
    )
    has_newer_market_input = (
        pd.notna(freshest_market_date)
        and pd.notna(regime_date)
        and pd.Timestamp(freshest_market_date) > pd.Timestamp(regime_date)
    )
    return {
        "status": status,
        "system_update_timestamp": latest_processed_timestamp(country),
        "freshest_input_date": freshest_input_date,
        "freshest_market_date": freshest_market_date,
        "freshest_market_series": freshest_market_series,
        "has_newer_market_input": bool(has_newer_market_input),
    }


def build_global_nowcast_overlay(
    summary_date: pd.Timestamp | pd.NaT,
    country_regime_dates: dict[str, pd.Timestamp | pd.NaT],
) -> dict[str, object]:
    """Build a global overlay summary from the country-level overlays."""
    overlays = {
        country: build_country_nowcast_overlay(country, regime_date)
        for country, regime_date in country_regime_dates.items()
    }
    freshest_dates = [
        overlay["freshest_market_date"]
        for overlay in overlays.values()
        if pd.notna(overlay["freshest_market_date"])
    ]
    freshest_market_date = max(freshest_dates) if freshest_dates else pd.NaT
    countries_with_newer_inputs = sorted(
        [
            country
            for country, overlay in overlays.items()
            if overlay["has_newer_market_input"]
        ]
    )
    freshest_market_sources: list[str] = []
    if pd.notna(freshest_market_date):
        for country, overlay in overlays.items():
            if overlay["freshest_market_date"] == freshest_market_date:
                freshest_market_sources.extend(
                    [f"{country}:{series_id}" for series_id in overlay["freshest_market_series"]]
                )
    return {
        "system_update_timestamp": latest_processed_timestamp(None),
        "freshest_market_date": freshest_market_date,
        "countries_with_newer_inputs": countries_with_newer_inputs,
        "freshest_market_sources": freshest_market_sources,
        "country_overlays": overlays,
        "summary_date": summary_date,
    }
