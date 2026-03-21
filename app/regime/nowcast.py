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
    "us": [Path("data/raw/api/us/normalized"), Path("data/raw/api/global_markets/normalized"), Path("data/raw/fred")],
    "china": [Path("data/raw/api/china/normalized"), Path("data/raw/api/global_markets/normalized"), Path("data/raw/manual/china")],
    "eurozone": [Path("data/raw/api/eurozone/normalized"), Path("data/raw/api/global_markets/normalized"), Path("data/raw/manual/eurozone")],
}

MARKET_SENSITIVE_SERIES = {
    "us": {
        "cpi", "core_cpi", "policy_rate", "yield_10y",
        "equity_pe_proxy", "equity_pb_proxy", "shiller_pe_proxy",
        "dxy_proxy", "credit_spread_proxy", "vix_proxy", "gold_proxy",
        "oil_proxy", "copper_proxy", "sp500_proxy",
    },
    "china": {
        "cpi", "core_cpi", "policy_rate", "yield_10y",
        "hs300_pe_proxy", "hs300_pb_proxy", "shiller_pe_proxy",
        "dxy_proxy", "credit_spread_proxy", "vix_proxy", "gold_proxy",
        "oil_proxy", "copper_proxy", "china_equity_proxy",
    },
    "eurozone": {
        "cpi", "core_cpi", "policy_rate", "yield_10y",
        "equity_pe_proxy", "equity_pb_proxy", "shiller_pe_proxy",
        "dxy_proxy", "credit_spread_proxy", "vix_proxy", "gold_proxy",
        "oil_proxy", "copper_proxy", "eurostoxx50_proxy",
    },
}

NOWCAST_DIMENSION_BY_SERIES = {
    "cpi": "inflation",
    "core_cpi": "inflation",
    "policy_rate": "rates",
    "yield_10y": "rates",
    "equity_pe_proxy": "risk",
    "equity_pb_proxy": "risk",
    "shiller_pe_proxy": "risk",
    "hs300_pe_proxy": "risk",
    "hs300_pb_proxy": "risk",
    "dxy_proxy": "risk",
    "credit_spread_proxy": "risk",
    "vix_proxy": "risk",
    "copper_proxy": "risk",
    "sp500_proxy": "risk",
    "china_equity_proxy": "risk",
    "eurostoxx50_proxy": "risk",
    "gold_proxy": "inflation",
    "oil_proxy": "inflation",
}

NOWCAST_SIGNAL_THRESHOLDS = {
    "policy_rate": 0.05,
    "yield_10y": 0.08,
    "hs300_pe_proxy": 0.3,
    "hs300_pb_proxy": 0.03,
    "dxy_proxy": 0.25,
    "credit_spread_proxy": 0.05,
    "vix_proxy": 0.75,
    "gold_proxy": 5.0,
    "oil_proxy": 1.5,
    "copper_proxy": 0.05,
    "sp500_proxy": 5.0,
    "china_equity_proxy": 20.0,
    "eurostoxx50_proxy": 20.0,
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
    Path("data/processed/global_change_log.csv"),
]


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values and drop invalid observations."""
    return pd.to_numeric(series, errors="coerce")


def _series_signal(series_id: str, latest_value: float, previous_value: float) -> tuple[float, str, str]:
    """Convert one market-sensitive series change into a signed overlay signal."""
    delta = latest_value - previous_value
    threshold = NOWCAST_SIGNAL_THRESHOLDS.get(series_id, 0.05)
    dimension = NOWCAST_DIMENSION_BY_SERIES.get(series_id, "risk")
    if abs(delta) < threshold:
        return 0.0, "stable", dimension

    if series_id in {"policy_rate", "yield_10y"}:
        if delta < 0:
            return 1.0, "easing", dimension
        return -1.0, "tightening", dimension

    if series_id in {"cpi", "core_cpi"}:
        if delta < 0:
            return 1.0, "cooling", dimension
        return -1.0, "reheating", dimension

    if series_id == "dxy_proxy":
        if delta < 0:
            return 1.0, "weaker_dollar", dimension
        return -1.0, "stronger_dollar", dimension

    if series_id == "credit_spread_proxy":
        if delta < 0:
            return 1.0, "tighter_spreads", dimension
        return -1.0, "wider_spreads", dimension

    if series_id == "vix_proxy":
        if delta < 0:
            return 1.0, "lower_volatility", dimension
        return -1.0, "higher_volatility", dimension

    if series_id in {"gold_proxy", "oil_proxy"}:
        if delta < 0:
            return 1.0, "cooling", dimension
        return -1.0, "reheating", dimension

    if series_id in {"copper_proxy", "sp500_proxy", "china_equity_proxy", "eurostoxx50_proxy"}:
        if delta > 0:
            return 1.0, "stronger", dimension
        return -1.0, "weaker", dimension

    if series_id in {"hs300_pe_proxy", "hs300_pb_proxy"}:
        if delta < 0:
            return 1.0, "cheaper", dimension
        return -1.0, "richer", dimension

    if series_id in {"equity_pe_proxy", "equity_pb_proxy", "shiller_pe_proxy"}:
        if delta < 0:
            return 1.0, "cheaper", dimension
        return -1.0, "richer", dimension

    return 0.0, "stable", dimension


def _score_to_direction(score: float) -> str:
    """Map an overlay score into a compact direction label."""
    if score >= 0.35:
        return "risk_on"
    if score <= -0.35:
        return "defensive"
    return "neutral"


def _aggregate_dimension_scores(signal_drivers: list[dict[str, object]]) -> dict[str, float]:
    """Aggregate active signal drivers into dimension-level scores."""
    dimension_scores: dict[str, float] = {}
    for dimension in ["risk", "rates", "inflation"]:
        values = [
            float(item["signal"])
            for item in signal_drivers
            if item.get("dimension") == dimension and float(item["signal"]) != 0.0
        ]
        dimension_scores[dimension] = float(sum(values) / len(values)) if values else 0.0
    return dimension_scores


def _score_to_confidence(active_signals: int, has_newer_market_input: bool) -> str:
    """Map signal breadth and timeliness into a confidence label."""
    if active_signals >= 2 and has_newer_market_input:
        return "high"
    if active_signals >= 1:
        return "medium"
    return "low"


def _read_series_file(path: Path, country: str, source_label: str) -> dict[str, object] | None:
    """Read one raw series file and return a compact status row."""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty or "date" not in frame.columns:
        return None

    dates = pd.to_datetime(frame["date"], errors="coerce")
    values = _safe_numeric(frame["value"]) if "value" in frame.columns else pd.Series(dtype=float)
    enriched = pd.DataFrame({"date": dates, "value": values}).dropna(subset=["date"])
    if enriched.empty:
        return None
    latest_date = enriched["date"].max()

    if "country" in frame.columns:
        countries = frame["country"].dropna().astype(str).unique().tolist()
        if countries and countries[0] not in {country, "global"}:
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

    enriched = enriched.sort_values("date").reset_index(drop=True)
    latest_row = enriched.iloc[-1]
    previous_row = enriched.iloc[-2] if len(enriched) >= 2 else latest_row
    latest_value = latest_row["value"] if pd.notna(latest_row["value"]) else pd.NA
    previous_value = previous_row["value"] if pd.notna(previous_row["value"]) else pd.NA

    return {
        "series_id": series_id,
        "latest_date": latest_date,
        "row_count": int(len(frame)),
        "source_used": source_used,
        "latest_value": latest_value,
        "previous_value": previous_value,
    }


def _source_label_from_folder(folder: Path) -> str:
    """Map one input folder to a stable source label."""
    if "normalized" in folder.parts:
        if "global_markets" in folder.parts:
            return "global_markets"
        return "normalized_api"
    if "fred" in folder.parts:
        return "fred"
    return "manual_fallback"


def collect_country_input_status(country: str) -> pd.DataFrame:
    """Collect actual loaded input status for a country across preferred source paths."""
    folders = [folder for folder in COUNTRY_INPUT_PRIORITY[country] if folder.exists()]
    if not folders:
        return pd.DataFrame(columns=["series_id", "latest_date", "row_count", "source_used", "latest_value", "previous_value"])

    rows: list[dict[str, object]] = []
    for priority, folder in enumerate(folders):
        source_label = _source_label_from_folder(folder)
        for path in sorted(folder.glob("*.csv")):
            if path.stem == "_summary":
                continue
            row = _read_series_file(path, country=country, source_label=source_label)
            if row is not None:
                row["source_priority"] = priority
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["series_id", "latest_date", "row_count", "source_used", "latest_value", "previous_value"])
    output = pd.DataFrame(rows).sort_values(["source_priority", "latest_date"], ascending=[True, False])
    output = output.drop_duplicates(subset=["series_id"], keep="first")
    if "source_priority" in output.columns:
        output = output.drop(columns=["source_priority"])
    return output.sort_values(["latest_date", "series_id"]).reset_index(drop=True)


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
    signal_drivers: list[dict[str, object]] = []
    ignored_drivers: list[dict[str, object]] = []
    for _, row in market_status.iterrows():
        latest_value = row.get("latest_value")
        previous_value = row.get("previous_value")
        if pd.isna(latest_value) or pd.isna(previous_value):
            continue
        signal, driver, dimension = _series_signal(str(row["series_id"]), float(latest_value), float(previous_value))
        payload = {
            "series_id": str(row["series_id"]),
            "signal": signal,
            "driver": driver,
            "dimension": dimension,
            "latest_date": row["latest_date"],
            "latest_value": float(latest_value),
            "previous_value": float(previous_value),
        }
        if pd.notna(regime_date) and pd.notna(row["latest_date"]) and pd.Timestamp(row["latest_date"]) <= pd.Timestamp(regime_date):
            ignored_drivers.append(payload)
            continue
        signal_drivers.append(payload)

    active_signals = [item["signal"] for item in signal_drivers if item["signal"] != 0]
    overlay_score = float(sum(active_signals) / len(active_signals)) if active_signals else 0.0
    overlay_direction = _score_to_direction(overlay_score)
    overlay_confidence = _score_to_confidence(len(active_signals), bool(has_newer_market_input))
    dimension_scores = _aggregate_dimension_scores(signal_drivers)
    return {
        "status": status,
        "system_update_timestamp": latest_processed_timestamp(country),
        "freshest_input_date": freshest_input_date,
        "freshest_market_date": freshest_market_date,
        "freshest_market_series": freshest_market_series,
        "has_newer_market_input": bool(has_newer_market_input),
        "overlay_score": overlay_score,
        "overlay_direction": overlay_direction,
        "overlay_confidence": overlay_confidence,
        "dimension_scores": dimension_scores,
        "signal_drivers": signal_drivers,
        "ignored_drivers": ignored_drivers,
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
    active_country_scores = [overlay["overlay_score"] for overlay in overlays.values() if overlay["has_newer_market_input"]]
    overlay_score = float(sum(active_country_scores) / len(active_country_scores)) if active_country_scores else 0.0
    overlay_direction = _score_to_direction(overlay_score)
    overlay_confidence = _score_to_confidence(len(active_country_scores), bool(countries_with_newer_inputs))
    overlay_drivers: list[str] = []
    dimension_scores: dict[str, float] = {}
    for dimension in ["risk", "rates", "inflation"]:
        scores = [
            float(overlay["dimension_scores"].get(dimension, 0.0))
            for overlay in overlays.values()
            if overlay["has_newer_market_input"]
        ]
        non_zero = [score for score in scores if score != 0.0]
        dimension_scores[dimension] = float(sum(non_zero) / len(non_zero)) if non_zero else 0.0
    for country, overlay in overlays.items():
        for driver in overlay["signal_drivers"]:
            if driver["signal"] == 0:
                continue
            overlay_drivers.append(f"{country}:{driver['series_id']}:{driver['driver']}:{driver['dimension']}")
    return {
        "system_update_timestamp": latest_processed_timestamp(None),
        "freshest_market_date": freshest_market_date,
        "countries_with_newer_inputs": countries_with_newer_inputs,
        "freshest_market_sources": freshest_market_sources,
        "country_overlays": overlays,
        "summary_date": summary_date,
        "overlay_score": overlay_score,
        "overlay_direction": overlay_direction,
        "overlay_confidence": overlay_confidence,
        "dimension_scores": dimension_scores,
        "overlay_drivers": overlay_drivers,
    }
