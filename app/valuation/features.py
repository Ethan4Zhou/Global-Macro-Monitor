"""Valuation feature engineering for country-level macro monitoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.valuation.china_models import compute_china_valuation_score, label_china_valuation_regime
from app.valuation.eurozone_models import (
    compute_eurozone_valuation_score,
    label_eurozone_valuation_regime,
)
from app.utils.config import get_country_indicators

CHINA_CANONICAL_INPUT_IDS = [
    "cpi",
    "pmi",
    "policy_rate",
    "yield_10y",
    "m2",
    "industrial_production",
    "core_cpi",
    "unrate",
]
CHINA_VALUATION_REQUIRED_INPUT_IDS = ["cpi", "policy_rate", "yield_10y"]
CHINA_VALUATION_OPTIONAL_INPUT_IDS = ["hs300_pe_proxy", "hs300_pb_proxy"]
EUROZONE_CANONICAL_INPUT_IDS = [
    "cpi",
    "pmi_or_growth_proxy",
    "policy_rate",
    "yield_10y",
    "m3",
    "industrial_production",
    "core_cpi",
    "unrate",
    "sentiment",
]
EUROZONE_VALUATION_REQUIRED_INPUT_IDS = ["cpi", "policy_rate", "yield_10y"]
EUROZONE_VALUATION_OPTIONAL_INPUT_IDS = ["equity_pe_proxy", "equity_pb_proxy"]

VALUATION_COLUMNS = [
    "date",
    "country",
    "buffett_indicator",
    "real_yield",
    "term_spread",
    "equity_risk_proxy",
    "credit_spread_proxy",
    "hs300_pe_proxy",
    "hs300_pb_proxy",
    "equity_pe_proxy",
    "equity_pb_proxy",
    "real_yield_proxy",
    "valuation_method",
    "valuation_score",
    "valuation_regime",
]


def load_macro_feature_frame(path: str) -> pd.DataFrame:
    """Load a processed country macro feature dataset."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def _empty_series(index: pd.Index) -> pd.Series:
    """Create an all-NaN float series."""
    return pd.Series(float("nan"), index=index, dtype="float64")


def _load_optional_series(
    country: str,
    source: str,
    series_id: str,
    manual_dir: str = "data/raw/manual",
    fred_dir: str = "data/raw/fred",
    api_dir: str = "data/raw/api",
) -> pd.DataFrame | None:
    """Load one optional valuation series for a country."""
    if source == "manual":
        path = Path(manual_dir) / country / f"{series_id}.csv"
    elif source == "fred":
        path = Path(fred_dir) / f"{series_id}.csv"
    elif source in {"china_akshare", "china_nbs", "china_rates", "imf", "eurozone_ecb", "eurozone_eurostat", "eurozone_oecd"}:
        path = Path(api_dir) / country / "normalized" / f"{series_id}.csv"
    else:
        return None

    if not path.exists():
        return None

    frame = pd.read_csv(path)
    required_columns = {"date", "value"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    series_frame = frame.loc[:, ["date", "value"]].copy()
    series_frame["date"] = pd.to_datetime(series_frame["date"], errors="coerce")
    series_frame["value"] = pd.to_numeric(series_frame["value"], errors="coerce")
    return series_frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def inspect_china_valuation_inputs(
    api_dir: str = "data/raw/api",
    manual_dir: str = "data/raw/manual",
) -> dict[str, object]:
    """Inspect China valuation inputs from normalized API first, then manual fallback."""
    normalized_dir = Path(api_dir) / "china" / "normalized"
    manual_china_dir = Path(manual_dir) / "china"
    normalized_files_found = sorted(
        path.name for path in normalized_dir.glob("*.csv") if path.name != "_summary.csv"
    )
    canonical_series_ids_found: list[str] = []
    source_by_series: dict[str, str] = {}

    for series_id in CHINA_VALUATION_REQUIRED_INPUT_IDS + CHINA_VALUATION_OPTIONAL_INPUT_IDS:
        normalized_path = normalized_dir / f"{series_id}.csv"
        manual_path = manual_china_dir / f"{series_id}.csv"
        chosen_path = normalized_path if normalized_path.exists() else manual_path if manual_path.exists() else None
        if chosen_path is None:
            continue
        frame = pd.read_csv(chosen_path)
        if {"date", "value"}.difference(frame.columns):
            continue
        if frame.empty:
            continue
        if "series_id" in frame.columns:
            raw_ids = set(frame["series_id"].dropna().astype(str).tolist())
            if raw_ids and raw_ids != {series_id}:
                continue
        canonical_series_ids_found.append(series_id)
        source_by_series[series_id] = (
            "normalized_api" if chosen_path.parent == normalized_dir else "manual_fallback"
        )

    proxy_inputs_used: list[str] = []
    proxy_inputs_missing: list[str] = []
    for series_id in CHINA_VALUATION_REQUIRED_INPUT_IDS:
        if series_id in canonical_series_ids_found:
            proxy_inputs_used.append(series_id)
        else:
            proxy_inputs_missing.append(series_id)
    if all(series_id in canonical_series_ids_found for series_id in CHINA_VALUATION_REQUIRED_INPUT_IDS):
        proxy_inputs_used.extend(["real_yield_proxy", "term_spread"])
    for series_id in CHINA_VALUATION_OPTIONAL_INPUT_IDS:
        if series_id in canonical_series_ids_found:
            proxy_inputs_used.append(series_id)
        else:
            proxy_inputs_missing.append(series_id)

    return {
        "loaded_data_path": str(normalized_dir),
        "normalized_files_found": normalized_files_found,
        "canonical_series_ids_found": sorted(canonical_series_ids_found),
        "actual_sources_found": sorted(set(source_by_series.values())),
        "proxy_inputs_used": proxy_inputs_used,
        "proxy_inputs_missing": proxy_inputs_missing,
        "valuation_ready": all(
            series_id in canonical_series_ids_found
            for series_id in CHINA_VALUATION_REQUIRED_INPUT_IDS
        ),
        "source_by_series": source_by_series,
    }


def inspect_eurozone_valuation_inputs(
    api_dir: str = "data/raw/api",
    manual_dir: str = "data/raw/manual",
) -> dict[str, object]:
    """Inspect Eurozone valuation inputs from normalized API first, then manual fallback."""
    normalized_dir = Path(api_dir) / "eurozone" / "normalized"
    manual_dir_country = Path(manual_dir) / "eurozone"
    normalized_files_found = sorted(
        path.name for path in normalized_dir.glob("*.csv") if path.name != "_summary.csv"
    )
    canonical_series_ids_found: list[str] = []
    source_by_series: dict[str, str] = {}
    for series_id in EUROZONE_VALUATION_REQUIRED_INPUT_IDS + EUROZONE_VALUATION_OPTIONAL_INPUT_IDS:
        normalized_path = normalized_dir / f"{series_id}.csv"
        manual_path = manual_dir_country / f"{series_id}.csv"
        chosen_path = normalized_path if normalized_path.exists() else manual_path if manual_path.exists() else None
        if chosen_path is None:
            continue
        frame = pd.read_csv(chosen_path)
        if {"date", "value"}.difference(frame.columns) or frame.empty:
            continue
        if "series_id" in frame.columns:
            raw_ids = set(frame["series_id"].dropna().astype(str).tolist())
            if raw_ids and raw_ids != {series_id}:
                continue
        canonical_series_ids_found.append(series_id)
        source_by_series[series_id] = "normalized_api" if chosen_path.parent == normalized_dir else "manual_fallback"
    proxy_inputs_used: list[str] = []
    proxy_inputs_missing: list[str] = []
    for series_id in EUROZONE_VALUATION_REQUIRED_INPUT_IDS:
        if series_id in canonical_series_ids_found:
            proxy_inputs_used.append(series_id)
        else:
            proxy_inputs_missing.append(series_id)
    if all(series_id in canonical_series_ids_found for series_id in EUROZONE_VALUATION_REQUIRED_INPUT_IDS):
        proxy_inputs_used.extend(["real_yield_proxy", "term_spread"])
    for series_id in EUROZONE_VALUATION_OPTIONAL_INPUT_IDS:
        if series_id in canonical_series_ids_found:
            proxy_inputs_used.append(series_id)
        else:
            proxy_inputs_missing.append(series_id)
    return {
        "loaded_data_path": str(normalized_dir),
        "normalized_files_found": normalized_files_found,
        "canonical_series_ids_found": sorted(canonical_series_ids_found),
        "actual_sources_found": sorted(set(source_by_series.values())),
        "proxy_inputs_used": proxy_inputs_used,
        "proxy_inputs_missing": proxy_inputs_missing,
        "valuation_ready": all(
            series_id in canonical_series_ids_found
            for series_id in EUROZONE_VALUATION_REQUIRED_INPUT_IDS
        ),
        "source_by_series": source_by_series,
    }


def build_country_valuation_features_frame(
    macro_features: pd.DataFrame,
    country: str,
    manual_dir: str = "data/raw/manual",
    fred_dir: str = "data/raw/fred",
    api_dir: str = "data/raw/api",
) -> pd.DataFrame:
    """Build the combined valuation feature dataset for one country."""
    if macro_features.empty:
        return pd.DataFrame(columns=VALUATION_COLUMNS)

    valuation = pd.DataFrame(
        {
            "date": pd.to_datetime(macro_features["date"], errors="coerce"),
            "country": country,
        }
    )
    index = valuation.index
    cpi_yoy = (
        pd.to_numeric(macro_features["cpi_yoy"], errors="coerce")
        if "cpi_yoy" in macro_features.columns
        else _empty_series(index)
    )
    if "policy_rate_level" in macro_features.columns:
        policy_rate = pd.to_numeric(macro_features["policy_rate_level"], errors="coerce")
    elif "fedfunds_level" in macro_features.columns:
        policy_rate = pd.to_numeric(macro_features["fedfunds_level"], errors="coerce")
    else:
        policy_rate = _empty_series(index)

    if "yield_10y_level" in macro_features.columns:
        yield_10y = pd.to_numeric(macro_features["yield_10y_level"], errors="coerce")
    elif "gs10_level" in macro_features.columns:
        yield_10y = pd.to_numeric(macro_features["gs10_level"], errors="coerce")
    else:
        yield_10y = _empty_series(index)

    valuation["buffett_indicator"] = _empty_series(index)
    valuation["real_yield"] = yield_10y - cpi_yoy
    valuation["term_spread"] = yield_10y - policy_rate
    valuation["equity_risk_proxy"] = _empty_series(index)
    valuation["credit_spread_proxy"] = policy_rate - yield_10y
    valuation["hs300_pe_proxy"] = _empty_series(index)
    valuation["hs300_pb_proxy"] = _empty_series(index)
    valuation["equity_pe_proxy"] = _empty_series(index)
    valuation["equity_pb_proxy"] = _empty_series(index)
    valuation["real_yield_proxy"] = valuation["real_yield"]
    valuation["valuation_method"] = "standard"

    for indicator in get_country_indicators(country, "valuation"):
        key = str(indicator["key"])
        source = str(indicator.get("source", "manual"))
        if source == "derived":
            continue
        optional_series = _load_optional_series(
            country=country,
            source=source,
            series_id=str(indicator.get("series_id", key)),
            manual_dir=manual_dir,
            fred_dir=fred_dir,
            api_dir=api_dir,
        )
        if optional_series is None:
            continue
        series_frame = optional_series.rename(columns={"value": key})
        valuation = valuation.merge(series_frame, on="date", how="left", suffixes=("", "_manual"))
        manual_column = f"{key}_manual"
        if manual_column in valuation.columns:
            valuation[key] = valuation[manual_column].combine_first(valuation[key])
            valuation = valuation.drop(columns=[manual_column])

    if valuation["equity_risk_proxy"].isna().all() and valuation["buffett_indicator"].notna().any():
        valuation["equity_risk_proxy"] = 1.0 / valuation["buffett_indicator"]
    if country == "china":
        if valuation["equity_risk_proxy"].isna().all() and valuation["hs300_pe_proxy"].notna().any():
            valuation["equity_risk_proxy"] = 1.0 / valuation["hs300_pe_proxy"]
        valuation["valuation_method"] = "proxy_based"
        valuation["valuation_score"] = compute_china_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_china_valuation_regime)
    elif country == "eurozone":
        valuation["valuation_method"] = "proxy_based"
        valuation["real_yield_proxy"] = valuation["real_yield"]
        valuation["valuation_score"] = compute_eurozone_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_eurozone_valuation_regime)
    else:
        valuation["valuation_score"] = _empty_series(index)
        valuation["valuation_regime"] = pd.Series(pd.NA, index=index, dtype="object")

    return valuation.loc[:, VALUATION_COLUMNS].sort_values("date").reset_index(drop=True)


def save_country_valuation_features(
    frame: pd.DataFrame,
    country: str,
    output_path: str | None = None,
) -> Path:
    """Save the processed country valuation feature dataset."""
    destination = Path(output_path or f"data/processed/{country}_valuation_features.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def build_country_valuation_features(
    country: str,
    macro_feature_path: str | None = None,
    output_path: str | None = None,
    manual_dir: str = "data/raw/manual",
    fred_dir: str = "data/raw/fred",
    api_dir: str = "data/raw/api",
) -> pd.DataFrame:
    """Load macro features, combine optional valuation inputs, and save the result."""
    path = macro_feature_path or f"data/processed/{country}_macro_features.csv"
    macro_features = load_macro_feature_frame(path=path)
    valuation = build_country_valuation_features_frame(
        macro_features=macro_features,
        country=country,
        manual_dir=manual_dir,
        fred_dir=fred_dir,
        api_dir=api_dir,
    )
    save_country_valuation_features(valuation, country=country, output_path=output_path)
    return valuation


def build_us_valuation_features(
    macro_feature_path: str = "data/processed/us_macro_features.csv",
    output_path: str = "data/processed/us_valuation_features.csv",
    manual_dir: str = "data/raw/manual",
    fred_dir: str = "data/raw/fred",
) -> pd.DataFrame:
    """Backward-compatible wrapper for the US valuation pipeline."""
    return build_country_valuation_features(
        country="us",
        macro_feature_path=macro_feature_path,
        output_path=output_path,
        manual_dir=manual_dir,
        fred_dir=fred_dir,
    )
