"""Valuation feature engineering for country-level macro monitoring."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from app.data.fetchers import fetch_fred_series
from app.data.sources.international_market_client import fetch_international_market_series
from app.data.sources.us_market_client import fetch_us_market_series
from app.valuation.china_models import (
    compute_china_valuation_confidence,
    compute_china_valuation_score,
    label_china_valuation_regime,
)
from app.valuation.eurozone_models import (
    compute_eurozone_valuation_confidence,
    compute_eurozone_valuation_score,
    label_eurozone_valuation_regime,
)
from app.valuation.models import (
    compute_valuation_confidence,
    compute_valuation_score,
    label_valuation_regime,
    summarize_valuation_inputs,
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
CHINA_VALUATION_OPTIONAL_INPUT_IDS = ["hs300_pe_proxy", "hs300_pb_proxy", "shiller_pe_proxy"]
US_CANONICAL_INPUT_IDS = [
    "cpi",
    "policy_rate",
    "yield_10y",
    "buffett_indicator",
    "equity_pe_proxy",
    "shiller_pe_proxy",
    "equity_pb_proxy",
    "earnings_yield_proxy",
    "credit_spread_proxy",
]
US_VALUATION_REQUIRED_INPUT_IDS = ["cpi", "policy_rate", "yield_10y"]
US_VALUATION_OPTIONAL_INPUT_IDS = [
    "buffett_indicator",
    "equity_pe_proxy",
    "shiller_pe_proxy",
    "equity_pb_proxy",
    "earnings_yield_proxy",
    "credit_spread_proxy",
]
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
EUROZONE_VALUATION_OPTIONAL_INPUT_IDS = ["equity_pe_proxy", "shiller_pe_proxy", "equity_pb_proxy"]
VALUATION_EXPECTED_INPUTS = {
    "us": [
        "buffett_indicator",
        "equity_pe_proxy",
        "shiller_pe_proxy",
        "equity_pb_proxy",
        "real_yield_proxy",
        "term_spread",
        "equity_risk_proxy",
        "credit_spread_proxy",
    ],
    "china": [
        "equity_pe_proxy",
        "shiller_pe_proxy",
        "equity_pb_proxy",
        "real_yield_proxy",
        "term_spread",
        "equity_risk_proxy",
    ],
    "eurozone": [
        "equity_pe_proxy",
        "shiller_pe_proxy",
        "real_yield_proxy",
        "term_spread",
        "equity_risk_proxy",
    ],
}

VALUATION_COLUMNS = [
    "date",
    "country",
    "buffett_indicator",
    "real_yield",
    "term_spread",
    "equity_risk_proxy",
    "credit_spread_proxy",
    "earnings_yield_proxy",
    "hs300_pe_proxy",
    "hs300_pb_proxy",
    "equity_pe_proxy",
    "shiller_pe_proxy",
    "equity_pb_proxy",
    "real_yield_proxy",
    "valuation_method",
    "valuation_score",
    "valuation_regime",
    "valuation_confidence",
    "valuation_inputs_used",
    "valuation_inputs_missing",
]


def load_macro_feature_frame(path: str) -> pd.DataFrame:
    """Load a processed country macro feature dataset."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def _empty_series(index: pd.Index) -> pd.Series:
    """Create an all-NaN float series."""
    return pd.Series(float("nan"), index=index, dtype="float64")


def _normalized_api_path(api_dir: str, country: str, series_id: str) -> Path:
    """Return the normalized API path for one country series."""
    return Path(api_dir) / country / "normalized" / f"{series_id}.csv"


def _normalize_optional_frame(
    frame: pd.DataFrame,
    series_id: str,
    source_name: str,
    country: str,
    frequency: str,
) -> pd.DataFrame:
    """Normalize an optional valuation frame to the shared schema."""
    if frame.empty:
        return pd.DataFrame(
            columns=["date", "value", "series_id", "country", "source", "frequency", "release_date", "ingested_at"]
        )

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    if "release_date" not in normalized.columns:
        normalized["release_date"] = normalized["date"]
    normalized["release_date"] = pd.to_datetime(normalized["release_date"], errors="coerce")
    normalized["series_id"] = series_id
    normalized["country"] = country
    normalized["source"] = source_name
    normalized["frequency"] = frequency
    if "ingested_at" not in normalized.columns:
        normalized["ingested_at"] = pd.Timestamp.utcnow()
    normalized = normalized.dropna(subset=["date", "value"])
    normalized = normalized.sort_values(["date", "release_date"]).drop_duplicates(
        subset=["series_id", "date"], keep="last"
    )
    return normalized.loc[
        :,
        ["date", "value", "series_id", "country", "source", "frequency", "release_date", "ingested_at"],
    ].reset_index(drop=True)


def _save_normalized_optional_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Save a normalized optional valuation frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def refresh_us_valuation_source_data(
    fred_dir: str = "data/raw/fred",
    api_dir: str = "data/raw/api",
) -> dict[str, object]:
    """Refresh US valuation source files from FRED and public market pages."""
    normalized_dir = Path(api_dir) / "us" / "normalized"
    raw_fred_dir = Path(api_dir) / "us" / "fred"
    raw_multpl_dir = Path(api_dir) / "us" / "multpl"
    normalized_files: list[str] = []
    actual_sources: set[str] = set()
    series_ids_found: list[str] = []

    api_key = os.getenv("FRED_API_KEY")
    for indicator in get_country_indicators("us", "valuation"):
        key = str(indicator["key"])
        source = str(indicator.get("source", "manual"))
        source_series_id = str(indicator.get("source_series_id") or indicator.get("series_id") or key)
        frequency = str(indicator.get("frequency", "monthly"))
        normalized_path = normalized_dir / f"{key}.csv"

        try:
            if source == "us_fred" and api_key:
                frame = fetch_fred_series(source_series_id, api_key=api_key)
                normalized = _normalize_optional_frame(
                    frame=frame,
                    series_id=key,
                    source_name="fred",
                    country="us",
                    frequency=frequency,
                )
                _save_normalized_optional_frame(normalized, normalized_path)
                _save_normalized_optional_frame(normalized, raw_fred_dir / f"{key}.csv")
            elif source == "us_multpl":
                normalized = fetch_us_market_series(
                    source_series_id=key,
                    country="us",
                    frequency=frequency,
                    source_hint=source_series_id,
                )
                _save_normalized_optional_frame(normalized, normalized_path)
                _save_normalized_optional_frame(normalized, raw_multpl_dir / f"{key}.csv")
            else:
                continue
        except Exception:
            continue

        if normalized_path.exists():
            normalized_files.append(normalized_path.name)
            series_ids_found.append(key)
            if not normalized.empty:
                actual_sources.update(normalized["source"].dropna().astype(str).tolist())

    return {
        "loaded_data_path": str(normalized_dir),
        "normalized_files_found": sorted(set(normalized_files)),
        "canonical_series_ids_found": sorted(set(series_ids_found)),
        "actual_sources_found": sorted(actual_sources),
    }


def refresh_international_valuation_source_data(
    country: str,
    api_dir: str = "data/raw/api",
) -> dict[str, object]:
    """Refresh China or Eurozone market valuation proxies from public market pages."""
    normalized_dir = Path(api_dir) / country / "normalized"
    normalized_files: list[str] = []
    actual_sources: set[str] = set()
    series_ids_found: list[str] = []

    source_to_subdir = {
        "siblis_market": "siblis",
        "public_site": "public",
    }

    for indicator in get_country_indicators(country, "valuation"):
        source = str(indicator.get("source", "manual"))
        if source not in source_to_subdir:
            continue
        key = str(indicator["key"])
        source_series_id = str(indicator.get("source_series_id") or indicator.get("series_id") or key)
        frequency = str(indicator.get("frequency", "monthly"))
        normalized_path = normalized_dir / f"{key}.csv"

        try:
            if source == "siblis_market":
                normalized = fetch_international_market_series(
                    source_series_id=source_series_id,
                    country=country,
                    frequency=frequency,
                )
            else:
                from app.data.sources.public_site_client import fetch_public_site_series

                normalized = fetch_public_site_series(
                    source_series_id=source_series_id,
                    country=country,
                    frequency=frequency,
                )
        except Exception:
            continue

        if normalized.empty:
            continue

        _save_normalized_optional_frame(normalized, normalized_path)
        _save_normalized_optional_frame(normalized, Path(api_dir) / country / source_to_subdir[source] / f"{key}.csv")
        normalized_files.append(normalized_path.name)
        series_ids_found.append(key)
        actual_sources.update(normalized["source"].dropna().astype(str).tolist())

    return {
        "loaded_data_path": str(normalized_dir),
        "normalized_files_found": sorted(set(normalized_files)),
        "canonical_series_ids_found": sorted(set(series_ids_found)),
        "actual_sources_found": sorted(actual_sources),
    }


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
    elif source in {"us_fred", "us_multpl", "siblis_market", "public_site"}:
        path = _normalized_api_path(api_dir, country, series_id)
    elif source in {"tushare", "china_akshare", "china_nbs", "china_rates", "imf", "eurozone_ecb", "eurozone_eurostat", "eurozone_oecd"}:
        path = _normalized_api_path(api_dir, country, series_id)
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


def _merge_optional_series_asof(
    valuation: pd.DataFrame,
    optional_series: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    """Merge a low-frequency valuation series using monthly as-of alignment."""
    if optional_series.empty:
        return valuation

    left = valuation.loc[:, ["date"]].copy().sort_values("date").reset_index(drop=True)
    right = optional_series.rename(columns={"value": key}).loc[:, ["date", key]].copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce")
    right[key] = pd.to_numeric(right[key], errors="coerce")
    right = right.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if right.empty:
        return valuation

    merged = pd.merge_asof(left, right, on="date", direction="backward")
    valuation[key] = merged[key]
    return valuation


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


def inspect_us_valuation_inputs(
    api_dir: str = "data/raw/api",
) -> dict[str, object]:
    """Inspect US valuation inputs from the normalized API-first directory."""
    normalized_dir = Path(api_dir) / "us" / "normalized"
    normalized_files_found = sorted(
        path.name for path in normalized_dir.glob("*.csv") if path.name != "_summary.csv"
    )
    series_ids_found: list[str] = []
    source_by_series: dict[str, str] = {}
    for series_id in US_VALUATION_OPTIONAL_INPUT_IDS:
        path = normalized_dir / f"{series_id}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if {"date", "value"}.difference(frame.columns) or frame.empty:
            continue
        series_ids_found.append(series_id)
        if "source" in frame.columns and frame["source"].notna().any():
            source_by_series[series_id] = str(frame["source"].dropna().iloc[-1])

    proxy_inputs_used = ["cpi", "policy_rate", "yield_10y", "real_yield_proxy", "term_spread"]
    proxy_inputs_missing: list[str] = []
    for series_id in US_VALUATION_OPTIONAL_INPUT_IDS:
        if series_id in series_ids_found:
            proxy_inputs_used.append(series_id)
        else:
            proxy_inputs_missing.append(series_id)
    if "earnings_yield_proxy" in series_ids_found:
        proxy_inputs_used.append("equity_risk_proxy")
    else:
        proxy_inputs_missing.append("equity_risk_proxy")

    return {
        "loaded_data_path": str(normalized_dir),
        "normalized_files_found": normalized_files_found,
        "canonical_series_ids_found": sorted(series_ids_found),
        "actual_sources_found": sorted(set(source_by_series.values())),
        "proxy_inputs_used": proxy_inputs_used,
        "proxy_inputs_missing": proxy_inputs_missing,
        "valuation_ready": len(series_ids_found) > 0,
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
    valuation = valuation.sort_values("date").reset_index(drop=True)
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
    valuation["credit_spread_proxy"] = _empty_series(index)
    valuation["earnings_yield_proxy"] = _empty_series(index)
    valuation["hs300_pe_proxy"] = _empty_series(index)
    valuation["hs300_pb_proxy"] = _empty_series(index)
    valuation["equity_pe_proxy"] = _empty_series(index)
    valuation["shiller_pe_proxy"] = _empty_series(index)
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
        valuation = _merge_optional_series_asof(valuation, optional_series, key)

    if valuation["equity_risk_proxy"].isna().all() and valuation["earnings_yield_proxy"].notna().any():
        valuation["equity_risk_proxy"] = valuation["earnings_yield_proxy"] - yield_10y
    if valuation["equity_risk_proxy"].isna().all() and valuation["equity_pe_proxy"].notna().any():
        valuation["equity_risk_proxy"] = (100.0 / valuation["equity_pe_proxy"]) - yield_10y
    if valuation["equity_risk_proxy"].isna().all() and valuation["buffett_indicator"].notna().any():
        valuation["equity_risk_proxy"] = 1.0 / valuation["buffett_indicator"]
    valuation["real_yield_proxy"] = valuation["real_yield"]
    if country == "china":
        valuation["equity_pe_proxy"] = valuation["equity_pe_proxy"].combine_first(valuation["hs300_pe_proxy"])
        valuation["equity_pb_proxy"] = valuation["equity_pb_proxy"].combine_first(valuation["hs300_pb_proxy"])
        if valuation["equity_risk_proxy"].isna().all() and valuation["equity_pe_proxy"].notna().any():
            valuation["equity_risk_proxy"] = (100.0 / valuation["equity_pe_proxy"]) - yield_10y
        valuation["valuation_method"] = "research_proxy_cn"
        valuation["valuation_score"] = compute_china_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_china_valuation_regime)
    elif country == "eurozone":
        valuation["valuation_method"] = "research_proxy_eurozone"
        valuation["valuation_score"] = compute_eurozone_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_eurozone_valuation_regime)
    else:
        valuation["valuation_method"] = "research_proxy"
        valuation["valuation_score"] = compute_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_valuation_regime)

    expected_inputs = VALUATION_EXPECTED_INPUTS.get(country, VALUATION_EXPECTED_INPUTS["us"])
    if country == "china":
        valuation["valuation_confidence"] = compute_china_valuation_confidence(valuation)
    elif country == "eurozone":
        valuation["valuation_confidence"] = compute_eurozone_valuation_confidence(valuation)
    else:
        valuation["valuation_confidence"] = compute_valuation_confidence(
            valuation,
            expected_inputs=expected_inputs,
        )
    valuation["valuation_inputs_used"], valuation["valuation_inputs_missing"] = summarize_valuation_inputs(
        valuation,
        expected_inputs=expected_inputs,
    )

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
    if country == "us":
        refresh_us_valuation_source_data(fred_dir=fred_dir, api_dir=api_dir)
    elif country in {"china", "eurozone"}:
        refresh_international_valuation_source_data(country=country, api_dir=api_dir)
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
    api_dir: str = "data/raw/api",
) -> pd.DataFrame:
    """Backward-compatible wrapper for the US valuation pipeline."""
    return build_country_valuation_features(
        country="us",
        macro_feature_path=macro_feature_path,
        output_path=output_path,
        manual_dir=manual_dir,
        fred_dir=fred_dir,
        api_dir=api_dir,
    )
