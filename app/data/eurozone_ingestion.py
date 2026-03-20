"""Eurozone API ingestion orchestration and validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.data.sources.eurozone_ecb_client import fetch_eurozone_ecb_series
from app.data.sources.eurozone_eurostat_client import fetch_eurozone_eurostat_series
from app.utils.config import get_country_indicators

EUROZONE_MINIMUM_REGIME_SERIES = ["cpi", "growth_proxy", "policy_rate", "yield_10y"]
EUROZONE_ENRICHMENT_SERIES = ["industrial_production", "m3", "unrate", "core_cpi"]
EUROZONE_CANONICAL_SERIES = EUROZONE_MINIMUM_REGIME_SERIES + EUROZONE_ENRICHMENT_SERIES
EUROZONE_VALUATION_OPTIONAL_SERIES = ["equity_pe_proxy", "equity_pb_proxy"]
EUROZONE_SERIES_ID_ALIASES = {
    "pmi": "growth_proxy",
    "pmi_or_growth_proxy": "growth_proxy",
    "growth_proxy": "growth_proxy",
    "equity_pe_proxy": "equity_pe_proxy",
    "equity_pb_proxy": "equity_pb_proxy",
    "cpi": "cpi",
    "core_cpi": "core_cpi",
    "m3": "m3",
    "unrate": "unrate",
    "policy_rate": "policy_rate",
    "yield_10y": "yield_10y",
    "industrial_production": "industrial_production",
}
EUROZONE_SOURCE_FETCHERS = {
    "eurozone_ecb": (fetch_eurozone_ecb_series, "ecb"),
    "eurozone_eurostat": (fetch_eurozone_eurostat_series, "eurostat"),
}
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


def _run_timestamp() -> pd.Timestamp:
    """Return a timezone-aware run timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def canonicalize_eurozone_series_id(series_id: object) -> str:
    """Map Eurozone source-specific ids into canonical ids."""
    text = str(series_id or "").strip()
    return EUROZONE_SERIES_ID_ALIASES.get(text, text)


def _empty_frame() -> pd.DataFrame:
    """Return an empty normalized frame."""
    return pd.DataFrame(columns=NORMALIZED_COLUMNS)


def _clean_normalized_frame(frame: pd.DataFrame, indicator_key: str) -> pd.DataFrame:
    """Apply common Eurozone data quality checks."""
    if frame.empty:
        return _empty_frame()
    missing = set(NORMALIZED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing normalized columns for {indicator_key}: {sorted(missing)}")
    cleaned = frame.loc[:, NORMALIZED_COLUMNS].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
    cleaned["series_id"] = cleaned["series_id"].map(canonicalize_eurozone_series_id)
    cleaned["release_date"] = pd.to_datetime(cleaned["release_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "value"])
    cleaned = cleaned.drop_duplicates(subset=["series_id", "date"], keep="last")
    return cleaned.sort_values("date").reset_index(drop=True)


def _save_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Save a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _fetch_indicator(indicator: dict[str, object], country: str) -> tuple[pd.DataFrame, str]:
    """Fetch one Eurozone indicator using configured source and fallback."""
    source = str(indicator.get("source", ""))
    fetcher_entry = EUROZONE_SOURCE_FETCHERS.get(source)
    source_series_id = str(indicator.get("source_series_id") or indicator.get("series_id") or indicator.get("key"))
    source_hint = str(indicator.get("source_hint") or indicator.get("key"))
    if fetcher_entry is None:
        return _empty_frame(), source
    fetcher, _subdir = fetcher_entry
    try:
        frame = fetcher(
            source_series_id=source_series_id,
            country=country,
            frequency=str(indicator.get("frequency", "monthly")),
            source_hint=source_hint,
        )
        return _clean_normalized_frame(frame, str(indicator["key"])), source
    except Exception:
        fallback_source = str(indicator.get("fallback_source") or "")
        if fallback_source in EUROZONE_SOURCE_FETCHERS:
            fallback_fetcher, _ = EUROZONE_SOURCE_FETCHERS[fallback_source]
            try:
                frame = fallback_fetcher(
                    source_series_id=source_series_id,
                    country=country,
                    frequency=str(indicator.get("frequency", "monthly")),
                    source_hint=source_hint,
                )
                return _clean_normalized_frame(frame, str(indicator["key"])), fallback_source
            except Exception:
                return _empty_frame(), fallback_source
        return _empty_frame(), source


def fetch_eurozone_api_bundle(base_dir: str = "data/raw/api/eurozone") -> pd.DataFrame:
    """Fetch Eurozone API-backed indicators and save normalized outputs."""
    root = Path(base_dir)
    normalized_dir = root / "normalized"
    summary_rows: list[dict[str, object]] = []
    indicators = get_country_indicators("eurozone", "macro")
    for indicator in indicators:
        key = canonicalize_eurozone_series_id(indicator["key"])
        source = str(indicator.get("source", ""))
        if source not in EUROZONE_SOURCE_FETCHERS:
            continue
        frame, source_used = _fetch_indicator(indicator, country="eurozone")
        source_subdir = EUROZONE_SOURCE_FETCHERS.get(source_used, (None, "normalized"))[1]
        if frame.empty:
            summary_rows.append(
                {
                    "series_id": key,
                    "source_used": source_used,
                    "row_count": 0,
                    "latest_date": pd.NaT,
                    "required_for_minimum_regime": key in EUROZONE_MINIMUM_REGIME_SERIES,
                    "status": "missing",
                }
            )
            continue
        raw_frame = frame.copy()
        raw_frame["series_id"] = key
        _save_frame(raw_frame, root / source_subdir / f"{key}.csv")
        _save_frame(raw_frame.loc[:, NORMALIZED_COLUMNS], normalized_dir / f"{key}.csv")
        summary_rows.append(
            {
                "series_id": key,
                "source_used": source_used,
                "row_count": len(raw_frame),
                "latest_date": raw_frame["date"].max(),
                "required_for_minimum_regime": key in EUROZONE_MINIMUM_REGIME_SERIES,
                "status": "ready",
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["latest_date"] = pd.to_datetime(summary["latest_date"], errors="coerce")
        summary["ingested_at"] = _run_timestamp()
        _save_frame(summary, normalized_dir / "_summary.csv")
    return summary


def rebuild_eurozone_normalized_data(base_dir: str = "data/raw/api/eurozone") -> pd.DataFrame:
    """Rebuild Eurozone normalized files from raw API source folders."""
    root = Path(base_dir)
    normalized_dir = root / "normalized"
    summary_rows: list[dict[str, object]] = []
    indicators = get_country_indicators("eurozone", "macro")
    for indicator in indicators:
        key = canonicalize_eurozone_series_id(indicator["key"])
        source_candidates = [str(indicator.get("source", "")), str(indicator.get("fallback_source") or "")]
        rebuilt = False
        for source_name in source_candidates:
            if source_name not in EUROZONE_SOURCE_FETCHERS:
                continue
            _, subdir = EUROZONE_SOURCE_FETCHERS[source_name]
            raw_path = root / subdir / f"{key}.csv"
            if not raw_path.exists():
                raw_path = root / subdir / f"{str(indicator['key'])}.csv"
            if not raw_path.exists():
                continue
            frame = pd.read_csv(raw_path)
            normalized = _clean_normalized_frame(frame, key)
            if normalized.empty:
                continue
            normalized["series_id"] = key
            _save_frame(normalized, normalized_dir / f"{key}.csv")
            summary_rows.append(
                {
                    "series_id": key,
                    "source_used": source_name,
                    "row_count": len(normalized),
                    "latest_date": normalized["date"].max(),
                    "required_for_minimum_regime": key in EUROZONE_MINIMUM_REGIME_SERIES,
                    "status": "ready",
                }
            )
            rebuilt = True
            break
        if not rebuilt:
            summary_rows.append(
                {
                    "series_id": key,
                    "source_used": "No loaded data",
                    "row_count": 0,
                    "latest_date": pd.NaT,
                    "required_for_minimum_regime": key in EUROZONE_MINIMUM_REGIME_SERIES,
                    "status": "missing",
                }
            )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["latest_date"] = pd.to_datetime(summary["latest_date"], errors="coerce")
        summary["ingested_at"] = _run_timestamp()
        _save_frame(summary, normalized_dir / "_summary.csv")
    return summary


def validate_eurozone_data(
    base_dir: str = "data/raw/api/eurozone/normalized",
    stale_days_fresh: int = 90,
) -> dict[str, object]:
    """Validate Eurozone normalized inputs for feature, regime, and valuation readiness."""
    normalized_dir = Path(base_dir)
    normalized_files = sorted(
        file_path.name for file_path in normalized_dir.glob("*.csv") if file_path.name != "_summary.csv"
    )
    found_series_ids: list[str] = []
    series_status: list[dict[str, object]] = []
    for key in EUROZONE_CANONICAL_SERIES + EUROZONE_VALUATION_OPTIONAL_SERIES:
        path = normalized_dir / f"{key}.csv"
        if not path.exists():
            series_status.append(
                {
                    "series_id": key,
                    "source_used": "No loaded data",
                    "row_count": 0,
                    "latest_date": pd.NaT,
                    "status": "missing",
                }
            )
            continue
        frame = _clean_normalized_frame(pd.read_csv(path), key)
        if not frame.empty:
            found_series_ids.extend(frame["series_id"].dropna().astype(str).tolist())
        sources = sorted(frame["source"].dropna().astype(str).unique().tolist()) if "source" in frame.columns and not frame.empty else []
        series_status.append(
            {
                "series_id": key,
                "source_used": ", ".join(sources) if sources else "No loaded data",
                "row_count": len(frame),
                "latest_date": frame["date"].max() if not frame.empty else pd.NaT,
                "status": "ready" if not frame.empty else "missing",
            }
        )
    status_frame = pd.DataFrame(series_status)
    available_series = sorted(status_frame.loc[status_frame["status"] == "ready", "series_id"].tolist())
    found_series_ids = sorted(set(canonicalize_eurozone_series_id(item) for item in found_series_ids))
    missing_required = [series for series in EUROZONE_MINIMUM_REGIME_SERIES if series not in available_series]
    enrichment_available = [series for series in EUROZONE_ENRICHMENT_SERIES if series in available_series]
    optional_missing = [series for series in EUROZONE_ENRICHMENT_SERIES if series not in available_series]
    max_date = status_frame["latest_date"].max() if not status_frame.empty else pd.NaT
    enrichment_used: list[str] = []
    enrichment_ignored_stale: list[str] = []
    stale_warning = None
    if pd.notna(max_date):
        days_stale = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(max_date)).days
        if days_stale > stale_days_fresh:
            stale_warning = f"Latest Eurozone API data is {days_stale} days old."
        for _, row in status_frame.iterrows():
            series_id = str(row["series_id"])
            if series_id not in EUROZONE_ENRICHMENT_SERIES or pd.isna(row["latest_date"]):
                continue
            series_stale_days = int((pd.Timestamp(max_date) - pd.Timestamp(row["latest_date"])).days)
            if series_stale_days > 180:
                enrichment_ignored_stale.append(series_id)
            elif row["status"] == "ready":
                enrichment_used.append(series_id)
    else:
        enrichment_used = enrichment_available.copy()
    richness_level = (
        "rich" if len(enrichment_used) >= 3
        else "enhanced_partially_stale" if enrichment_used and enrichment_ignored_stale
        else "enhanced" if enrichment_used
        else "minimum"
    )
    valuation_proxy_inputs_used = [series for series in ["cpi", "policy_rate", "yield_10y"] if series in available_series]
    if len(valuation_proxy_inputs_used) == 3:
        valuation_proxy_inputs_used.extend(["real_yield_proxy", "term_spread"])
    valuation_proxy_inputs_missing = [series for series in ["cpi", "policy_rate", "yield_10y", "equity_pe_proxy", "equity_pb_proxy"] if series not in available_series]
    actual_sources_found = sorted(
        {
            part.strip()
            for value in status_frame.loc[status_frame["status"] == "ready", "source_used"]
            .replace("No loaded data", pd.NA)
            .dropna()
            .astype(str)
            .tolist()
            for part in value.split(",")
            if part.strip()
        }
    )
    return {
        "country": "eurozone",
        "loaded_data_path": str(normalized_dir),
        "series_status": status_frame,
        "normalized_files_found": normalized_files,
        "series_ids_found": found_series_ids,
        "available_series": available_series,
        "actual_sources_found": actual_sources_found,
        "missing_required_series": missing_required,
        "minimum_inputs_used": [series for series in EUROZONE_MINIMUM_REGIME_SERIES if series in available_series],
        "enrichment_available_series": enrichment_available,
        "enrichment_inputs_used": enrichment_used,
        "enrichment_inputs_ignored_stale": enrichment_ignored_stale,
        "optional_missing_series": optional_missing,
        "scoring_richness_level": richness_level,
        "feature_build_ready": bool(available_series),
        "regime_ready": not bool(missing_required),
        "valuation_loaded_data_path": str(normalized_dir),
        "valuation_normalized_files_found": normalized_files,
        "valuation_canonical_series_ids_found": [series for series in ["cpi", "policy_rate", "yield_10y", "equity_pe_proxy", "equity_pb_proxy"] if series in available_series],
        "valuation_actual_sources_found": actual_sources_found,
        "valuation_proxy_inputs_used": valuation_proxy_inputs_used,
        "valuation_proxy_inputs_missing": valuation_proxy_inputs_missing,
        "valuation_proxy_series_found": [series for series in ["cpi", "policy_rate", "yield_10y", "equity_pe_proxy", "equity_pb_proxy"] if series in available_series],
        "valuation_proxy_readiness": all(series in available_series for series in ["cpi", "policy_rate", "yield_10y"]),
        "valuation_ready": all(series in available_series for series in ["cpi", "policy_rate", "yield_10y"]),
        "stale_warning": stale_warning,
        "api_first_mode_active": bool(normalized_files),
    }
