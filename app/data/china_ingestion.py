"""China API ingestion orchestration and validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.data.sources.china_nbs_client import fetch_china_nbs_series
from app.data.sources.china_rates_client import fetch_china_rates_series
from app.data.sources.china_akshare_client import fetch_china_akshare_series
from app.data.sources.international_market_client import fetch_international_market_series
from app.data.sources.imf_client import fetch_imf_series
from app.data.sources.public_site_client import fetch_public_site_series
from app.data.sources.tushare_client import fetch_tushare_series
from app.valuation.features import inspect_china_valuation_inputs
from app.utils.config import get_country_indicators
from app.utils.logging import get_logger

logger = get_logger(__name__)

CHINA_MINIMUM_REGIME_SERIES = ["cpi", "pmi", "policy_rate", "yield_10y"]
CHINA_ENRICHMENT_SERIES = ["industrial_production", "m2", "core_cpi", "unrate"]
CHINA_CANONICAL_MACRO_SERIES = CHINA_MINIMUM_REGIME_SERIES + CHINA_ENRICHMENT_SERIES
CHINA_VALUATION_PROXY_SERIES = [
    "hs300_pe_proxy",
    "hs300_pb_proxy",
    "shiller_pe_proxy",
    "real_yield_proxy",
    "term_spread",
]
CHINA_SERIES_ID_ALIASES = {
    "cpi": "cpi",
    "consumer_price_index": "cpi",
    "cn_cpi": "cpi",
    "pmi": "pmi",
    "manufacturing_pmi": "pmi",
    "cn_pmi": "pmi",
    "shibor_lpr": "policy_rate",
    "policy_rate": "policy_rate",
    "policy_rate_proxy": "policy_rate",
    "short_rate_proxy": "policy_rate",
    "dr007_or_repo_proxy": "policy_rate",
    "yc_cb_10y": "yield_10y",
    "yield_10y": "yield_10y",
    "cgb_10y_proxy": "yield_10y",
    "m2": "m2",
    "money_supply_m2": "m2",
    "cn_m": "m2",
    "industrial_production": "industrial_production",
    "industrial_output": "industrial_production",
    "core_cpi": "core_cpi",
    "unrate": "unrate",
    "urban_unemployment": "unrate",
    "hs300_pe_proxy": "hs300_pe_proxy",
    "hs300_pb_proxy": "hs300_pb_proxy",
    "shiller_pe_proxy": "shiller_pe_proxy",
}
CHINA_SOURCE_FETCHERS = {
    "tushare": (fetch_tushare_series, "tushare"),
    "china_akshare": (fetch_china_akshare_series, "akshare"),
    "china_nbs": (fetch_china_nbs_series, "nbs"),
    "china_rates": (fetch_china_rates_series, "rates"),
    "imf": (fetch_imf_series, "imf"),
    "siblis_market": (fetch_international_market_series, "siblis"),
    "public_site": (fetch_public_site_series, "public"),
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


def _empty_normalized_frame() -> pd.DataFrame:
    """Return an empty normalized frame."""
    return pd.DataFrame(columns=NORMALIZED_COLUMNS)


def canonicalize_china_series_id(series_id: object) -> str:
    """Map source-specific or legacy China series ids into canonical ids."""
    text = str(series_id or "").strip()
    return CHINA_SERIES_ID_ALIASES.get(text, text)


def _clean_normalized_frame(frame: pd.DataFrame, indicator_key: str) -> pd.DataFrame:
    """Apply lightweight data quality checks to a normalized source frame."""
    if frame.empty:
        return _empty_normalized_frame()

    missing = set(NORMALIZED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing normalized columns for {indicator_key}: {sorted(missing)}")

    cleaned = frame.loc[:, NORMALIZED_COLUMNS].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
    cleaned["series_id"] = cleaned["series_id"].map(canonicalize_china_series_id)
    if "release_date" in cleaned.columns:
        cleaned["release_date"] = pd.to_datetime(cleaned["release_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "value"])
    cleaned = cleaned.drop_duplicates(subset=["series_id", "date"], keep="last")
    return cleaned.sort_values("date").reset_index(drop=True)


def _save_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Persist a frame to CSV, creating parents when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _latest_date(frame: pd.DataFrame) -> pd.Timestamp:
    """Return the latest normalized observation date for one frame."""
    if frame.empty or "date" not in frame.columns:
        return pd.NaT
    return pd.to_datetime(frame["date"], errors="coerce").max()


def _fetch_indicator(indicator: dict[str, object], country: str) -> tuple[pd.DataFrame, str]:
    """Fetch one China indicator using its configured source and fallback."""
    source = str(indicator.get("source", ""))
    source_series_id = str(indicator.get("source_series_id") or indicator.get("series_id") or indicator.get("key"))
    source_hint = str(indicator.get("source_hint") or source_series_id)
    fetcher_entry = CHINA_SOURCE_FETCHERS.get(source)
    if fetcher_entry is None:
        return _empty_normalized_frame(), source

    fetcher, _subdir = fetcher_entry
    primary_frame = _empty_normalized_frame()
    try:
        primary_frame = fetcher(
            source_series_id=source_series_id,
            country=country,
            frequency=str(indicator.get("frequency", "monthly")),
            source_hint=source_hint,
        )
        primary_frame = _clean_normalized_frame(primary_frame, str(indicator["key"]))
    except Exception as exc:
        fallback_source = str(indicator.get("fallback_source") or "")
        if fallback_source in CHINA_SOURCE_FETCHERS:
            fallback_series_id = str(indicator.get("fallback_source_series_id") or source_series_id)
            fallback_source_hint = str(indicator.get("fallback_source_hint") or source_hint)
            logger.warning(
                "Primary China source failed for %s/%s, trying fallback source %s.",
                country,
                indicator.get("key"),
                fallback_source,
            )
            fallback_fetcher, _subdir = CHINA_SOURCE_FETCHERS[fallback_source]
            try:
                frame = fallback_fetcher(
                    source_series_id=fallback_series_id,
                    country=country,
                    frequency=str(indicator.get("frequency", "monthly")),
                    source_hint=fallback_source_hint,
                )
                return _clean_normalized_frame(frame, str(indicator["key"])), fallback_source
            except Exception as fallback_exc:
                logger.warning(
                    "Fallback China source failed for %s/%s: %s",
                    country,
                    indicator.get("key"),
                    fallback_exc,
                )
                return _empty_normalized_frame(), fallback_source
        logger.warning("China source failed for %s/%s: %s", country, indicator.get("key"), exc)
        return _empty_normalized_frame(), source

    fallback_source = str(indicator.get("fallback_source") or "")
    if fallback_source not in CHINA_SOURCE_FETCHERS:
        return primary_frame, source

    fallback_series_id = str(indicator.get("fallback_source_series_id") or source_series_id)
    fallback_source_hint = str(indicator.get("fallback_source_hint") or source_hint)
    fallback_fetcher, _subdir = CHINA_SOURCE_FETCHERS[fallback_source]
    try:
        fallback_frame = fallback_fetcher(
            source_series_id=fallback_series_id,
            country=country,
            frequency=str(indicator.get("frequency", "monthly")),
            source_hint=fallback_source_hint,
        )
        fallback_frame = _clean_normalized_frame(fallback_frame, str(indicator["key"]))
    except Exception:
        return primary_frame, source

    if primary_frame.empty and not fallback_frame.empty:
        return fallback_frame, fallback_source
    if fallback_frame.empty:
        return primary_frame, source
    if _latest_date(fallback_frame) > _latest_date(primary_frame):
        return fallback_frame, fallback_source
    return primary_frame, source


def fetch_china_api_bundle(base_dir: str = "data/raw/api/china") -> pd.DataFrame:
    """Fetch China API-backed indicators, save source files, and return a summary."""
    root = Path(base_dir)
    normalized_dir = root / "normalized"
    summary_rows: list[dict[str, object]] = []

    indicators = get_country_indicators("china", "macro") + get_country_indicators("china", "valuation")
    for indicator in indicators:
        key = str(indicator["key"])
        source = str(indicator.get("source", ""))
        if source not in CHINA_SOURCE_FETCHERS:
            continue

        frame, source_used = _fetch_indicator(indicator, country="china")
        source_subdir = CHINA_SOURCE_FETCHERS.get(source_used, (None, "normalized"))[1]
        if frame.empty:
            summary_rows.append(
                {
                    "series_id": canonicalize_china_series_id(key),
                    "source_used": source_used,
                    "row_count": 0,
                    "latest_date": pd.NaT,
                    "required_for_minimum_regime": bool(indicator.get("required_for_minimum_regime", False)),
                    "status": "missing",
                }
            )
            continue

        raw_frame = frame.copy()
        raw_frame["series_id"] = canonicalize_china_series_id(key)
        _save_frame(raw_frame, root / source_subdir / f"{key}.csv")
        normalized_frame = raw_frame.loc[:, NORMALIZED_COLUMNS].copy()
        normalized_frame["series_id"] = canonicalize_china_series_id(key)
        _save_frame(normalized_frame, normalized_dir / f"{key}.csv")
        summary_rows.append(
            {
                "series_id": canonicalize_china_series_id(key),
                "source_used": source_used,
                "row_count": len(normalized_frame),
                "latest_date": normalized_frame["date"].max(),
                "required_for_minimum_regime": bool(indicator.get("required_for_minimum_regime", False)),
                "status": "ready",
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["latest_date"] = pd.to_datetime(summary["latest_date"], errors="coerce")
        summary["ingested_at"] = _run_timestamp()
        _save_frame(summary, root / "normalized" / "_summary.csv")
    return summary


def rebuild_china_normalized_data(base_dir: str = "data/raw/api/china") -> pd.DataFrame:
    """Rebuild normalized China files from source-specific raw API folders."""
    root = Path(base_dir)
    normalized_dir = root / "normalized"
    summary_rows: list[dict[str, object]] = []

    indicators = get_country_indicators("china", "macro") + get_country_indicators("china", "valuation")
    for indicator in indicators:
        key = canonicalize_china_series_id(indicator["key"])
        source = str(indicator.get("source", ""))
        fallback_source = str(indicator.get("fallback_source") or "")
        source_candidates = [source]
        if fallback_source in CHINA_SOURCE_FETCHERS:
            source_candidates.append(fallback_source)

        rebuilt = False
        for source_name in source_candidates:
            fetcher_entry = CHINA_SOURCE_FETCHERS.get(source_name)
            if fetcher_entry is None:
                continue
            _fetcher, subdir = fetcher_entry
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
                    "required_for_minimum_regime": bool(indicator.get("required_for_minimum_regime", False)),
                    "status": "ready",
                }
            )
            rebuilt = True
            break

        if not rebuilt:
            summary_rows.append(
                {
                    "series_id": key,
                    "source_used": source,
                    "row_count": 0,
                    "latest_date": pd.NaT,
                    "required_for_minimum_regime": bool(indicator.get("required_for_minimum_regime", False)),
                    "status": "missing",
                }
            )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["latest_date"] = pd.to_datetime(summary["latest_date"], errors="coerce")
        summary["ingested_at"] = _run_timestamp()
        _save_frame(summary, normalized_dir / "_summary.csv")
    return summary


def validate_china_data(
    base_dir: str = "data/raw/api/china/normalized",
    stale_days_fresh: int = 90,
) -> dict[str, object]:
    """Validate normalized China inputs for features and regime readiness."""
    normalized_dir = Path(base_dir)
    available_frames: list[pd.DataFrame] = []
    series_status: list[dict[str, object]] = []
    normalized_files = sorted(
        file_path.name for file_path in normalized_dir.glob("*.csv") if file_path.name != "_summary.csv"
    )
    found_series_ids: list[str] = []

    for key in CHINA_CANONICAL_MACRO_SERIES:
        path = normalized_dir / f"{key}.csv"
        if not path.exists():
            series_status.append(
                {
                    "series_id": key,
                    "required": key in CHINA_MINIMUM_REGIME_SERIES,
                    "source_used": "No loaded data",
                    "row_count": 0,
                    "latest_date": pd.NaT,
                    "status": "missing",
                }
            )
            continue

        frame = pd.read_csv(path)
        frame = _clean_normalized_frame(frame, key)
        if not frame.empty:
            found_series_ids.extend(frame["series_id"].dropna().astype(str).tolist())
        frame["series_id"] = key
        available_frames.append(frame)
        latest_date = frame["date"].max() if not frame.empty else pd.NaT
        source_values = (
            sorted(frame["source"].dropna().astype(str).unique().tolist())
            if "source" in frame.columns and not frame.empty
            else []
        )
        series_status.append(
            {
                "series_id": key,
                "required": key in CHINA_MINIMUM_REGIME_SERIES,
                "source_used": ", ".join(source_values) if source_values else "No loaded data",
                "row_count": len(frame),
                "latest_date": latest_date,
                "status": "ready" if not frame.empty else "missing",
            }
        )

    status_frame = pd.DataFrame(series_status)
    available_series = sorted(status_frame.loc[status_frame["status"] == "ready", "series_id"].tolist())
    found_series_ids = sorted(set(canonicalize_china_series_id(item) for item in found_series_ids))
    missing_required = [
        series
        for series in CHINA_MINIMUM_REGIME_SERIES
        if series not in available_series
    ]
    optional_missing = [
        str(row["series_id"])
        for _, row in status_frame.iterrows()
        if str(row["series_id"]) in CHINA_ENRICHMENT_SERIES and row["status"] != "ready"
    ]
    enrichment_available = [series for series in CHINA_ENRICHMENT_SERIES if series in available_series]
    max_date = status_frame["latest_date"].max() if not status_frame.empty else pd.NaT
    stale_warning = None
    enrichment_used: list[str] = []
    enrichment_ignored_stale: list[str] = []
    if pd.notna(max_date):
        days_stale = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(max_date)).days
        if days_stale > stale_days_fresh:
            stale_warning = f"Latest China API data is {days_stale} days old."
        for _, row in status_frame.iterrows():
            series_id = str(row["series_id"])
            latest_date = row["latest_date"]
            if series_id not in CHINA_ENRICHMENT_SERIES or pd.isna(latest_date):
                continue
            series_stale_days = int((pd.Timestamp(max_date) - pd.Timestamp(latest_date)).days)
            if series_stale_days > 180:
                enrichment_ignored_stale.append(series_id)
            elif row["status"] == "ready":
                enrichment_used.append(series_id)
    else:
        enrichment_used = enrichment_available.copy()

    if len(enrichment_used) >= 3:
        richness_level = "rich"
    elif len(enrichment_used) >= 1:
        richness_level = "enhanced_partially_stale" if enrichment_ignored_stale else "enhanced"
    else:
        richness_level = "minimum"
    valuation_inputs = inspect_china_valuation_inputs(
        api_dir=str(Path(base_dir).parent.parent),
        manual_dir="data/raw/manual",
    )

    return {
        "country": "china",
        "series_status": status_frame,
        "available_series": available_series,
        "normalized_files_found": normalized_files,
        "series_ids_found": found_series_ids,
        "missing_required_series": missing_required,
        "minimum_inputs_used": [series for series in CHINA_MINIMUM_REGIME_SERIES if series in available_series],
        "enrichment_available_series": enrichment_available,
        "enrichment_inputs_used": enrichment_used,
        "enrichment_inputs_ignored_stale": enrichment_ignored_stale,
        "optional_missing_series": optional_missing,
        "scoring_richness_level": richness_level,
        "feature_build_ready": len(available_series) > 0,
        "regime_ready": len(missing_required) == 0,
        "valuation_loaded_data_path": valuation_inputs["loaded_data_path"],
        "valuation_normalized_files_found": valuation_inputs["normalized_files_found"],
        "valuation_canonical_series_ids_found": valuation_inputs["canonical_series_ids_found"],
        "valuation_actual_sources_found": valuation_inputs["actual_sources_found"],
        "valuation_proxy_inputs_used": valuation_inputs["proxy_inputs_used"],
        "valuation_proxy_inputs_missing": valuation_inputs["proxy_inputs_missing"],
        "valuation_proxy_readiness": valuation_inputs["valuation_ready"],
        "valuation_proxy_series_found": valuation_inputs["canonical_series_ids_found"],
        "valuation_ready": valuation_inputs["valuation_ready"],
        "stale_warning": stale_warning,
        "api_first_mode_active": True,
    }
