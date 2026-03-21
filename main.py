"""CLI entry point for the global macro monitor project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from app.consensus.deviation import (
    build_consensus_deviation,
    build_consensus_snapshots,
)
from app.consensus.fetchers import fetch_and_ingest_consensus_sources
from app.consensus.sources import ingest_consensus_notes
from app.data.fetchers import (
    fetch_country_api_bundle,
    fetch_us_macro_bundle,
    save_api_series_to_csv,
    save_series_to_csv,
)
from app.data.china_ingestion import (
    CHINA_MINIMUM_REGIME_SERIES,
    fetch_china_api_bundle,
    rebuild_china_normalized_data,
    validate_china_data,
)
from app.data.eurozone_ingestion import (
    EUROZONE_MINIMUM_REGIME_SERIES,
    fetch_eurozone_api_bundle,
    rebuild_eurozone_normalized_data,
    validate_eurozone_data,
)
from app.data.manual_loader import MINIMUM_MANUAL_SERIES, assess_manual_country_readiness
from app.data.market_overlay_ingestion import fetch_market_overlay_bundle, save_market_overlay_series
from app.factors.features import build_country_macro_features, build_us_macro_features
from app.regime.allocation import map_asset_preferences, save_country_asset_preferences, save_us_asset_preferences
from app.regime.alerts import build_monitor_alerts
from app.regime.classifier import (
    classify_country_macro_regime,
    classify_us_macro_regime,
    save_country_macro_regimes,
    save_us_macro_regimes,
)
from app.regime.change_detection import build_global_change_log
from app.regime.change_detection import (
    ALLOCATION_HISTORY_PATH,
    HISTORY_DIR,
    SUMMARY_HISTORY_PATH,
    append_global_allocation_history,
    append_global_summary_history,
)
from app.regime.evaluation import build_regime_evaluation_outputs
from app.regime.global_allocation import build_global_allocation_map
from app.regime.global_monitor import build_global_regime_summary
from app.utils.config import get_supported_countries, load_country_configs, load_indicator_configs
from app.utils.logging import get_logger
from app.valuation.features import (
    build_country_valuation_features,
    build_us_valuation_features,
    inspect_eurozone_valuation_inputs,
    inspect_china_valuation_inputs,
    inspect_us_valuation_inputs,
)
from app.valuation.models import compute_valuation_score, label_valuation_regime
from app.valuation.china_models import compute_china_valuation_score, label_china_valuation_regime
from app.valuation.eurozone_models import compute_eurozone_valuation_score, label_eurozone_valuation_regime

logger = get_logger(__name__)


def _load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ."""
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _is_likely_network_error(message: str) -> bool:
    """Heuristically detect transient network/DNS failures from exception text."""
    normalized = message.lower()
    patterns = [
        "name resolution",
        "failed to resolve",
        "temporary failure in name resolution",
        "max retries exceeded",
        "connection refused",
        "connection aborted",
        "connection reset",
        "timed out",
    ]
    return any(pattern in normalized for pattern in patterns)


def _csv_has_rows(path: Path) -> bool:
    """Return True when a CSV exists and has at least one data row."""
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    return not frame.empty


def _us_fred_cache_available() -> bool:
    """Return whether the local raw US FRED cache is usable."""
    required = ["CPIAUCSL", "CPILFESL", "UNRATE", "FEDFUNDS", "GS10", "M2SL"]
    root = Path("data/raw/fred")
    return all(_csv_has_rows(root / f"{series_id}.csv") for series_id in required)


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(description="Global macro monitor CLI.")
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=[
            "status",
            "fetch-us",
            "build-us-features",
            "classify-us-regime",
            "build-us-valuation",
            "map-us-assets",
            "build-country-features",
            "classify-country-regime",
            "build-country-valuation",
            "map-country-assets",
            "validate-manual-data",
            "validate-country-data",
            "fetch-country-api-data",
            "fetch-market-overlay-data",
            "rebuild-country-normalized-data",
            "build-global-summary",
            "build-global-allocation",
            "build-alerts",
            "ingest-consensus-notes",
            "fetch-consensus-sources",
            "build-consensus-snapshots",
            "build-consensus-deviation",
            "evaluate-regimes",
            "evaluate-confidence",
            "refresh-monitor",
            "run-global-monitor",
        ],
        help="CLI command to run.",
    )
    parser.add_argument(
        "--country",
        default="us",
        choices=get_supported_countries(),
        help="Country to run for country-specific commands.",
    )
    parser.add_argument(
        "--region",
        default="us",
        choices=get_supported_countries(),
        help="Region to run for consensus commands.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Input file or folder path for manual ingestion commands.",
    )
    return parser


def print_status() -> None:
    """Print a simple startup summary for the configured project."""
    countries = load_country_configs()
    indicators = load_indicator_configs()
    logger.info(
        "Loaded %s countries and indicator configs for %s markets.",
        len(countries),
        len(indicators),
    )
    print("global-macro-monitor is ready.")


def _format_date(value: pd.Timestamp | float) -> str:
    """Format a date value for CLI output."""
    if pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).date().isoformat()


def _load_csv(path: str) -> pd.DataFrame:
    """Load a processed CSV and normalize its date column."""
    frame = pd.read_csv(path)
    for column in ["date", "summary_date", "run_timestamp", "current_snapshot_timestamp", "prior_snapshot_timestamp"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def run_fetch_us() -> None:
    """Fetch and save the V1 US raw FRED macro bundle."""
    _load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise SystemExit(
            "FRED_API_KEY is not set. Add it to your environment before running fetch-us."
        )

    bundle = fetch_us_macro_bundle(api_key=api_key)
    print("Fetched US macro bundle from FRED:")
    for series_id, frame in bundle.items():
        save_series_to_csv(frame, series_id=series_id)
        date_min = _format_date(frame["date"].min()) if not frame.empty else "n/a"
        date_max = _format_date(frame["date"].max()) if not frame.empty else "n/a"
        print(f"- {series_id}: {len(frame)} rows ({date_min} -> {date_max})")


def run_fetch_market_overlay_data() -> None:
    """Fetch shared high-frequency market-overlay inputs."""
    _load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise SystemExit(
            "FRED_API_KEY is not set. Add it to your environment before running fetch-market-overlay-data."
        )

    bundle = fetch_market_overlay_bundle(api_key=api_key)
    print("Fetched shared market overlay bundle:")
    for series_id, frame in bundle.items():
        save_market_overlay_series(frame, series_id=series_id)
        date_min = _format_date(frame["date"].min()) if not frame.empty else "n/a"
        date_max = _format_date(frame["date"].max()) if not frame.empty else "n/a"
        country = frame["country"].iloc[-1] if not frame.empty else "n/a"
        print(f"- {series_id} ({country}): {len(frame)} rows ({date_min} -> {date_max})")


def run_build_country_features(country: str) -> None:
    """Build the processed macro feature panel for one country."""
    features = build_country_macro_features(country=country)
    date_min = _format_date(features["date"].min()) if not features.empty else "n/a"
    date_max = _format_date(features["date"].max()) if not features.empty else "n/a"
    print(
        f"Built {country} macro features: "
        f"{len(features)} rows ({date_min} -> {date_max}) -> data/processed/{country}_macro_features.csv"
    )


def _load_or_build_country_features(country: str) -> pd.DataFrame:
    """Load or build the processed country feature file."""
    path = Path(f"data/processed/{country}_macro_features.csv")
    if path.exists():
        return _load_csv(str(path))
    return build_country_macro_features(country=country)


def run_classify_country_regime(country: str) -> None:
    """Classify macro regimes for one country."""
    features = _load_or_build_country_features(country)
    regimes = classify_country_macro_regime(features, country=country)
    save_country_macro_regimes(regimes, country=country)

    if regimes.empty:
        print(f"No regime rows available for {country}. Saved empty file -> data/processed/{country}_macro_regimes.csv")
        return

    latest = regimes.dropna(subset=["date"]).iloc[-1]
    print(f"Latest {country} macro regime snapshot:")
    print(f"- date: {_format_date(latest['date'])}")
    print(f"- country: {latest['country']}")
    print(f"- growth_score: {latest['growth_score']:.2f}")
    print(f"- inflation_score: {latest['inflation_score']:.2f}")
    print(f"- liquidity_score: {latest['liquidity_score']:.2f}")
    if "regime_raw" in latest.index:
        print(f"- regime_raw: {latest['regime_raw']}")
    print(f"- regime: {latest['regime']}")
    if "regime_confidence" in latest.index:
        print(f"- regime_confidence: {latest['regime_confidence']}")
    if "regime_note" in latest.index:
        print(f"- regime_note: {latest['regime_note']}")
    print(f"- liquidity_regime: {latest['liquidity_regime']}")
    print(f"Saved -> data/processed/{country}_macro_regimes.csv")


def _load_or_build_country_valuation(country: str) -> pd.DataFrame:
    """Load or build the processed country valuation file."""
    path = Path(f"data/processed/{country}_valuation_features.csv")
    if path.exists():
        return _load_csv(str(path))
    if not Path(f"data/processed/{country}_macro_features.csv").exists():
        build_country_macro_features(country=country)
    return build_country_valuation_features(country=country)


def run_build_country_valuation(country: str) -> None:
    """Build valuation features for one country."""
    if not Path(f"data/processed/{country}_macro_features.csv").exists():
        build_country_macro_features(country=country)
    valuation = build_country_valuation_features(country=country)
    if country == "china":
        diagnostics = inspect_china_valuation_inputs()
        valuation["valuation_score"] = compute_china_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_china_valuation_regime)
    elif country == "eurozone":
        diagnostics = inspect_eurozone_valuation_inputs()
        valuation["valuation_score"] = compute_eurozone_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_eurozone_valuation_regime)
    else:
        diagnostics = inspect_us_valuation_inputs()
        valuation["valuation_score"] = compute_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_valuation_regime)
    valuation.to_csv(f"data/processed/{country}_valuation_features.csv", index=False)

    if valuation.empty:
        print(f"No valuation rows available for {country}. Saved empty file -> data/processed/{country}_valuation_features.csv")
        return

    latest = valuation.dropna(subset=["date"]).iloc[-1]
    print(f"Built {country} valuation features:")
    print(f"- rows: {len(valuation)}")
    print(f"- latest_date: {_format_date(latest['date'])}")
    print(f"- latest_valuation_score: {latest['valuation_score']:.2f}")
    print(f"- latest_valuation_regime: {latest['valuation_regime']}")
    if "valuation_confidence" in latest.index:
        print(f"- latest_valuation_confidence: {latest['valuation_confidence']}")
    if "valuation_inputs_used" in latest.index:
        print(f"- valuation_inputs_used: {latest['valuation_inputs_used'] or 'none'}")
    if "valuation_inputs_missing" in latest.index:
        print(f"- valuation_inputs_missing: {latest['valuation_inputs_missing'] or 'none'}")
    if country in {"us", "china", "eurozone"}:
        print(f"- loaded_data_path: {diagnostics['loaded_data_path']}")
        print(
            f"- normalized_files_found: "
            f"{', '.join(diagnostics['normalized_files_found']) if diagnostics['normalized_files_found'] else 'none'}"
        )
        print(
            f"- canonical_series_ids_found: "
            f"{', '.join(diagnostics['canonical_series_ids_found']) if diagnostics['canonical_series_ids_found'] else 'none'}"
        )
        print(
            f"- actual_sources_found: "
            f"{', '.join(diagnostics['actual_sources_found']) if diagnostics['actual_sources_found'] else 'none'}"
        )
        print(
            f"- proxy_inputs_used: "
            f"{', '.join(diagnostics['proxy_inputs_used']) if diagnostics['proxy_inputs_used'] else 'none'}"
        )
        print(
            f"- proxy_inputs_missing: "
            f"{', '.join(diagnostics['proxy_inputs_missing']) if diagnostics['proxy_inputs_missing'] else 'none'}"
        )
        print(f"- valuation_can_be_computed: {'yes' if diagnostics['valuation_ready'] else 'no'}")
    print(f"Saved -> data/processed/{country}_valuation_features.csv")


def run_map_country_assets(country: str) -> None:
    """Map macro plus valuation into asset preferences for one country."""
    regime_path = Path(f"data/processed/{country}_macro_regimes.csv")
    if not regime_path.exists():
        run_classify_country_regime(country)
    regimes = _load_csv(str(regime_path))
    valuation = _load_or_build_country_valuation(country)
    if "valuation_score" not in valuation.columns:
        valuation["valuation_score"] = compute_valuation_score(valuation)
    if "valuation_regime" not in valuation.columns:
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_valuation_regime)

    allocation = map_asset_preferences(regimes, valuation)
    save_country_asset_preferences(allocation, country=country)

    if allocation.empty:
        print(f"No asset preference rows available for {country}. Saved empty file -> data/processed/{country}_asset_preferences.csv")
        return

    latest = allocation.dropna(subset=["date"]).iloc[-1]
    print(f"Latest {country} asset preference snapshot:")
    print(f"- date: {_format_date(latest['date'])}")
    if "country" in latest.index:
        print(f"- country: {latest['country']}")
    print(f"- regime: {latest['regime']}")
    print(f"- liquidity_regime: {latest['liquidity_regime']}")
    print(f"- valuation_score: {latest['valuation_score']:.2f}")
    print(f"- valuation_regime: {latest['valuation_regime']}")
    print(f"- equities: {latest['equities']} ({latest['equities_score']:.1f})")
    print(f"- duration: {latest['duration']} ({latest['duration_score']:.1f})")
    print(f"- gold: {latest['gold']} ({latest['gold_score']:.1f})")
    print(f"- dollar: {latest['dollar']} ({latest['dollar_score']:.1f})")
    print(f"Saved -> data/processed/{country}_asset_preferences.csv")


def run_validate_manual_data(country: str) -> None:
    """Validate manual CSV inputs for one country."""
    readiness = assess_manual_country_readiness(country=country)
    available = readiness["available_series"]
    missing = readiness["missing_series"]
    print(f"Manual data validation for {country}:")
    print(f"- available_series: {', '.join(available) if available else 'none'}")
    print(f"- missing_minimum_series: {', '.join(missing) if missing else 'none'}")
    print(f"- required_minimum_series: {', '.join(MINIMUM_MANUAL_SERIES)}")
    print(f"- ready_for_regime_classification: {'yes' if readiness['ready'] else 'no'}")


def run_ingest_consensus(region: str, path: str | None) -> None:
    """Normalize raw consensus notes into processed storage."""
    target = path or f"data/raw/consensus/{region}"
    notes = ingest_consensus_notes(region=region, path=target)
    region_notes = notes.loc[notes["region"] == region].copy() if not notes.empty else pd.DataFrame()
    print(f"Consensus note ingestion for {region}:")
    print(f"- input_path: {target}")
    print(f"- normalized_rows: {len(region_notes)}")
    print(
        f"- source_names: "
        f"{', '.join(sorted(region_notes['source_name'].dropna().astype(str).unique().tolist())) if not region_notes.empty else 'none'}"
    )
    latest_date = _format_date(region_notes["date"].max()) if not region_notes.empty else "n/a"
    print(f"- latest_note_date: {latest_date}")
    print("Saved -> data/processed/consensus_notes.csv")


def run_fetch_consensus_sources(region: str) -> None:
    """Fetch automatic consensus sources for one region and refresh downstream outputs."""
    notes, summary = fetch_and_ingest_consensus_sources(region=region)
    print(f"Fetched automatic consensus sources for {region}:")
    if summary.empty:
        print("- no automatic sources are configured")
    else:
        for _, row in summary.iterrows():
            latest_date = _format_date(row["latest_date"]) if pd.notna(row["latest_date"]) else "n/a"
            error_suffix = f", error={row['error']}" if str(row.get("error", "")).strip() else ""
            print(f"- {row['source_key']}: notes={int(row['note_count'])}, latest_date={latest_date}{error_suffix}")
    print(f"- ingested_region_notes: {len(notes)}")
    print(f"Saved raw files -> data/raw/consensus/{region}/auto/")
    print("Saved -> data/processed/consensus_notes.csv")
    build_consensus_snapshots()
    build_consensus_deviation()
    print("Refreshed -> data/processed/consensus_snapshots.csv")
    print("Refreshed -> data/processed/consensus_deviation.csv")


def run_build_consensus_snapshots() -> None:
    """Aggregate recent consensus notes into region-level snapshots."""
    snapshots = build_consensus_snapshots()
    print("Built consensus snapshots:")
    if snapshots.empty:
        print("- no consensus notes are available yet")
        print("Saved -> data/processed/consensus_snapshots.csv")
        print("Saved -> data/processed/consensus_diagnostics.csv")
        return
    for _, row in snapshots.iterrows():
        print(
            f"- {row['region']}: growth={row['growth_consensus']}, "
            f"inflation={row['inflation_consensus']}, policy_bias={row['policy_bias_consensus']}, "
            f"source_count={int(row['source_count'])}, latest_note_date={_format_date(row['latest_note_date'])}"
        )
    print("Saved -> data/processed/consensus_snapshots.csv")
    print("Saved -> data/processed/consensus_diagnostics.csv")


def run_build_consensus_deviation() -> None:
    """Compare the current model state against public-consensus snapshots."""
    deviation = build_consensus_deviation()
    print("Built consensus deviation:")
    if deviation.empty:
        print("- no deviation rows are available yet")
        print("Saved -> data/processed/consensus_deviation.csv")
        return
    for _, row in deviation.iterrows():
        print(
            f"- {row['region']}: total={float(row['consensus_deviation_score']):.2f}, "
            f"growth={float(row['growth_deviation_score']):.2f}, "
            f"inflation={float(row['inflation_deviation_score']):.2f}, "
            f"policy={float(row['policy_deviation_score']):.2f}"
        )
        print(f"  summary: {row['deviation_summary']}")
    print("Saved -> data/processed/consensus_deviation.csv")


def run_fetch_country_api_data(country: str) -> None:
    """Fetch configured API-backed macro series for one country and save them to raw/api."""
    if country == "china":
        summary = fetch_china_api_bundle()
        if summary.empty:
            print("No China API series were fetched successfully. Falling back to manual data if available.")
            return
        print("Fetched China API bundle:")
        for _, row in summary.iterrows():
            latest_date = _format_date(row["latest_date"])
            print(
                f"- {row['series_id']}: source={row['source_used']}, rows={int(row['row_count'])}, "
                f"latest_date={latest_date}, status={row['status']}"
            )
        missing_required = summary.loc[
            (summary["required_for_minimum_regime"] == True) & (summary["status"] != "ready"),
            "series_id",
        ].tolist()
        print(f"- missing_required_series: {', '.join(missing_required) if missing_required else 'none'}")
        return
    if country == "eurozone":
        summary = fetch_eurozone_api_bundle()
        ready_series = []
        if not summary.empty and "status" in summary.columns:
            ready_series = (
                summary.loc[summary["status"] == "ready", "series_id"]
                .dropna()
                .astype(str)
                .tolist()
            )
        if not ready_series:
            raise SystemExit(
                "Eurozone API fetch produced no normalized minimum-series output. "
                "Expected canonical ids: cpi, growth_proxy, policy_rate, yield_10y."
            )
        print("Fetched Eurozone API bundle:")
        for _, row in summary.iterrows():
            latest_date = _format_date(row["latest_date"])
            print(
                f"- {row['series_id']}: source={row['source_used']}, rows={int(row['row_count'])}, "
                f"latest_date={latest_date}, status={row['status']}"
            )
        missing_required = summary.loc[
            (summary["required_for_minimum_regime"] == True) & (summary["status"] != "ready"),
            "series_id",
        ].tolist()
        print(f"- missing_required_series: {', '.join(missing_required) if missing_required else 'none'}")
        return

    bundle = fetch_country_api_bundle(country)
    if not bundle:
        print(f"No API series were fetched successfully for {country}. Falling back to manual data if available.")
        return
    print(f"Fetched API macro bundle for {country}:")
    for key, frame in bundle.items():
        save_api_series_to_csv(frame, country=country, indicator_key=key)
        latest_date = _format_date(frame["date"].max()) if not frame.empty else "n/a"
        print(f"- {key}: {len(frame)} rows (latest_date={latest_date})")


def run_validate_country_data(country: str) -> None:
    """Validate country inputs for downstream pipeline readiness."""
    if country == "china":
        result = validate_china_data()
        print("China data validation:")
        print(
            f"- normalized_files_found: "
            f"{', '.join(result['normalized_files_found']) if result['normalized_files_found'] else 'none'}"
        )
        print(
            f"- series_ids_found: "
            f"{', '.join(result['series_ids_found']) if result['series_ids_found'] else 'none'}"
        )
        print(f"- available_series: {', '.join(result['available_series']) if result['available_series'] else 'none'}")
        print(
            f"- missing_required_series: "
            f"{', '.join(result['missing_required_series']) if result['missing_required_series'] else 'none'}"
        )
        print(
            f"- optional_missing_series: "
            f"{', '.join(result['optional_missing_series']) if result['optional_missing_series'] else 'none'}"
        )
        print(
            f"- enrichment_available_series: "
            f"{', '.join(result['enrichment_available_series']) if result['enrichment_available_series'] else 'none'}"
        )
        print(
            f"- valuation_proxy_series_found: "
            f"{', '.join(result['valuation_proxy_series_found']) if result['valuation_proxy_series_found'] else 'none'}"
        )
        print(f"- valuation_loaded_data_path: {result['valuation_loaded_data_path']}")
        print(
            f"- valuation_normalized_files_found: "
            f"{', '.join(result['valuation_normalized_files_found']) if result['valuation_normalized_files_found'] else 'none'}"
        )
        print(
            f"- valuation_canonical_series_ids_found: "
            f"{', '.join(result['valuation_canonical_series_ids_found']) if result['valuation_canonical_series_ids_found'] else 'none'}"
        )
        print(
            f"- valuation_actual_sources_found: "
            f"{', '.join(result['valuation_actual_sources_found']) if result['valuation_actual_sources_found'] else 'none'}"
        )
        print(
            f"- valuation_proxy_inputs_used: "
            f"{', '.join(result['valuation_proxy_inputs_used']) if result['valuation_proxy_inputs_used'] else 'none'}"
        )
        print(
            f"- valuation_proxy_inputs_missing: "
            f"{', '.join(result['valuation_proxy_inputs_missing']) if result['valuation_proxy_inputs_missing'] else 'none'}"
        )
        print(f"- valuation_proxy_readiness: {'yes' if result['valuation_proxy_readiness'] else 'no'}")
        print(
            f"- minimum_inputs_used: "
            f"{', '.join(result['minimum_inputs_used']) if result['minimum_inputs_used'] else 'none'}"
        )
        print(
            f"- enrichment_inputs_used: "
            f"{', '.join(result['enrichment_inputs_used']) if result['enrichment_inputs_used'] else 'none'}"
        )
        print(
            f"- enrichment_inputs_ignored_stale: "
            f"{', '.join(result['enrichment_inputs_ignored_stale']) if result['enrichment_inputs_ignored_stale'] else 'none'}"
        )
        print(f"- scoring_richness_level: {result['scoring_richness_level']}")
        print(f"- required_minimum_series: {', '.join(CHINA_MINIMUM_REGIME_SERIES)}")
        print(f"- api_first_mode_active: {'yes' if result['api_first_mode_active'] else 'no'}")
        print(f"- feature_build_ready: {'yes' if result['feature_build_ready'] else 'no'}")
        print(f"- regime_classification_ready: {'yes' if result['regime_ready'] else 'no'}")
        print(f"- valuation_mapping_ready: {'yes' if result['valuation_ready'] else 'no'}")
        if result["stale_warning"]:
            print(f"- stale_warning: {result['stale_warning']}")
        return
    if country == "eurozone":
        result = validate_eurozone_data()
        print("Eurozone data validation:")
        print(f"- loaded_data_path: {result['loaded_data_path']}")
        print(
            f"- normalized_files_found: "
            f"{', '.join(result['normalized_files_found']) if result['normalized_files_found'] else 'none'}"
        )
        print(
            f"- series_ids_found: "
            f"{', '.join(result['series_ids_found']) if result['series_ids_found'] else 'none'}"
        )
        print(
            f"- actual_sources_found: "
            f"{', '.join(result['actual_sources_found']) if result['actual_sources_found'] else 'none'}"
        )
        print(f"- available_series: {', '.join(result['available_series']) if result['available_series'] else 'none'}")
        print(
            f"- missing_required_series: "
            f"{', '.join(result['missing_required_series']) if result['missing_required_series'] else 'none'}"
        )
        print(
            f"- optional_missing_series: "
            f"{', '.join(result['optional_missing_series']) if result['optional_missing_series'] else 'none'}"
        )
        print(
            f"- enrichment_available_series: "
            f"{', '.join(result['enrichment_available_series']) if result['enrichment_available_series'] else 'none'}"
        )
        print(
            f"- minimum_inputs_used: "
            f"{', '.join(result['minimum_inputs_used']) if result['minimum_inputs_used'] else 'none'}"
        )
        print(
            f"- enrichment_inputs_used: "
            f"{', '.join(result['enrichment_inputs_used']) if result['enrichment_inputs_used'] else 'none'}"
        )
        print(
            f"- enrichment_inputs_ignored_stale: "
            f"{', '.join(result['enrichment_inputs_ignored_stale']) if result['enrichment_inputs_ignored_stale'] else 'none'}"
        )
        print(f"- scoring_richness_level: {result['scoring_richness_level']}")
        print(f"- required_minimum_series: {', '.join(EUROZONE_MINIMUM_REGIME_SERIES)}")
        print(f"- valuation_loaded_data_path: {result['valuation_loaded_data_path']}")
        print(
            f"- valuation_normalized_files_found: "
            f"{', '.join(result['valuation_normalized_files_found']) if result['valuation_normalized_files_found'] else 'none'}"
        )
        print(
            f"- valuation_canonical_series_ids_found: "
            f"{', '.join(result['valuation_canonical_series_ids_found']) if result['valuation_canonical_series_ids_found'] else 'none'}"
        )
        print(
            f"- valuation_actual_sources_found: "
            f"{', '.join(result['valuation_actual_sources_found']) if result['valuation_actual_sources_found'] else 'none'}"
        )
        print(
            f"- valuation_proxy_inputs_used: "
            f"{', '.join(result['valuation_proxy_inputs_used']) if result['valuation_proxy_inputs_used'] else 'none'}"
        )
        print(
            f"- valuation_proxy_inputs_missing: "
            f"{', '.join(result['valuation_proxy_inputs_missing']) if result['valuation_proxy_inputs_missing'] else 'none'}"
        )
        print(f"- valuation_proxy_readiness: {'yes' if result['valuation_proxy_readiness'] else 'no'}")
        print(f"- api_first_mode_active: {'yes' if result['api_first_mode_active'] else 'no'}")
        print(f"- feature_build_ready: {'yes' if result['feature_build_ready'] else 'no'}")
        print(f"- regime_classification_ready: {'yes' if result['regime_ready'] else 'no'}")
        print(f"- valuation_mapping_ready: {'yes' if result['valuation_ready'] else 'no'}")
        if result["stale_warning"]:
            print(f"- stale_warning: {result['stale_warning']}")
        return

    run_validate_manual_data(country)


def run_rebuild_country_normalized_data(country: str) -> None:
    """Rebuild normalized API files from source-specific raw files."""
    if country != "china":
        if country == "eurozone":
            summary = rebuild_eurozone_normalized_data()
            print("Rebuilt Eurozone normalized API data:")
            if summary.empty:
                print("- no normalized files were created")
                return
            for _, row in summary.iterrows():
                latest_date = _format_date(row["latest_date"])
                print(
                    f"- {row['series_id']}: source={row['source_used']}, rows={int(row['row_count'])}, "
                    f"latest_date={latest_date}, status={row['status']}"
                )
            return
        print(f"Rebuild is not required for {country}.")
        return
    summary = rebuild_china_normalized_data()
    print("Rebuilt China normalized API data:")
    if summary.empty:
        print("- no normalized files were created")
        return
    for _, row in summary.iterrows():
        latest_date = _format_date(row["latest_date"])
        print(
            f"- {row['series_id']}: source={row['source_used']}, rows={int(row['row_count'])}, "
            f"latest_date={latest_date}, status={row['status']}"
        )


def run_build_global_summary() -> None:
    """Build the aggregated global summary table."""
    run_timestamp = pd.Timestamp.utcnow().tz_localize(None)
    summary = build_global_regime_summary()
    append_global_summary_history(summary, run_timestamp=run_timestamp)
    if summary.empty:
        print("Built empty global summary -> data/processed/global_macro_summary.csv")
        return
    print("Latest global macro summary:")
    for mode in ["latest_available", "last_common_date"]:
        matched = summary.loc[summary["as_of_mode"] == mode]
        if matched.empty:
            continue
        latest = matched.iloc[-1]
        print(f"- mode: {mode}")
        print(f"  summary_date: {_format_date(latest['summary_date'])}")
        print(f"  global_growth_score: {latest['global_growth_score']:.2f}")
        print(f"  global_inflation_score: {latest['global_inflation_score']:.2f}")
        print(f"  global_liquidity_score: {latest['global_liquidity_score']:.2f}")
        print(f"  global_valuation_score: {latest['global_valuation_score']:.2f}")
        print(f"  global_regime: {latest['global_regime']}")
        print(f"  investment_clock: {latest['investment_clock']}")
    print("Saved -> data/processed/global_macro_summary.csv")
    print(f"Saved -> {SUMMARY_HISTORY_PATH}")


def run_build_global_allocation() -> None:
    """Build the global cross-asset allocation map."""
    run_timestamp = pd.Timestamp.utcnow().tz_localize(None)
    summary = build_global_regime_summary()
    allocation = build_global_allocation_map(summary=summary)
    append_global_summary_history(summary, run_timestamp=run_timestamp)
    append_global_allocation_history(allocation, run_timestamp=run_timestamp)
    changes = build_global_change_log()
    evaluation_outputs = build_regime_evaluation_outputs(history_dir=str(HISTORY_DIR))
    if allocation.empty:
        print("Built empty global allocation map -> data/processed/global_allocation_map.csv")
        return

    print("Latest global allocation map:")
    for mode in ["latest_available", "last_common_date"]:
        matched = allocation.loc[allocation["as_of_mode"] == mode]
        if matched.empty:
            continue
        print(f"- mode: {mode}")
        latest_mode = matched.loc[matched["date"] == matched["date"].max()]
        for _, row in latest_mode.iterrows():
            print(
                f"  {row['asset']}: {row['preference']} "
                f"(score={float(row['score']):.1f}, confidence={row['confidence']})"
            )
    print("Saved -> data/processed/global_allocation_map.csv")
    print(f"Saved -> {SUMMARY_HISTORY_PATH}")
    print(f"Saved -> {ALLOCATION_HISTORY_PATH}")
    print(f"Saved -> data/processed/global_change_log.csv ({len(changes)} rows)")
    print(
        "Saved evaluation files -> "
        + ", ".join(
            f"{name} ({len(frame)} rows)" for name, frame in evaluation_outputs.items()
        )
    )


def run_build_alerts() -> None:
    """Build the monitor alert table."""
    alerts = build_monitor_alerts()
    if alerts.empty:
        print("Built empty monitor alerts -> data/processed/monitor_alerts.csv")
        return
    print("Latest monitor alerts:")
    for _, row in alerts.head(10).iterrows():
        date_text = _format_date(row["date"])
        print(
            f"- [{row['severity']}] {row['selected_mode']} {row['region']} {row['alert_type']}: "
            f"{row['entity_name']} ({date_text})"
        )
    print("Saved -> data/processed/monitor_alerts.csv")


def run_evaluate_regimes() -> None:
    """Build descriptive regime evaluation summaries."""
    outputs = build_regime_evaluation_outputs(history_dir=str(HISTORY_DIR))
    print("Built regime evaluation outputs:")
    print(f"- regime_frequency_summary.csv: {len(outputs['regime_frequency_summary.csv'])} rows")
    print(f"- regime_transition_matrix.csv: {len(outputs['regime_transition_matrix.csv'])} rows")
    print(f"- regime_forward_return_summary.csv: {len(outputs['regime_forward_return_summary.csv'])} rows")


def run_evaluate_confidence() -> None:
    """Build descriptive confidence-bucket evaluation summaries."""
    outputs = build_regime_evaluation_outputs(history_dir=str(HISTORY_DIR))
    print("Built confidence evaluation outputs:")
    print(f"- confidence_bucket_summary.csv: {len(outputs['confidence_bucket_summary.csv'])} rows")


def run_refresh_monitor() -> None:
    """Refresh the full monitoring stack with best-effort data fetching."""
    _load_dotenv()
    print("Refreshing global macro monitor:")

    fetch_errors: list[str] = []
    offline_mode = False
    for country in ["us", "china", "eurozone"]:
        if offline_mode:
            print(f"- fetch_{country}: skipped (network unavailable, using cached data)")
            continue
        try:
            if country == "us":
                run_fetch_us()
            else:
                run_fetch_country_api_data(country)
        except SystemExit as exc:
            message = str(exc)
            if country == "us" and _us_fred_cache_available():
                print(f"- fetch_{country}: degraded ({message}); using cached local FRED data")
            else:
                fetch_errors.append(f"{country}: {message}")
                print(f"- fetch_{country}: failed ({message})")
            if _is_likely_network_error(message):
                offline_mode = True
                print("- remote_fetch_mode: network appears unavailable; remaining remote fetches will be skipped")
        except Exception as exc:
            message = str(exc)
            if country == "us" and _us_fred_cache_available() and _is_likely_network_error(message):
                print(f"- fetch_{country}: degraded ({message}); using cached local FRED data")
            else:
                fetch_errors.append(f"{country}: {message}")
                print(f"- fetch_{country}: failed ({message})")
            if _is_likely_network_error(message):
                offline_mode = True
                print("- remote_fetch_mode: network appears unavailable; remaining remote fetches will be skipped")

    if not offline_mode:
        try:
            run_fetch_market_overlay_data()
        except SystemExit as exc:
            message = str(exc)
            fetch_errors.append(f"market_overlay: {message}")
            print(f"- market_overlay: failed ({message})")
        except Exception as exc:
            message = str(exc)
            fetch_errors.append(f"market_overlay: {message}")
            print(f"- market_overlay: failed ({message})")

    run_global_monitor()

    consensus_errors: list[str] = []
    for region in ["us", "eurozone", "china"]:
        if offline_mode:
            print(f"- consensus_{region}: skipped (network unavailable, reusing existing notes)")
            continue
        try:
            run_fetch_consensus_sources(region)
        except SystemExit as exc:
            message = str(exc)
            consensus_errors.append(f"{region}: {message}")
            print(f"- consensus_{region}: failed ({message})")
        except Exception as exc:
            consensus_errors.append(f"{region}: {exc}")
            print(f"- consensus_{region}: failed ({exc})")

    run_build_consensus_snapshots()
    run_build_consensus_deviation()
    run_evaluate_regimes()
    run_evaluate_confidence()
    run_build_alerts()

    if fetch_errors:
        print("- data_fetch_warnings:")
        for item in fetch_errors:
            print(f"  - {item}")
    if consensus_errors:
        print("- consensus_fetch_warnings:")
        for item in consensus_errors:
            print(f"  - {item}")

    print("Refresh complete.")


def run_global_monitor() -> None:
    """Run the country pipelines for all configured countries and then build the global summary."""
    for country in get_supported_countries():
        run_build_country_features(country)
        run_classify_country_regime(country)
        run_build_country_valuation(country)
        run_map_country_assets(country)
    run_build_global_summary()
    run_build_global_allocation()
    run_build_alerts()


def main() -> None:
    """Run the requested CLI command."""
    _load_dotenv()
    args = build_parser().parse_args()
    if args.command == "fetch-us":
        run_fetch_us()
        return
    if args.command == "build-country-features":
        run_build_country_features(args.country)
        return
    if args.command == "classify-country-regime":
        run_classify_country_regime(args.country)
        return
    if args.command == "build-country-valuation":
        run_build_country_valuation(args.country)
        return
    if args.command == "map-country-assets":
        run_map_country_assets(args.country)
        return
    if args.command == "validate-manual-data":
        run_validate_manual_data(args.country)
        return
    if args.command == "validate-country-data":
        run_validate_country_data(args.country)
        return
    if args.command == "fetch-country-api-data":
        run_fetch_country_api_data(args.country)
        return
    if args.command == "fetch-market-overlay-data":
        run_fetch_market_overlay_data()
        return
    if args.command == "rebuild-country-normalized-data":
        run_rebuild_country_normalized_data(args.country)
        return
    if args.command == "build-global-summary":
        run_build_global_summary()
        return
    if args.command == "build-global-allocation":
        run_build_global_allocation()
        return
    if args.command == "build-alerts":
        run_build_alerts()
        return
    if args.command == "ingest-consensus-notes":
        run_ingest_consensus(args.region, args.path)
        return
    if args.command == "fetch-consensus-sources":
        run_fetch_consensus_sources(args.region)
        return
    if args.command == "build-consensus-snapshots":
        run_build_consensus_snapshots()
        return
    if args.command == "build-consensus-deviation":
        run_build_consensus_deviation()
        return
    if args.command == "evaluate-regimes":
        run_evaluate_regimes()
        return
    if args.command == "evaluate-confidence":
        run_evaluate_confidence()
        return
    if args.command == "refresh-monitor":
        run_refresh_monitor()
        return
    if args.command == "run-global-monitor":
        run_global_monitor()
        return

    # Backward-compatible US commands.
    if args.command == "build-us-features":
        features = build_us_macro_features()
        print(
            "Built US macro features: "
            f"{len(features)} rows ({_format_date(features['date'].min()) if not features.empty else 'n/a'} -> {_format_date(features['date'].max()) if not features.empty else 'n/a'}) -> data/processed/us_macro_features.csv"
        )
        return
    if args.command == "classify-us-regime":
        features = _load_or_build_country_features("us")
        regimes = classify_us_macro_regime(features)
        save_us_macro_regimes(regimes)
        if not regimes.empty:
            latest = regimes.iloc[-1]
            print("Latest US macro regime snapshot:")
            print(f"- date: {_format_date(latest['date'])}")
            print(f"- growth_score: {latest['growth_score']:.2f}")
            print(f"- inflation_score: {latest['inflation_score']:.2f}")
            print(f"- liquidity_score: {latest['liquidity_score']:.2f}")
            print(f"- regime: {latest['regime']}")
            print(f"- liquidity_regime: {latest['liquidity_regime']}")
        print("Saved -> data/processed/us_macro_regimes.csv")
        return
    if args.command == "build-us-valuation":
        valuation = build_us_valuation_features()
        valuation["valuation_score"] = compute_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_valuation_regime)
        valuation.to_csv("data/processed/us_valuation_features.csv", index=False)
        if not valuation.empty:
            latest = valuation.iloc[-1]
            print("Built US valuation features:")
            print(f"- rows: {len(valuation)}")
            print(f"- latest_date: {_format_date(latest['date'])}")
            print(f"- latest_valuation_score: {latest['valuation_score']:.2f}")
            print(f"- latest_valuation_regime: {latest['valuation_regime']}")
        print("Saved -> data/processed/us_valuation_features.csv")
        return
    if args.command == "map-us-assets":
        regimes = _load_csv("data/processed/us_macro_regimes.csv")
        valuation = _load_or_build_country_valuation("us")
        valuation["valuation_score"] = compute_valuation_score(valuation)
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_valuation_regime)
        allocation = map_asset_preferences(regimes, valuation)
        save_us_asset_preferences(allocation)
        if not allocation.empty:
            latest = allocation.iloc[-1]
            print("Latest US asset preference snapshot:")
            print(f"- date: {_format_date(latest['date'])}")
            print(f"- regime: {latest['regime']}")
            print(f"- liquidity_regime: {latest['liquidity_regime']}")
            print(f"- valuation_score: {latest['valuation_score']:.2f}")
            print(f"- valuation_regime: {latest['valuation_regime']}")
            print(f"- equities: {latest['equities']} ({latest['equities_score']:.1f})")
            print(f"- duration: {latest['duration']} ({latest['duration_score']:.1f})")
            print(f"- gold: {latest['gold']} ({latest['gold_score']:.1f})")
            print(f"- dollar: {latest['dollar']} ({latest['dollar_score']:.1f})")
        print("Saved -> data/processed/us_asset_preferences.csv")
        return

    print_status()


if __name__ == "__main__":
    main()
