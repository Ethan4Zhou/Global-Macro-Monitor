"""Global aggregation for country-level macro regimes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.classifier import label_regime

COUNTRY_WEIGHTS = {"us": 0.5, "china": 0.25, "eurozone": 0.25}
PARTIAL_COVERAGE_WARNING = "Global summary is based on incomplete country coverage."


def map_global_investment_clock(growth_score: float, inflation_score: float) -> str:
    """Map global growth and inflation scores to a simple investment clock quadrant."""
    if pd.isna(growth_score) or pd.isna(inflation_score):
        return "unknown"
    if growth_score >= 0 and inflation_score >= 0:
        return "overheating" if inflation_score > growth_score else "reflation"
    if growth_score >= 0 and inflation_score < 0:
        return "disinflationary_growth"
    return "slowdown"


def classify_staleness(days_stale: float) -> str:
    """Convert days stale into a simple freshness bucket."""
    if pd.isna(days_stale):
        return "missing"
    if days_stale <= 90:
        return "fresh"
    if days_stale <= 180:
        return "stale"
    return "very_stale"


def _load_processed_frame(path: Path) -> pd.DataFrame:
    """Load a processed CSV if present, else return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _format_weights(weights: dict[str, float]) -> str:
    """Format weights as a compact string for CSV export and dashboard display."""
    return ",".join(f"{country}:{weight:.2f}" for country, weight in weights.items())


def _renormalize_weights(countries: list[str]) -> dict[str, float]:
    """Renormalize configured weights across available countries."""
    total = sum(COUNTRY_WEIGHTS[country] for country in countries)
    if total == 0:
        return {country: 0.0 for country in COUNTRY_WEIGHTS}
    return {
        country: (COUNTRY_WEIGHTS[country] / total if country in countries else 0.0)
        for country in COUNTRY_WEIGHTS
    }


def _weighted_average(values: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Aggregate a metric across available countries using renormalized weights."""
    available = [country for country, value in values.items() if pd.notna(value)]
    effective_weights = _renormalize_weights(available)
    if not available:
        return float("nan"), effective_weights
    score = sum(float(values[country]) * effective_weights[country] for country in available)
    return score, effective_weights


def _latest_valid_row(frame: pd.DataFrame) -> pd.Series | None:
    """Return the latest row with valid regime inputs."""
    if frame.empty:
        return None
    required = ["date", "growth_score", "inflation_score", "liquidity_score", "regime"]
    available = [column for column in required if column in frame.columns]
    if len(available) < len(required):
        return None
    valid = frame.dropna(subset=required)
    if valid.empty:
        return None
    return valid.sort_values("date").iloc[-1]


def _latest_valuation_row(frame: pd.DataFrame) -> pd.Series | None:
    """Return the latest valuation row with a valid valuation score."""
    if frame.empty or "valuation_score" not in frame.columns:
        return None
    valid = frame.dropna(subset=["date", "valuation_score"])
    if valid.empty:
        return None
    return valid.sort_values("date").iloc[-1]


def _row_for_date(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    """Return the row at a specific date if present and valid."""
    if frame.empty:
        return None
    matched = frame.loc[frame["date"] == date]
    if matched.empty:
        return None
    row = matched.iloc[-1]
    required = ["growth_score", "inflation_score", "liquidity_score", "regime"]
    if any(pd.isna(row.get(column)) for column in required):
        return None
    return row


def _valuation_row_for_date(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    """Return the valuation row at a specific date if a score exists."""
    if frame.empty or "valuation_score" not in frame.columns:
        return None
    matched = frame.loc[frame["date"] == date].dropna(subset=["valuation_score"])
    if matched.empty:
        return None
    return matched.iloc[-1]


def _country_snapshot(processed: Path, country: str) -> dict[str, object]:
    """Load the processed regime and valuation files for one country."""
    regime = _load_processed_frame(processed / f"{country}_macro_regimes.csv")
    valuation = _load_processed_frame(processed / f"{country}_valuation_features.csv")
    latest_regime = _latest_valid_row(regime)
    latest_valuation = _latest_valuation_row(valuation)
    latest_date = latest_regime["date"] if latest_regime is not None else pd.NaT
    country_ready = latest_regime is not None
    valuation_ready = latest_valuation is not None
    return {
        "country": country,
        "regime_frame": regime,
        "valuation_frame": valuation,
        "latest_regime": latest_regime,
        "latest_valuation": latest_valuation,
        "latest_date": latest_date,
        "country_ready": country_ready,
        "valuation_ready": valuation_ready,
    }


def _build_mode_row(
    mode: str,
    snapshots: dict[str, dict[str, object]],
    max_observed_date: pd.Timestamp | pd.NaT,
) -> dict[str, object]:
    """Build one global summary row for a specific time-alignment mode."""
    ready_countries = [
        country for country, snapshot in snapshots.items() if bool(snapshot["country_ready"])
    ]

    if mode == "latest_available":
        summary_date = max_observed_date
        regime_rows = {
            country: snapshots[country]["latest_regime"] for country in ready_countries
        }
        valuation_rows = {
            country: snapshots[country]["latest_valuation"] for country in ready_countries
        }
    else:
        date_sets = []
        for country in ready_countries:
            frame = snapshots[country]["regime_frame"]
            date_sets.append(set(frame.dropna(subset=["growth_score", "inflation_score", "liquidity_score", "regime"])["date"].tolist()))
        common_dates = set.intersection(*date_sets) if date_sets else set()
        summary_date = max(common_dates) if common_dates else pd.NaT
        regime_rows = {
            country: _row_for_date(snapshots[country]["regime_frame"], summary_date)
            for country in ready_countries
        } if pd.notna(summary_date) else {country: None for country in ready_countries}
        valuation_rows = {
            country: _valuation_row_for_date(snapshots[country]["valuation_frame"], summary_date)
            for country in ready_countries
        } if pd.notna(summary_date) else {country: None for country in ready_countries}

    countries_available = [
        country for country, row in regime_rows.items() if row is not None
    ]
    countries_missing = [country for country in COUNTRY_WEIGHTS if country not in countries_available]
    coverage_ratio = len(countries_available) / len(COUNTRY_WEIGHTS)

    growth_score, effective_weights = _weighted_average(
        {
            country: (regime_rows.get(country)["growth_score"] if regime_rows.get(country) is not None else float("nan"))
            for country in COUNTRY_WEIGHTS
        }
    )
    inflation_score, _ = _weighted_average(
        {
            country: (regime_rows.get(country)["inflation_score"] if regime_rows.get(country) is not None else float("nan"))
            for country in COUNTRY_WEIGHTS
        }
    )
    liquidity_score, _ = _weighted_average(
        {
            country: (regime_rows.get(country)["liquidity_score"] if regime_rows.get(country) is not None else float("nan"))
            for country in COUNTRY_WEIGHTS
        }
    )
    valuation_score, _ = _weighted_average(
        {
            country: (valuation_rows.get(country)["valuation_score"] if valuation_rows.get(country) is not None else float("nan"))
            for country in COUNTRY_WEIGHTS
        }
    )

    if coverage_ratio < 0.7:
        global_regime = "partial_view"
        investment_clock = "partial_view"
        coverage_warning = PARTIAL_COVERAGE_WARNING
    else:
        global_regime = label_regime(growth_score, inflation_score)
        investment_clock = map_global_investment_clock(growth_score, inflation_score)
        coverage_warning = ""

    row: dict[str, object] = {
        "as_of_mode": mode,
        "summary_date": summary_date,
        "date": summary_date,
        "countries_available": ",".join(countries_available),
        "countries_missing": ",".join(countries_missing),
        "coverage_ratio": coverage_ratio,
        "coverage_warning": coverage_warning,
        "configured_weights": _format_weights(COUNTRY_WEIGHTS),
        "effective_weights": _format_weights(effective_weights),
        "global_growth_score": growth_score,
        "global_inflation_score": inflation_score,
        "global_liquidity_score": liquidity_score,
        "global_valuation_score": valuation_score,
        "global_regime": global_regime,
        "investment_clock": investment_clock,
        "global_investment_clock": investment_clock,
    }

    for country in COUNTRY_WEIGHTS:
        regime_row = regime_rows.get(country)
        row[f"{country}_regime"] = regime_row["regime"] if regime_row is not None else pd.NA
        row[f"{country}_latest_date"] = snapshots[country]["latest_date"]

    return row


def build_country_status(processed_dir: str = "data/processed", mode: str = "latest_available") -> pd.DataFrame:
    """Build a country status table for dashboard use."""
    processed = Path(processed_dir)
    snapshots = {country: _country_snapshot(processed, country) for country in COUNTRY_WEIGHTS}
    latest_dates = [snapshot["latest_date"] for snapshot in snapshots.values() if pd.notna(snapshot["latest_date"])]
    max_observed_date = max(latest_dates) if latest_dates else pd.NaT

    mode_row = _build_mode_row(mode=mode, snapshots=snapshots, max_observed_date=max_observed_date)
    summary_date = mode_row["summary_date"]
    globally_usable = set(
        item for item in str(mode_row["countries_available"]).split(",") if item
    )

    rows: list[dict[str, object]] = []
    for country, snapshot in snapshots.items():
        latest_date = snapshot["latest_date"]
        days_stale = (
            int((max_observed_date - latest_date).days)
            if pd.notna(max_observed_date) and pd.notna(latest_date)
            else float("nan")
        )
        latest_regime = snapshot["latest_regime"]
        rows.append(
            {
                "country": country,
                "regime": latest_regime["regime"] if latest_regime is not None else pd.NA,
                "country_ready": bool(snapshot["country_ready"]),
                "globally_usable_latest": (
                    bool(snapshot["country_ready"])
                    if mode == "latest_available"
                    else (
                        country in globally_usable
                        and pd.notna(summary_date)
                        and _row_for_date(snapshot["regime_frame"], summary_date) is not None
                    )
                ),
                "latest_date": latest_date,
                "days_stale": days_stale,
                "staleness_status": classify_staleness(days_stale),
                "valuation_status": "ready" if snapshot["valuation_ready"] else "missing",
                "summary_date": summary_date,
                "as_of_mode": mode,
            }
        )

    return pd.DataFrame(rows)


def build_global_regime_summary(processed_dir: str = "data/processed") -> pd.DataFrame:
    """Combine country outputs into one global summary table for both time modes."""
    processed = Path(processed_dir)
    snapshots = {country: _country_snapshot(processed, country) for country in COUNTRY_WEIGHTS}
    latest_dates = [snapshot["latest_date"] for snapshot in snapshots.values() if pd.notna(snapshot["latest_date"])]
    max_observed_date = max(latest_dates) if latest_dates else pd.NaT

    rows = [
        _build_mode_row("latest_available", snapshots, max_observed_date),
        _build_mode_row("last_common_date", snapshots, max_observed_date),
    ]
    output = pd.DataFrame(rows)
    output_path = processed / "global_macro_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output
