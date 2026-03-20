"""Global cross-asset allocation mapping built on top of macro summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.global_monitor import build_country_status, build_global_regime_summary
from app.valuation.models import label_valuation_regime

GLOBAL_ASSETS = [
    "global_equities",
    "us_equities",
    "china_equities",
    "eurozone_equities",
    "duration",
    "gold",
    "dollar",
    "commodities",
]

CONFIDENCE_LEVELS = ["low", "medium", "high"]
DISPLAY_LABELS = {
    "goldilocks": "Goldilocks",
    "reflation": "Reflation",
    "stagflation": "Stagflation",
    "slowdown": "Slowdown",
    "partial_view": "Partial view",
    "unknown": "Unknown",
    "easy": "easy",
    "neutral": "neutral",
    "tight": "tight",
    "cheap": "cheap",
    "fair": "fair",
    "expensive": "expensive",
    "fresh": "fresh",
    "stale": "stale",
    "very_stale": "very stale",
    "us": "United States",
    "china": "China",
    "eurozone": "Eurozone",
    "latest_available": "Latest available",
    "last_common_date": "Last common date",
}


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with normalized dates when it exists."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _row_for_mode(frame: pd.DataFrame, mode: str, summary_date: pd.Timestamp) -> pd.Series | None:
    """Return a row that matches the selected time-alignment mode."""
    if frame.empty:
        return None
    if mode == "latest_available":
        valid = frame.dropna(subset=["date"]).sort_values("date")
        return None if valid.empty else valid.iloc[-1]
    matched = frame.loc[frame["date"] == summary_date].dropna(subset=["date"])
    return None if matched.empty else matched.iloc[-1]


def _liquidity_bucket(score: float) -> str:
    """Convert a numeric liquidity score into a simple regime label."""
    if pd.isna(score):
        return "unknown"
    if score >= 0.5:
        return "easy"
    if score <= -0.5:
        return "tight"
    return "neutral"


def _tag_preference(score: float) -> str:
    """Map a numeric score to a simple preference tag."""
    if score >= 1.5:
        return "bullish"
    if score <= -1.0:
        return "cautious"
    return "neutral"


def _display_label(value: object) -> str:
    """Convert internal labels into concise display text."""
    text = str(value)
    return DISPLAY_LABELS.get(text, text.replace("_", " ").title())


def _join_phrases(parts: list[str]) -> str:
    """Join short phrases into a readable sentence fragment."""
    cleaned = [part for part in parts if part]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _downgrade_confidence(level: str, steps: int = 1) -> str:
    """Downgrade a confidence level without dropping below low."""
    index = max(CONFIDENCE_LEVELS.index(level) - steps, 0)
    return CONFIDENCE_LEVELS[index]


def _cap_confidence(level: str, cap: str) -> str:
    """Cap a confidence level at a maximum allowed level."""
    return CONFIDENCE_LEVELS[min(CONFIDENCE_LEVELS.index(level), CONFIDENCE_LEVELS.index(cap))]


def _equities_score(regime: str, liquidity_regime: str, valuation_regime: str) -> float:
    """Compute an explainable equity preference score."""
    score = {
        "goldilocks": 2.0,
        "reflation": 1.0,
        "slowdown": -1.0,
        "stagflation": -2.0,
        "partial_view": 0.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {"easy": 0.5, "neutral": 0.0, "tight": -0.5, "unknown": 0.0}.get(
        liquidity_regime,
        0.0,
    )
    score += {"cheap": 1.0, "fair": 0.0, "expensive": -1.0, "unknown": 0.0}.get(
        valuation_regime,
        0.0,
    )
    return score


def _duration_score(regime: str, investment_clock: str) -> float:
    """Compute a simple duration preference score."""
    score = {
        "slowdown": 2.0,
        "goldilocks": 0.5,
        "reflation": -1.0,
        "stagflation": -2.0,
        "partial_view": 0.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {
        "slowdown": 0.5,
        "disinflationary_growth": 0.5,
        "reflation": -0.5,
        "overheating": -1.0,
        "partial_view": 0.0,
        "unknown": 0.0,
    }.get(investment_clock, 0.0)
    return score


def _gold_score(regime: str, investment_clock: str) -> float:
    """Compute a simple gold preference score."""
    score = {
        "stagflation": 2.0,
        "slowdown": 1.0,
        "reflation": 0.5,
        "goldilocks": -1.0,
        "partial_view": 0.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {
        "overheating": 1.0,
        "reflation": 0.5,
        "disinflationary_growth": 0.0,
        "slowdown": 0.5,
        "partial_view": 0.0,
        "unknown": 0.0,
    }.get(investment_clock, 0.0)
    return score


def _dollar_score(regime: str, liquidity_regime: str) -> float:
    """Compute a simple dollar preference score."""
    score = {
        "slowdown": 1.0,
        "stagflation": 1.0,
        "reflation": 0.0,
        "goldilocks": -1.0,
        "partial_view": 0.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {"tight": 1.5, "neutral": 0.0, "easy": -1.0, "unknown": 0.0}.get(
        liquidity_regime,
        0.0,
    )
    return score


def _commodities_score(regime: str, investment_clock: str) -> float:
    """Compute a simple commodities preference score."""
    score = {
        "reflation": 2.0,
        "stagflation": 1.0,
        "goldilocks": 0.5,
        "slowdown": -1.0,
        "partial_view": 0.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {
        "overheating": 1.0,
        "reflation": 0.5,
        "disinflationary_growth": -0.5,
        "slowdown": -0.5,
        "partial_view": 0.0,
        "unknown": 0.0,
    }.get(investment_clock, 0.0)
    return score


def _mode_context(
    mode: str,
    used_status: pd.DataFrame,
    summary_date: pd.Timestamp,
    coverage_ratio: float,
) -> str:
    """Describe the selected time-alignment mode once per allocation table."""
    countries = used_status["country"].tolist()
    if not countries:
        return "No countries were usable in the selected mode."
    if mode == "latest_available":
        pieces = [
            (
                f"{_display_label(country)} "
                f"{pd.Timestamp(latest_date).date().isoformat()} "
                f"({_display_label(staleness)})"
            )
            for country, latest_date, staleness in zip(
                used_status["country"],
                used_status["latest_date"],
                used_status["staleness_status"],
            )
        ]
        return (
            "Latest available compares each region on its own latest valid date: "
            + ", ".join(pieces)
            + f". Coverage is {coverage_ratio:.0%}."
        )
    return (
        "Last common date compares all contributing regions on "
        f"{pd.Timestamp(summary_date).date().isoformat()} across "
        f"{', '.join(_display_label(country) for country in countries)}. "
        f"Coverage is {coverage_ratio:.0%}."
    )


def _confidence_reason(
    *,
    coverage_ratio: float | None = None,
    staleness_statuses: list[str] | None = None,
    valuation_missing: bool = False,
    globally_usable_latest: bool | None = None,
    staleness_status: str | None = None,
) -> str:
    """Summarize why confidence may be capped for one allocation row."""
    issues: list[str] = []
    if coverage_ratio is not None and coverage_ratio < 1.0:
        issues.append("partial country coverage")
    statuses = staleness_statuses or []
    if any(status == "very_stale" for status in statuses):
        issues.append("very stale country data")
    elif any(status == "stale" for status in statuses):
        issues.append("stale country data")
    if globally_usable_latest is False:
        issues.append("the market is excluded from this mode")
    if staleness_status == "very_stale":
        issues.append("very stale local data")
    elif staleness_status == "stale":
        issues.append("stale local data")
    if valuation_missing:
        issues.append("missing valuation input")
    if not issues:
        return ""
    return f"Confidence is reduced by {_join_phrases(issues)}."


def _global_asset_reason(
    asset: str,
    score: float,
    global_regime: str,
    investment_clock: str,
    liquidity_regime: str,
    valuation_regime: str,
    valuation_missing: bool,
    confidence_note: str,
) -> str:
    """Build a short asset-specific reason for global allocation rows."""
    regime_label = _display_label(global_regime)
    liquidity_label = _display_label(liquidity_regime)
    valuation_label = _display_label(valuation_regime)
    clock_label = _display_label(investment_clock)

    if asset == "global_equities":
        if score >= 1.5:
            base = f"{regime_label} and {liquidity_label} liquidity still support global equities."
        elif score <= -1.0:
            base = f"{regime_label} and {liquidity_label} liquidity argue against global equities."
        else:
            base = f"The macro backdrop is mixed for global equities."
        valuation_text = (
            " This is a macro-only view because valuation input is missing."
            if valuation_missing
            else f" Valuations look {valuation_label}."
        )
        return base + valuation_text + (f" {confidence_note}" if confidence_note else "")

    if asset == "duration":
        if score >= 1.5:
            base = f"Softer growth and a {clock_label.lower()} clock favor duration."
        elif score <= -1.0:
            base = f"A {clock_label.lower()} clock limits the case for duration."
        else:
            base = "Cycle signals are not strong enough for a clear duration call."
        return base + (f" {confidence_note}" if confidence_note else "")

    if asset == "gold":
        if score >= 1.5:
            base = f"Macro stress and the {clock_label.lower()} backdrop support gold."
        elif score <= -1.0:
            base = "The current macro backdrop reduces the need for gold protection."
        else:
            base = "Gold signals are mixed."
        return base + (f" {confidence_note}" if confidence_note else "")

    if asset == "dollar":
        if score >= 1.5:
            base = f"{liquidity_label.capitalize()} liquidity and slower growth support the dollar."
        elif score <= -1.0:
            base = f"{liquidity_label.capitalize()} liquidity reduces support for the dollar."
        else:
            base = "The macro backdrop is mixed for the dollar."
        return base + (f" {confidence_note}" if confidence_note else "")

    if score >= 1.5:
        base = f"{clock_label} conditions support commodities."
    elif score <= -1.0:
        base = "Weak growth lowers the case for commodities."
    else:
        base = "Commodity signals are mixed."
    return base + (f" {confidence_note}" if confidence_note else "")


def _us_duration_reason(
    *,
    regime: str,
    liquidity_regime: str,
    valuation_regime: str,
    valuation_missing: bool,
    confidence_note: str,
) -> str:
    """Build a duration reason using the US local allocation lens."""
    regime_label = _display_label(regime)
    liquidity_label = _display_label(liquidity_regime)
    valuation_label = _display_label(valuation_regime)

    if regime == "slowdown":
        base = (
            f"United States is in {regime_label} with {liquidity_label} liquidity, "
            "which supports dollar duration."
        )
    elif regime in {"reflation", "stagflation"}:
        base = (
            f"United States is in {regime_label} with {liquidity_label} liquidity, "
            "which limits the dollar-duration view."
        )
    else:
        base = (
            f"United States is in {regime_label} with {liquidity_label} liquidity, "
            "so dollar duration remains neutral."
        )

    valuation_text = (
        " This is a macro-only view because valuation input is missing."
        if valuation_missing
        else f" Valuations look {valuation_label}."
    )
    return base + valuation_text + (f" {confidence_note}" if confidence_note else "")


def _country_equity_reason(
    *,
    country: str,
    regime_row: pd.Series | None,
    regime: str,
    liquidity_regime: str,
    valuation_status: str,
    valuation_score: float,
    staleness_status: str,
    globally_usable_latest: bool,
) -> str:
    """Build a short reason for one country's equity preference row."""
    country_label = _display_label(country)
    if regime_row is None:
        return f"No current macro snapshot is available for {country_label}."

    if not globally_usable_latest:
        return (
            f"{country_label} is excluded from this global mode because its latest usable data is "
            f"{_display_label(staleness_status)}."
        )

    regime_label = _display_label(regime)
    liquidity_label = _display_label(liquidity_regime)
    if regime in {"goldilocks", "reflation"}:
        base = f"{country_label} is in {regime_label} with {liquidity_label} liquidity, which supports equities."
    elif regime in {"slowdown", "stagflation"}:
        base = f"{country_label} is in {regime_label} with {liquidity_label} liquidity, which limits the equity view."
    else:
        base = f"{country_label} macro signals are mixed for equities."

    valuation_note = (
        f" Valuations look {_display_label(label_valuation_regime(valuation_score))}."
        if valuation_status == "ready" and not pd.isna(valuation_score)
        else " Valuation is missing, so this is a macro-only view."
    )
    confidence_note = _confidence_reason(
        globally_usable_latest=globally_usable_latest,
        staleness_status=staleness_status,
        valuation_missing=valuation_status != "ready" or pd.isna(valuation_score),
    )
    return base + valuation_note + (f" {confidence_note}" if confidence_note else "")


def _global_confidence(
    coverage_ratio: float,
    staleness_statuses: list[str],
    valuation_missing: bool,
    partial_view: bool,
) -> str:
    """Apply conservative confidence rules for global assets."""
    level = (
        "high"
        if coverage_ratio >= 0.99 and all(status == "fresh" for status in staleness_statuses)
        else "medium"
    )
    if coverage_ratio < 1.0:
        level = _cap_confidence(level, "medium")
    if partial_view or coverage_ratio < 0.7:
        level = "low"
    if any(status == "stale" for status in staleness_statuses):
        level = _cap_confidence(level, "medium")
    if any(status == "very_stale" for status in staleness_statuses):
        level = "low"
    if valuation_missing:
        level = _downgrade_confidence(level, steps=1)
    return level


def _country_confidence(
    globally_usable_latest: bool,
    staleness_status: str,
    valuation_status: str,
) -> str:
    """Apply conservative confidence rules for country assets."""
    if not globally_usable_latest or staleness_status == "very_stale":
        return "low"
    level = "high"
    if staleness_status == "stale":
        level = _cap_confidence(level, "medium")
    if valuation_status != "ready":
        level = _downgrade_confidence(level, steps=1)
    return level


def _build_global_asset_rows(
    processed: Path,
    summary_row: pd.Series,
    status_table: pd.DataFrame,
) -> list[dict[str, object]]:
    """Build global cross-asset rows for one summary mode."""
    coverage_ratio = float(summary_row.get("coverage_ratio", 0.0) or 0.0)
    global_regime = str(summary_row.get("global_regime", "unknown"))
    investment_clock = str(summary_row.get("investment_clock", "unknown"))
    liquidity_regime = _liquidity_bucket(summary_row.get("global_liquidity_score", float("nan")))
    valuation_score = summary_row.get("global_valuation_score", float("nan"))
    valuation_regime = label_valuation_regime(valuation_score)
    mode = str(summary_row["as_of_mode"])
    summary_date = pd.Timestamp(summary_row["summary_date"])

    used_countries = [
        country for country in str(summary_row.get("countries_available", "")).split(",") if country
    ]
    used_status = status_table.loc[status_table["country"].isin(used_countries)].copy()
    staleness_statuses = used_status["staleness_status"].astype(str).tolist()
    valuation_missing = pd.isna(valuation_score)
    confidence = _global_confidence(
        coverage_ratio=coverage_ratio,
        staleness_statuses=staleness_statuses,
        valuation_missing=valuation_missing,
        partial_view=global_regime == "partial_view",
    )

    mode_context = _mode_context(mode, used_status, summary_date, coverage_ratio)
    confidence_note = _confidence_reason(
        coverage_ratio=coverage_ratio,
        staleness_statuses=staleness_statuses,
        valuation_missing=valuation_missing,
    )

    us_assets = _load_csv(processed / "us_asset_preferences.csv")
    us_regime = _load_csv(processed / "us_macro_regimes.csv")
    us_valuation = _load_csv(processed / "us_valuation_features.csv")
    us_asset_row = _row_for_mode(us_assets, mode, summary_date)
    us_regime_row = _row_for_mode(us_regime, mode, summary_date)
    us_valuation_row = _row_for_mode(us_valuation, mode, summary_date)

    us_duration_score = (
        float(us_asset_row.get("duration_score", float("nan")))
        if us_asset_row is not None
        else float("nan")
    )
    us_duration_confidence = (
        str(us_asset_row.get("allocation_confidence", confidence))
        if us_asset_row is not None
        else confidence
    )
    us_duration_regime = (
        str(us_regime_row.get("regime", "unknown")) if us_regime_row is not None else "unknown"
    )
    us_duration_liquidity = (
        str(us_regime_row.get("liquidity_regime", "unknown"))
        if us_regime_row is not None
        else "unknown"
    )
    us_duration_valuation_score = (
        us_valuation_row.get("valuation_score", float("nan"))
        if us_valuation_row is not None
        else float("nan")
    )
    us_duration_valuation_regime = label_valuation_regime(us_duration_valuation_score)
    us_duration_valuation_missing = pd.isna(us_duration_valuation_score)

    asset_specs = {
        "global_equities": (
            _equities_score(global_regime, liquidity_regime, valuation_regime),
            _global_asset_reason(
                "global_equities",
                _equities_score(global_regime, liquidity_regime, valuation_regime),
                global_regime,
                investment_clock,
                liquidity_regime,
                valuation_regime,
                valuation_missing,
                confidence_note,
            ),
        ),
        "duration": (
            us_duration_score if pd.notna(us_duration_score) else _duration_score(global_regime, investment_clock),
            _us_duration_reason(
                regime=us_duration_regime,
                liquidity_regime=us_duration_liquidity,
                valuation_regime=us_duration_valuation_regime,
                valuation_missing=us_duration_valuation_missing,
                confidence_note=(
                    _confidence_reason(valuation_missing=us_duration_valuation_missing)
                    if pd.notna(us_duration_score)
                    else confidence_note
                ),
            ),
        ),
        "gold": (
            _gold_score(global_regime, investment_clock),
            _global_asset_reason(
                "gold",
                _gold_score(global_regime, investment_clock),
                global_regime,
                investment_clock,
                liquidity_regime,
                valuation_regime,
                valuation_missing,
                confidence_note,
            ),
        ),
        "dollar": (
            _dollar_score(global_regime, liquidity_regime),
            _global_asset_reason(
                "dollar",
                _dollar_score(global_regime, liquidity_regime),
                global_regime,
                investment_clock,
                liquidity_regime,
                valuation_regime,
                valuation_missing,
                confidence_note,
            ),
        ),
        "commodities": (
            _commodities_score(global_regime, investment_clock),
            _global_asset_reason(
                "commodities",
                _commodities_score(global_regime, investment_clock),
                global_regime,
                investment_clock,
                liquidity_regime,
                valuation_regime,
                valuation_missing,
                confidence_note,
            ),
        ),
    }

    rows: list[dict[str, object]] = []
    for asset, (score, reason) in asset_specs.items():
        row_confidence = us_duration_confidence if asset == "duration" and pd.notna(us_duration_score) else confidence
        rows.append(
            {
                "date": summary_date,
                "summary_date": summary_date,
                "as_of_mode": mode,
                "asset": asset,
                "preference": _tag_preference(score),
                "score": score,
                "confidence": row_confidence,
                "reason": reason,
                "mode_context": mode_context,
            }
        )
    return rows


def _build_country_equity_rows(
    processed: Path,
    summary_row: pd.Series,
    status_table: pd.DataFrame,
) -> list[dict[str, object]]:
    """Build country-equity preference rows for the selected mode."""
    rows: list[dict[str, object]] = []
    mode = str(summary_row["as_of_mode"])
    summary_date = pd.Timestamp(summary_row["summary_date"])

    for country in ["us", "china", "eurozone"]:
        regime_frame = _load_csv(processed / f"{country}_macro_regimes.csv")
        valuation_frame = _load_csv(processed / f"{country}_valuation_features.csv")
        regime_row = _row_for_mode(regime_frame, mode, summary_date)
        valuation_row = _row_for_mode(valuation_frame, mode, summary_date)

        status_row = status_table.loc[status_table["country"] == country]
        staleness_status = (
            str(status_row.iloc[0]["staleness_status"]) if not status_row.empty else "missing"
        )
        valuation_status = (
            str(status_row.iloc[0]["valuation_status"]) if not status_row.empty else "missing"
        )
        globally_usable_latest = (
            bool(status_row.iloc[0]["globally_usable_latest"]) if not status_row.empty else False
        )
        latest_date = status_row.iloc[0]["latest_date"] if not status_row.empty else pd.NaT

        regime = str(regime_row.get("regime", "unknown")) if regime_row is not None else "unknown"
        liquidity_regime = (
            str(regime_row.get("liquidity_regime", "unknown"))
            if regime_row is not None
            else "unknown"
        )
        valuation_score = (
            valuation_row.get("valuation_score", float("nan")) if valuation_row is not None else float("nan")
        )
        valuation_regime = label_valuation_regime(valuation_score)
        score = _equities_score(regime, liquidity_regime, valuation_regime)
        if not globally_usable_latest:
            score = 0.0

        if mode == "latest_available":
            mode_context = (
                f"Latest available uses {_display_label(country)} data through "
                f"{pd.Timestamp(latest_date).date().isoformat()}."
                if pd.notna(latest_date)
                else f"Latest available has no usable {_display_label(country)} date."
            )
        else:
            mode_context = (
                "Last common date compares this market on the shared date "
                f"{summary_date.date().isoformat()}."
            )

        reason = _country_equity_reason(
            country=country,
            regime_row=regime_row,
            regime=regime,
            liquidity_regime=liquidity_regime,
            valuation_status=valuation_status,
            valuation_score=valuation_score,
            staleness_status=staleness_status,
            globally_usable_latest=globally_usable_latest,
        )

        rows.append(
            {
                "date": summary_date,
                "summary_date": summary_date,
                "as_of_mode": mode,
                "asset": f"{country}_equities",
                "preference": _tag_preference(score),
                "score": score,
                "confidence": _country_confidence(
                    globally_usable_latest=globally_usable_latest,
                    staleness_status=staleness_status,
                    valuation_status=valuation_status,
                ),
                "reason": reason,
                "mode_context": mode_context,
            }
        )
    return rows


def build_global_allocation_map(
    processed_dir: str = "data/processed",
    summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a simple global cross-asset preference table for both time modes."""
    processed = Path(processed_dir)
    global_summary = summary if summary is not None else build_global_regime_summary(processed_dir=processed_dir)
    rows: list[dict[str, object]] = []

    for mode in ["latest_available", "last_common_date"]:
        matched = global_summary.loc[global_summary["as_of_mode"] == mode]
        if matched.empty:
            continue
        latest = matched.iloc[-1]
        status_table = build_country_status(processed_dir=processed_dir, mode=mode)
        rows.extend(_build_global_asset_rows(processed, latest, status_table))
        rows.extend(_build_country_equity_rows(processed, latest, status_table))

    output = pd.DataFrame(rows)
    output = output.loc[
        :,
        [
            column
            for column in [
                "date",
                "summary_date",
                "as_of_mode",
                "asset",
                "preference",
                "score",
                "confidence",
                "reason",
                "mode_context",
            ]
            if column in output.columns
        ],
    ]
    output_path = processed / "global_allocation_map.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output
