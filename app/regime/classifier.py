"""Macro regime classification rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.factors.scoring import (
    compute_growth_score,
    compute_inflation_score,
    compute_liquidity_score,
)

REGIME_COLUMNS = [
    "date",
    "country",
    "growth_score",
    "inflation_score",
    "liquidity_score",
    "regime_raw",
    "regime",
    "regime_confidence",
    "regime_note",
    "liquidity_regime",
]

REGIME_NEUTRAL_BAND = 0.2
REGIME_HIGH_CONFIDENCE_THRESHOLD = 0.75


def classify_macro_regime(scores: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible simple classifier for generic score tables."""
    if scores.empty:
        return scores.assign(regime=pd.Series(dtype="object"))

    required = {"growth", "inflation"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing required score columns: {sorted(missing)}")

    classified = scores.copy()

    def decide_regime(row: pd.Series) -> str:
        if row["growth"] >= 0 and row["inflation"] >= 0:
            return "Overheat"
        if row["growth"] >= 0 and row["inflation"] < 0:
            return "Goldilocks"
        if row["growth"] < 0 and row["inflation"] >= 0:
            return "Stagflation"
        return "Slowdown"

    classified["regime"] = classified.apply(decide_regime, axis=1)
    return classified


def label_regime(growth_score: float, inflation_score: float) -> str:
    """Map growth and inflation scores to a macro regime."""
    if pd.isna(growth_score) or pd.isna(inflation_score):
        return "unknown"
    if growth_score >= 0 and inflation_score >= 0:
        return "reflation"
    if growth_score >= 0 and inflation_score < 0:
        return "goldilocks"
    if growth_score < 0 and inflation_score >= 0:
        return "stagflation"
    return "slowdown"


def _regime_confidence(growth_score: float, inflation_score: float) -> str:
    """Bucket regime confidence using distance from the growth/inflation axes."""
    if pd.isna(growth_score) or pd.isna(inflation_score):
        return "unknown"
    signal_strength = min(abs(float(growth_score)), abs(float(inflation_score)))
    if signal_strength >= REGIME_HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if signal_strength >= REGIME_NEUTRAL_BAND:
        return "medium"
    return "low"


def _smooth_regime_path(raw_regimes: pd.Series, growth: pd.Series, inflation: pd.Series) -> pd.Series:
    """Stabilize regime labels when scores sit near the zero boundary.

    When a raw regime flip happens but at least one axis remains inside the
    neutral band, the prior regime is carried forward. This keeps month-to-month
    monitoring readable without obscuring genuinely strong state changes.
    """
    smoothed: list[str] = []
    previous_regime = "unknown"

    for raw_regime, growth_score, inflation_score in zip(raw_regimes, growth, inflation, strict=False):
        current_regime = str(raw_regime)
        if current_regime == "unknown":
            smoothed.append(previous_regime if previous_regime != "unknown" else "unknown")
            continue

        near_boundary = (
            pd.isna(growth_score)
            or pd.isna(inflation_score)
            or abs(float(growth_score)) < REGIME_NEUTRAL_BAND
            or abs(float(inflation_score)) < REGIME_NEUTRAL_BAND
        )

        if previous_regime != "unknown" and current_regime != previous_regime and near_boundary:
            smoothed.append(previous_regime)
            continue

        smoothed.append(current_regime)
        previous_regime = current_regime

    return pd.Series(smoothed, index=raw_regimes.index, dtype="object")


def _regime_note(
    growth_score: float,
    inflation_score: float,
    raw_regime: str,
    smoothed_regime: str,
) -> str:
    """Explain how the final regime label was assigned."""
    if pd.isna(growth_score) or pd.isna(inflation_score):
        return "Growth or inflation score is unavailable."
    if raw_regime != smoothed_regime:
        return "Regime change was held back because one score remained near neutral."
    if abs(float(growth_score)) < REGIME_NEUTRAL_BAND or abs(float(inflation_score)) < REGIME_NEUTRAL_BAND:
        return "Regime sits near the neutral boundary and should be read with caution."
    return "Regime is based on growth and inflation scores with sufficient distance from neutral."


def label_liquidity_regime(liquidity_score: float) -> str:
    """Map liquidity score to a simple overlay label."""
    if pd.isna(liquidity_score):
        return "unknown"
    if liquidity_score >= 0.5:
        return "easy"
    if liquidity_score <= -0.5:
        return "tight"
    return "neutral"


def classify_country_macro_regime(features: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Compute country macro scores and classify each monthly regime."""
    if features.empty:
        return pd.DataFrame(columns=REGIME_COLUMNS)

    inferred_country = country or str(features["country"].dropna().iloc[0])
    classified = pd.DataFrame(
        {
            "date": pd.to_datetime(features["date"], errors="coerce"),
            "country": inferred_country,
        }
    )
    classified["growth_score"] = compute_growth_score(features, country=inferred_country)
    classified["inflation_score"] = compute_inflation_score(features, country=inferred_country)
    classified["liquidity_score"] = compute_liquidity_score(features, country=inferred_country)
    classified["regime_raw"] = classified.apply(
        lambda row: label_regime(row["growth_score"], row["inflation_score"]),
        axis=1,
    )
    classified["regime"] = _smooth_regime_path(
        raw_regimes=classified["regime_raw"],
        growth=classified["growth_score"],
        inflation=classified["inflation_score"],
    )
    classified["regime_confidence"] = classified.apply(
        lambda row: _regime_confidence(row["growth_score"], row["inflation_score"]),
        axis=1,
    )
    classified["regime_note"] = classified.apply(
        lambda row: _regime_note(
            row["growth_score"],
            row["inflation_score"],
            str(row["regime_raw"]),
            str(row["regime"]),
        ),
        axis=1,
    )
    classified["liquidity_regime"] = classified["liquidity_score"].apply(label_liquidity_regime)
    return classified.loc[:, REGIME_COLUMNS]


def save_country_macro_regimes(
    frame: pd.DataFrame,
    country: str,
    output_path: str | None = None,
) -> Path:
    """Save the classified country macro regime panel to CSV."""
    destination = Path(output_path or f"data/processed/{country}_macro_regimes.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def classify_us_macro_regime(features: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the US regime classifier."""
    return classify_country_macro_regime(features, country="us")


def save_us_macro_regimes(
    frame: pd.DataFrame,
    output_path: str = "data/processed/us_macro_regimes.csv",
) -> Path:
    """Backward-compatible wrapper for saving the US regime file."""
    return save_country_macro_regimes(frame, country="us", output_path=output_path)
