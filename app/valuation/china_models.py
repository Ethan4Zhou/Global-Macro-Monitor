"""China-specific valuation helpers."""

from __future__ import annotations

import pandas as pd

from app.valuation.models import (
    build_valuation_component_frame,
    build_weighted_block_score,
    label_valuation_regime,
)


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric column or an all-NaN fallback."""
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def compute_china_valuation_score(df: pd.DataFrame) -> pd.Series:
    """Compute a China valuation score where higher means cheaper.

    The structure mirrors a buy-side style rubric:
    - equity valuation anchor: HS300 PE / PB plus China CAPE proxy
    - discount-rate block: real yield
    - curve block: term spread
    - risk-compensation block: equity risk proxy
    """
    component_frame = build_valuation_component_frame(df)
    return build_weighted_block_score(
        component_frame,
        {
            "equity_valuation": (
                [
                    "equity_pe_proxy_component",
                    "shiller_pe_proxy_component",
                    "equity_pb_proxy_component",
                ],
                0.5,
            ),
            "discount_rate": (["real_yield_proxy_component"], 0.2),
            "curve_support": (["term_spread_component"], 0.1),
            "risk_compensation": (["equity_risk_proxy_component"], 0.2),
        },
    )


def compute_china_valuation_confidence(df: pd.DataFrame) -> pd.Series:
    """Score China valuation confidence from actual coverage of equity and rates blocks."""
    has_pe = _series(df, "equity_pe_proxy").notna()
    has_pb = _series(df, "equity_pb_proxy").notna()
    has_cape = _series(df, "shiller_pe_proxy").notna()
    has_real_yield = _series(df, "real_yield_proxy").notna()
    has_term_spread = _series(df, "term_spread").notna()
    has_equity_risk = _series(df, "equity_risk_proxy").notna()

    equity_count = has_pe.astype(int) + has_pb.astype(int) + has_cape.astype(int)
    rates_count = has_real_yield.astype(int) + has_term_spread.astype(int)
    risk_count = has_equity_risk.astype(int)

    confidence = pd.Series("low", index=df.index, dtype="object")
    confidence = confidence.mask((equity_count >= 1) & (rates_count >= 2), "medium")
    confidence = confidence.mask((equity_count >= 2) & (rates_count >= 2) & (risk_count >= 1), "high")
    return confidence


def label_china_valuation_regime(score: float) -> str:
    """Map China valuation score to a simple rich/fair/cheap bucket."""
    return label_valuation_regime(score)
