"""Scoring helpers for consensus aggregation and deviation."""

from __future__ import annotations

import math

import pandas as pd

SOURCE_TYPE_WEIGHTS = {
    "official": 1.25,
    "institution": 1.10,
    "media": 1.00,
    "other": 0.90,
}
CONFIDENCE_WEIGHTS = {
    "high": 1.20,
    "medium": 1.00,
    "low": 0.80,
}
GROWTH_SCALE = {
    "strong_negative": -2,
    "negative": -1,
    "neutral": 0,
    "positive": 1,
    "strong_positive": 2,
}
INFLATION_RISK_SCALE = {
    "strong_disinflation": 2,
    "disinflation": 1,
    "neutral": 0,
    "inflationary": -1,
    "strong_inflationary": -2,
}
POLICY_DOVISH_SCALE = {
    "strongly_hawkish": -2,
    "hawkish": -1,
    "neutral": 0,
    "dovish": 1,
    "strongly_dovish": 2,
}


def recency_weight(age_days: int, full_weight_days: int = 14, reduced_weight_days: int = 30) -> float:
    """Return a simple recency weight."""
    if age_days <= full_weight_days:
        return 1.0
    if age_days <= reduced_weight_days:
        return 0.5
    return 0.0


def confidence_weight(confidence: str) -> float:
    """Return a weight multiplier for parsing confidence."""
    return CONFIDENCE_WEIGHTS.get(str(confidence), 0.8)


def source_type_weight(source_type: str) -> float:
    """Return a weight multiplier for source type."""
    return SOURCE_TYPE_WEIGHTS.get(str(source_type), 0.9)


def weighted_note_score(age_days: int, source_type: str, confidence: str) -> float:
    """Combine recency, source type, and parsing confidence into one weight."""
    return recency_weight(age_days) * source_type_weight(source_type) * confidence_weight(confidence)


def _label_from_score(score: float, mapping_name: str) -> str:
    """Convert a numeric score back into a discrete label."""
    if pd.isna(score):
        return "neutral"
    if mapping_name == "growth":
        if score >= 1.5:
            return "strong_positive"
        if score >= 0.5:
            return "positive"
        if score <= -1.5:
            return "strong_negative"
        if score <= -0.5:
            return "negative"
        return "neutral"
    if mapping_name == "inflation":
        if score >= 1.5:
            return "strong_disinflation"
        if score >= 0.5:
            return "disinflation"
        if score <= -1.5:
            return "strong_inflationary"
        if score <= -0.5:
            return "inflationary"
        return "neutral"
    if score >= 1.5:
        return "strongly_dovish"
    if score >= 0.5:
        return "dovish"
    if score <= -1.5:
        return "strongly_hawkish"
    if score <= -0.5:
        return "hawkish"
    return "neutral"


def label_growth_from_score(score: float) -> str:
    """Map a weighted average growth score to a label."""
    return _label_from_score(score, "growth")


def label_inflation_from_score(score: float) -> str:
    """Map a weighted average inflation-risk score to a label."""
    return _label_from_score(score, "inflation")


def label_policy_from_score(score: float) -> str:
    """Map a weighted average policy score to a label."""
    return _label_from_score(score, "policy")


def aggregate_confidence(source_count: int, source_recency_score: float) -> str:
    """Summarize the reliability of a region-level consensus snapshot."""
    if source_count >= 4 and source_recency_score >= 0.8:
        return "high"
    if source_count >= 2 and source_recency_score >= 0.4:
        return "medium"
    return "low"


def deviation_score(model_value: int | float, consensus_value: int | float) -> float:
    """Compute a simple centered deviation score on [-1, 1]."""
    if pd.isna(model_value) or pd.isna(consensus_value):
        return float("nan")
    return max(-1.0, min(1.0, (float(model_value) - float(consensus_value)) / 2.0))


def safe_mean(values: list[float]) -> float:
    """Return the mean of finite values."""
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))

