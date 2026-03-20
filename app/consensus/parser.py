"""Rules-based parsing for consensus notes."""

from __future__ import annotations

import re

import pandas as pd

GROWTH_VIEWS = ["strong_positive", "positive", "neutral", "negative", "strong_negative"]
INFLATION_VIEWS = ["strong_disinflation", "disinflation", "neutral", "inflationary", "strong_inflationary"]
POLICY_VIEWS = ["strongly_dovish", "dovish", "neutral", "hawkish", "strongly_hawkish"]
CONFIDENCE_LEVELS = ["low", "medium", "high"]

GROWTH_RULES = {
    "strong_positive": [
        "reacceleration",
        "re-acceleration",
        "robust growth",
        "strong growth",
        "broadening expansion",
        "solid pace",
        "continued to expand",
    ],
    "positive": [
        "resilient growth",
        "soft landing",
        "rebound",
        "stabilising growth",
        "improvement",
        "economy remains resilient",
        "recovery is strengthening",
        "stabilized and improved",
    ],
    "negative": [
        "slowdown",
        "slower growth",
        "deceleration",
        "weak demand",
        "loss of momentum",
        "domestic demand remains weak",
        "subdued demand",
        "growth has slowed",
        "downside risks to growth",
    ],
    "strong_negative": ["contraction", "recession", "hard landing", "slump", "sharp slowdown", "marked slowdown"],
}
INFLATION_RULES = {
    "strong_disinflation": ["deflation", "sharp disinflation", "inflation undershoot"],
    "disinflation": [
        "disinflation",
        "cooling inflation",
        "inflation easing",
        "moderating inflation",
        "price pressures easing",
        "inflation has eased",
        "inflation has declined",
        "price pressures moderated",
        "price level remained low",
    ],
    "inflationary": [
        "sticky inflation",
        "renewed inflation pressure",
        "reacceleration in inflation",
        "inflation risk",
        "inflation remains elevated",
        "price pressures remain high",
        "upside risks to inflation",
    ],
    "strong_inflationary": ["surging inflation", "runaway inflation", "broad inflation shock", "severe inflation pressure"],
}
POLICY_RULES = {
    "strongly_dovish": ["aggressive easing", "rapid cuts", "forceful support", "strongly accommodative"],
    "dovish": [
        "easing bias",
        "cuts ahead",
        "supportive policy",
        "accommodative",
        "more room to ease",
        "maintain ample liquidity",
        "supportive monetary policy",
        "provide support",
        "policy support",
    ],
    "hawkish": [
        "higher for longer",
        "restrictive",
        "tightening bias",
        "inflation vigilance",
        "rates stay elevated",
        "maintain restrictive",
        "restrictive stance",
    ],
    "strongly_hawkish": ["further hikes", "aggressive tightening", "forceful tightening", "very restrictive"],
}


def _tokenize(text: str) -> str:
    """Normalize text for simple keyword matching."""
    normalized = re.sub(r"\s+", " ", text.lower())
    return normalized.strip()


def _score_dimension(text: str, rules: dict[str, list[str]], positive_labels: list[str], negative_labels: list[str]) -> tuple[int, list[str]]:
    """Score one dimension from keyword hits."""
    matches: list[str] = []
    seen_keywords: set[str] = set()
    score = 0
    for label in positive_labels:
        for keyword in rules.get(label, []):
            if keyword in text and keyword not in seen_keywords:
                seen_keywords.add(keyword)
                matches.append(keyword)
                score += 2 if "strong" in label else 1
    for label in negative_labels:
        for keyword in rules.get(label, []):
            if keyword in text and keyword not in seen_keywords:
                seen_keywords.add(keyword)
                matches.append(keyword)
                score -= 2 if "strong" in label else 1
    return score, matches


def _label_growth(score: int) -> str:
    """Map growth keyword score to a stance label."""
    if score >= 2:
        return "strong_positive"
    if score >= 1:
        return "positive"
    if score <= -2:
        return "strong_negative"
    if score <= -1:
        return "negative"
    return "neutral"


def _label_inflation(score: int) -> str:
    """Map inflation keyword score to a stance label."""
    if score >= 2:
        return "strong_inflationary"
    if score >= 1:
        return "inflationary"
    if score <= -2:
        return "strong_disinflation"
    if score <= -1:
        return "disinflation"
    return "neutral"


def _label_policy(score: int) -> str:
    """Map policy keyword score to a stance label."""
    if score >= 2:
        return "strongly_dovish"
    if score >= 1:
        return "dovish"
    if score <= -2:
        return "strongly_hawkish"
    if score <= -1:
        return "hawkish"
    return "neutral"


def _confidence(source_type: str, match_count: int) -> str:
    """Assign a coarse parsing confidence."""
    if source_type == "official" and match_count >= 2:
        return "high"
    if match_count >= 2:
        return "medium"
    return "low"


def parse_consensus_notes(notes: pd.DataFrame) -> pd.DataFrame:
    """Parse normalized notes into structured stance labels."""
    if notes.empty:
        return pd.DataFrame(
            columns=list(notes.columns) + [
                "growth_view",
                "inflation_view",
                "policy_bias_view",
                "confidence",
                "classification_reason",
            ]
        )

    parsed_rows: list[dict[str, object]] = []
    for _, row in notes.iterrows():
        text = _tokenize(f"{row.get('title', '')} {row.get('body', '')}")
        growth_score, growth_matches = _score_dimension(
            text,
            GROWTH_RULES,
            positive_labels=["strong_positive", "positive"],
            negative_labels=["negative", "strong_negative"],
        )
        inflation_score, inflation_matches = _score_dimension(
            text,
            INFLATION_RULES,
            positive_labels=["inflationary", "strong_inflationary"],
            negative_labels=["disinflation", "strong_disinflation"],
        )
        policy_score, policy_matches = _score_dimension(
            text,
            POLICY_RULES,
            positive_labels=["strongly_dovish", "dovish"],
            negative_labels=["hawkish", "strongly_hawkish"],
        )
        total_matches = len(growth_matches) + len(inflation_matches) + len(policy_matches)
        parsed_rows.append(
            {
                **row.to_dict(),
                "growth_view": _label_growth(growth_score),
                "inflation_view": _label_inflation(inflation_score),
                "policy_bias_view": _label_policy(policy_score),
                "confidence": _confidence(str(row.get("source_type", "other")), total_matches),
                "classification_reason": "; ".join(
                    part
                    for part in [
                        f"growth_terms={', '.join(growth_matches)}" if growth_matches else "",
                        f"inflation_terms={', '.join(inflation_matches)}" if inflation_matches else "",
                        f"policy_terms={', '.join(policy_matches)}" if policy_matches else "",
                    ]
                    if part
                ) or "No strong rubric keywords found.",
            }
        )

    return pd.DataFrame(parsed_rows)
