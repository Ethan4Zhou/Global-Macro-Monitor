"""Consensus snapshot construction and model deviation scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.consensus.parser import parse_consensus_notes
from app.consensus.scoring import (
    GROWTH_SCALE,
    INFLATION_RISK_SCALE,
    POLICY_DOVISH_SCALE,
    aggregate_confidence,
    deviation_score,
    label_growth_from_score,
    label_inflation_from_score,
    label_policy_from_score,
    safe_mean,
    weighted_note_score,
)

CONSENSUS_SNAPSHOTS_PATH = Path("data/processed/consensus_snapshots.csv")
CONSENSUS_DEVIATION_PATH = Path("data/processed/consensus_deviation.csv")
CONSENSUS_DIAGNOSTICS_PATH = Path("data/processed/consensus_diagnostics.csv")
CONSENSUS_NOTES_PATH = Path("data/processed/consensus_notes.csv")
REGION_ORDER = ["us", "eurozone", "china"]


def _load_notes(path: str | Path = CONSENSUS_NOTES_PATH) -> pd.DataFrame:
    """Load normalized consensus notes."""
    note_path = Path(path)
    if not note_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(note_path)
    for column in ["date", "ingestion_timestamp"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _load_region_regime(region: str) -> pd.DataFrame:
    """Load the latest region macro regime frame."""
    path = Path(f"data/processed/{region}_macro_regimes.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def _growth_view_from_model(score: float) -> str:
    """Map model growth score into the consensus comparison space."""
    if pd.isna(score):
        return "neutral"
    if score >= 1.0:
        return "strong_positive"
    if score >= 0.25:
        return "positive"
    if score <= -1.0:
        return "strong_negative"
    if score <= -0.25:
        return "negative"
    return "neutral"


def _inflation_view_from_model(score: float) -> str:
    """Map model inflation score into the consensus comparison space."""
    if pd.isna(score):
        return "neutral"
    if score <= -1.0:
        return "strong_disinflation"
    if score <= -0.25:
        return "disinflation"
    if score >= 1.0:
        return "strong_inflationary"
    if score >= 0.25:
        return "inflationary"
    return "neutral"


def _policy_view_from_model(score: float, liquidity_regime: str) -> str:
    """Map model liquidity backdrop into a policy-bias view."""
    if liquidity_regime == "easy" or score >= 1.0:
        return "strongly_dovish"
    if score >= 0.25:
        return "dovish"
    if liquidity_regime == "tight" or score <= -1.0:
        return "strongly_hawkish"
    if score <= -0.25:
        return "hawkish"
    return "neutral"


def map_model_views(region: str) -> dict[str, object]:
    """Map the latest model state for one region into comparison dimensions."""
    regime = _load_region_regime(region)
    if regime.empty:
        return {
            "region": region,
            "model_date": pd.NaT,
            "model_regime": "unknown",
            "model_growth_view": "neutral",
            "model_inflation_view": "neutral",
            "model_policy_bias_view": "neutral",
        }
    latest = regime.dropna(subset=["date"]).iloc[-1]
    return {
        "region": region,
        "model_date": latest["date"],
        "model_regime": latest.get("regime", "unknown"),
        "model_growth_view": _growth_view_from_model(float(latest.get("growth_score", float("nan")))),
        "model_inflation_view": _inflation_view_from_model(float(latest.get("inflation_score", float("nan")))),
        "model_policy_bias_view": _policy_view_from_model(
            float(latest.get("liquidity_score", float("nan"))),
            str(latest.get("liquidity_regime", "neutral")),
        ),
    }


def _snapshot_reason(dimension: str, label: str, source_count: int) -> str:
    """Summarize why the snapshot landed on a final stance."""
    return f"{dimension} consensus leans {label} based on {source_count} recent notes."


def build_consensus_snapshots(
    notes_path: str | Path = CONSENSUS_NOTES_PATH,
    output_path: str | Path = CONSENSUS_SNAPSHOTS_PATH,
    diagnostics_path: str | Path = CONSENSUS_DIAGNOSTICS_PATH,
) -> pd.DataFrame:
    """Aggregate parsed consensus notes into region-level snapshots."""
    notes = _load_notes(notes_path)
    parsed = parse_consensus_notes(notes)
    diagnostics_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []

    if parsed.empty:
        empty_snapshots = pd.DataFrame(
            columns=[
                "region",
                "snapshot_date",
                "growth_consensus",
                "inflation_consensus",
                "policy_bias_consensus",
                "consensus_confidence",
                "source_count",
                "source_recency_score",
                "latest_note_date",
            ]
        )
        empty_snapshots.to_csv(output_path, index=False)
        pd.DataFrame(columns=[
            "region",
            "note_id",
            "source_name",
            "source_type",
            "date",
            "age_days",
            "recency_weight",
            "note_weight",
            "used_in_snapshot",
            "ignore_reason",
            "growth_view",
            "inflation_view",
            "policy_bias_view",
            "confidence",
            "classification_reason",
        ]).to_csv(diagnostics_path, index=False)
        return empty_snapshots

    for region, region_notes in parsed.groupby("region"):
        snapshot_date = pd.to_datetime(region_notes["date"], errors="coerce").max()
        weighted_growth: list[tuple[int, float]] = []
        weighted_inflation: list[tuple[int, float]] = []
        weighted_policy: list[tuple[int, float]] = []
        recency_scores: list[float] = []
        used_note_count = 0
        region_diagnostic_indexes: list[int] = []

        for _, row in region_notes.iterrows():
            age_days = int((pd.Timestamp(snapshot_date) - pd.Timestamp(row["date"])).days)
            note_weight = weighted_note_score(
                age_days=age_days,
                source_type=str(row.get("source_type", "other")),
                confidence=str(row.get("confidence", "low")),
            )
            used_in_snapshot = note_weight > 0
            ignore_reason = "" if used_in_snapshot else "ignored_due_to_age"
            if used_in_snapshot:
                used_note_count += 1
                recency_scores.append(note_weight)
                weighted_growth.append((GROWTH_SCALE[str(row["growth_view"])], note_weight))
                weighted_inflation.append((INFLATION_RISK_SCALE[str(row["inflation_view"])], note_weight))
                weighted_policy.append((POLICY_DOVISH_SCALE[str(row["policy_bias_view"])], note_weight))
            diagnostics_rows.append(
                {
                    "region": region,
                    "note_id": row["note_id"],
                    "source_name": row["source_name"],
                    "source_type": row["source_type"],
                    "date": row["date"],
                    "age_days": age_days,
                    "recency_weight": 1.0 if age_days <= 14 else 0.5 if age_days <= 30 else 0.0,
                    "note_weight": note_weight,
                    "used_in_snapshot": used_in_snapshot,
                    "ignore_reason": ignore_reason,
                    "growth_view": row["growth_view"],
                    "inflation_view": row["inflation_view"],
                    "policy_bias_view": row["policy_bias_view"],
                    "confidence": row["confidence"],
                    "classification_reason": row["classification_reason"],
                }
            )
            region_diagnostic_indexes.append(len(diagnostics_rows) - 1)

        def _weighted_average(items: list[tuple[int, float]]) -> float:
            if not items:
                return float("nan")
            numerator = sum(value * weight for value, weight in items)
            denominator = sum(weight for _, weight in items)
            return float("nan") if denominator == 0 else numerator / denominator

        growth_avg = _weighted_average(weighted_growth)
        inflation_avg = _weighted_average(weighted_inflation)
        policy_avg = _weighted_average(weighted_policy)
        source_recency_score = 0.0 if not recency_scores else min(1.0, sum(recency_scores) / max(used_note_count, 1))
        growth_consensus = label_growth_from_score(growth_avg)
        inflation_consensus = label_inflation_from_score(inflation_avg)
        policy_consensus = label_policy_from_score(policy_avg)
        growth_reason = _snapshot_reason("Growth", growth_consensus, used_note_count)
        inflation_reason = _snapshot_reason("Inflation", inflation_consensus, used_note_count)
        policy_reason = _snapshot_reason("Policy", policy_consensus, used_note_count)
        snapshot_rows.append(
            {
                "region": region,
                "snapshot_date": snapshot_date,
                "growth_consensus": growth_consensus,
                "inflation_consensus": inflation_consensus,
                "policy_bias_consensus": policy_consensus,
                "consensus_confidence": aggregate_confidence(used_note_count, source_recency_score),
                "source_count": used_note_count,
                "source_recency_score": source_recency_score,
                "latest_note_date": snapshot_date,
                "growth_reason": growth_reason,
                "inflation_reason": inflation_reason,
                "policy_reason": policy_reason,
            }
        )
        for index in region_diagnostic_indexes:
            diagnostics_rows[index]["snapshot_growth_consensus"] = growth_consensus
            diagnostics_rows[index]["snapshot_inflation_consensus"] = inflation_consensus
            diagnostics_rows[index]["snapshot_policy_bias_consensus"] = policy_consensus
            diagnostics_rows[index]["snapshot_reason"] = " | ".join([growth_reason, inflation_reason, policy_reason])

    snapshots = pd.DataFrame(snapshot_rows).sort_values("region").reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(["region", "date", "note_id"]).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(output_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    return snapshots


def build_deviation_summary(
    growth_deviation_score: float,
    inflation_deviation_score: float,
    policy_deviation_score: float,
) -> tuple[str, str]:
    """Build a human-readable summary and reason from deviation scores."""
    components = {
        "growth": growth_deviation_score,
        "inflation": inflation_deviation_score,
        "policy": policy_deviation_score,
    }
    finite = {key: value for key, value in components.items() if pd.notna(value)}
    if not finite or max(abs(value) for value in finite.values()) < 0.25:
        return (
            "Model is broadly aligned with consensus.",
            "Growth, inflation, and policy-bias signals are close to current public narratives.",
        )
    dominant = max(finite.items(), key=lambda item: abs(item[1]))
    dimension, score = dominant
    if dimension == "growth":
        summary = (
            "Model is more growth-positive than consensus."
            if score > 0
            else "Model is more growth-negative than consensus."
        )
        reason = (
            "The model reads stronger growth momentum than mainstream narratives."
            if score > 0
            else "The model reads weaker growth momentum than mainstream narratives."
        )
        return summary, reason
    if dimension == "inflation":
        summary = (
            "Model sees less inflation risk than current public narratives."
            if score > 0
            else "Model sees more inflation risk than current public narratives."
        )
        reason = (
            "The model's inflation view is more benign than consensus."
            if score > 0
            else "The model's inflation view is less benign than consensus."
        )
        return summary, reason
    summary = (
        "Model is more dovish than consensus."
        if score > 0
        else "Model is more hawkish than consensus."
    )
    reason = (
        "Liquidity conditions look easier than the mainstream policy narrative suggests."
        if score > 0
        else "Liquidity conditions look tighter than the mainstream policy narrative suggests."
    )
    return summary, reason


def build_consensus_deviation(
    snapshots_path: str | Path = CONSENSUS_SNAPSHOTS_PATH,
    output_path: str | Path = CONSENSUS_DEVIATION_PATH,
) -> pd.DataFrame:
    """Compare model region views with consensus snapshots."""
    snapshot_path = Path(snapshots_path)
    if not snapshot_path.exists():
        snapshots = build_consensus_snapshots()
    else:
        snapshots = pd.read_csv(snapshot_path)
        if not snapshots.empty and "snapshot_date" in snapshots.columns:
            snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"], errors="coerce")
    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        model = map_model_views(region)
        snapshot = snapshots.loc[snapshots["region"] == region].iloc[-1] if not snapshots.empty and (snapshots["region"] == region).any() else pd.Series(dtype="object")
        source_count = int(snapshot.get("source_count", 0)) if len(snapshot.index) > 0 else 0
        if source_count == 0:
            rows.append(
                {
                    "region": region,
                    "model_date": model["model_date"],
                    "snapshot_date": snapshot.get("snapshot_date"),
                    "model_regime": model["model_regime"],
                    "model_growth_view": model["model_growth_view"],
                    "model_inflation_view": model["model_inflation_view"],
                    "model_policy_bias_view": model["model_policy_bias_view"],
                    "growth_consensus": None,
                    "inflation_consensus": None,
                    "policy_bias_consensus": None,
                    "consensus_confidence": None,
                    "source_count": 0,
                    "source_recency_score": snapshot.get("source_recency_score"),
                    "latest_consensus_note_date": snapshot.get("latest_note_date"),
                    "growth_deviation_score": float("nan"),
                    "inflation_deviation_score": float("nan"),
                    "policy_deviation_score": float("nan"),
                    "consensus_deviation_score": float("nan"),
                    "deviation_summary": "No consensus snapshot is available for this region yet.",
                    "deviation_reason": "Add recent consensus notes before comparing the model with public narratives.",
                }
            )
            continue
        model_growth_view = model["model_growth_view"]
        model_inflation_view = model["model_inflation_view"]
        model_policy_view = model["model_policy_bias_view"]
        growth_score = deviation_score(
            GROWTH_SCALE.get(str(model_growth_view), 0),
            GROWTH_SCALE.get(str(snapshot.get("growth_consensus", "neutral")), 0),
        )
        inflation_score = deviation_score(
            INFLATION_RISK_SCALE.get(str(model_inflation_view), 0),
            INFLATION_RISK_SCALE.get(str(snapshot.get("inflation_consensus", "neutral")), 0),
        )
        policy_score = deviation_score(
            POLICY_DOVISH_SCALE.get(str(model_policy_view), 0),
            POLICY_DOVISH_SCALE.get(str(snapshot.get("policy_bias_consensus", "neutral")), 0),
        )
        summary, reason = build_deviation_summary(growth_score, inflation_score, policy_score)
        rows.append(
            {
                "region": region,
                "model_date": model["model_date"],
                "snapshot_date": snapshot.get("snapshot_date"),
                "model_regime": model["model_regime"],
                "model_growth_view": model_growth_view,
                "model_inflation_view": model_inflation_view,
                "model_policy_bias_view": model_policy_view,
                "growth_consensus": snapshot.get("growth_consensus"),
                "inflation_consensus": snapshot.get("inflation_consensus"),
                "policy_bias_consensus": snapshot.get("policy_bias_consensus"),
                "consensus_confidence": snapshot.get("consensus_confidence"),
                "source_count": source_count,
                "source_recency_score": snapshot.get("source_recency_score"),
                "latest_consensus_note_date": snapshot.get("latest_note_date"),
                "growth_deviation_score": growth_score,
                "inflation_deviation_score": inflation_score,
                "policy_deviation_score": policy_score,
                "consensus_deviation_score": safe_mean([growth_score, inflation_score, policy_score]),
                "deviation_summary": summary,
                "deviation_reason": reason,
            }
        )
    deviation = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    deviation.to_csv(output_path, index=False)
    return deviation
