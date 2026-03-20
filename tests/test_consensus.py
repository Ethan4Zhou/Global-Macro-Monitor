"""Tests for consensus note ingestion and deviation scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.consensus.deviation import (
    build_consensus_deviation,
    build_consensus_snapshots,
    build_deviation_summary,
    map_model_views,
)
from app.consensus.parser import parse_consensus_notes
from app.consensus.sources import ingest_consensus_notes, load_consensus_notes_from_path


def test_consensus_note_normalization_from_text(tmp_path: Path) -> None:
    """Markdown notes should normalize into the shared schema."""
    note = tmp_path / "fed.md"
    note.write_text(
        "source_name: Federal Reserve\n"
        "source_type: official\n"
        "date: 2026-03-15\n"
        "title: Fed stays patient\n\n"
        "Growth looks resilient and inflation is easing. Policy is supportive.\n",
        encoding="utf-8",
    )
    frame = load_consensus_notes_from_path(region="us", path=str(note))
    assert frame.loc[0, "region"] == "us"
    assert frame.loc[0, "source_name"] == "Federal Reserve"
    assert frame.loc[0, "title"] == "Fed stays patient"


def test_rules_based_stance_extraction() -> None:
    """Rules-based parser should classify growth, inflation, and policy views."""
    notes = pd.DataFrame(
        [
            {
                "region": "us",
                "source_name": "Reuters",
                "source_type": "media",
                "date": pd.Timestamp("2026-03-15"),
                "title": "Resilient growth with easing inflation",
                "body": "Analysts see resilient growth, cooling inflation, and cuts ahead.",
                "note_id": "note1",
                "ingestion_timestamp": pd.Timestamp("2026-03-20"),
            }
        ]
    )
    parsed = parse_consensus_notes(notes)
    assert parsed.loc[0, "growth_view"] == "positive"
    assert parsed.loc[0, "inflation_view"] == "disinflation"
    assert parsed.loc[0, "policy_bias_view"] == "dovish"


def test_region_level_aggregation_prefers_recent_official_notes(tmp_path: Path) -> None:
    """Recent official notes should drive the consensus snapshot more than stale notes."""
    notes = pd.DataFrame(
        [
            {
                "region": "eurozone",
                "source_name": "ECB",
                "source_type": "official",
                "date": "2026-03-18",
                "title": "Resilient activity and easing inflation",
                "body": "Officials describe resilient growth, disinflation, and an easing bias.",
                "note_id": "recent1",
                "ingestion_timestamp": "2026-03-20",
            },
            {
                "region": "eurozone",
                "source_name": "Old media",
                "source_type": "media",
                "date": "2026-01-01",
                "title": "Slowdown fears",
                "body": "Commentary warns of slowdown and sticky inflation.",
                "note_id": "old1",
                "ingestion_timestamp": "2026-03-20",
            },
        ]
    )
    notes_path = tmp_path / "consensus_notes.csv"
    notes.to_csv(notes_path, index=False)
    snapshots = build_consensus_snapshots(
        notes_path=notes_path,
        output_path=tmp_path / "snapshots.csv",
        diagnostics_path=tmp_path / "diagnostics.csv",
    )
    row = snapshots.iloc[0]
    assert row["growth_consensus"] in {"positive", "strong_positive"}
    assert row["source_count"] == 1


def test_model_to_consensus_mapping(tmp_path: Path, monkeypatch) -> None:
    """Model region state should map into consensus comparison labels."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-02-01"],
            "country": ["china"],
            "growth_score": [0.4],
            "inflation_score": [-0.6],
            "liquidity_score": [0.5],
            "regime": ["goldilocks"],
            "liquidity_regime": ["easy"],
        }
    ).to_csv(processed / "china_macro_regimes.csv", index=False)
    monkeypatch.chdir(tmp_path)
    mapped = map_model_views("china")
    assert mapped["model_growth_view"] == "positive"
    assert mapped["model_inflation_view"] == "disinflation"
    assert mapped["model_policy_bias_view"] == "strongly_dovish"


def test_deviation_scoring_direction(tmp_path: Path, monkeypatch) -> None:
    """Deviation scores should be positive when the model is more benign than consensus."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-02-01"],
            "country": ["us"],
            "growth_score": [0.6],
            "inflation_score": [-0.7],
            "liquidity_score": [0.5],
            "regime": ["goldilocks"],
            "liquidity_regime": ["easy"],
        }
    ).to_csv(processed / "us_macro_regimes.csv", index=False)
    pd.DataFrame(
        {
            "region": ["us"],
            "snapshot_date": ["2026-03-19"],
            "growth_consensus": ["neutral"],
            "inflation_consensus": ["inflationary"],
            "policy_bias_consensus": ["hawkish"],
            "consensus_confidence": ["medium"],
            "source_count": [2],
            "source_recency_score": [0.8],
            "latest_note_date": ["2026-03-19"],
        }
    ).to_csv(processed / "consensus_snapshots.csv", index=False)
    monkeypatch.chdir(tmp_path)
    deviation = build_consensus_deviation(
        snapshots_path=processed / "consensus_snapshots.csv",
        output_path=processed / "consensus_deviation.csv",
    )
    row = deviation.iloc[0]
    assert row["growth_deviation_score"] > 0
    assert row["inflation_deviation_score"] > 0
    assert row["policy_deviation_score"] > 0


def test_dashboard_friendly_summary_generation() -> None:
    """Deviation summaries should stay readable."""
    summary, reason = build_deviation_summary(0.5, 0.0, 0.0)
    assert "growth-positive" in summary
    assert reason.endswith(".")


def test_ingest_consensus_notes_deduplicates(tmp_path: Path) -> None:
    """Ingested notes should deduplicate on note_id."""
    folder = tmp_path / "notes"
    folder.mkdir()
    note = folder / "ecb.md"
    note.write_text(
        "source_name: ECB\n"
        "date: 2026-03-19\n"
        "title: ECB note\n\n"
        "Disinflation continues and policy stays restrictive.\n",
        encoding="utf-8",
    )
    output = tmp_path / "consensus_notes.csv"
    first = ingest_consensus_notes("eurozone", str(folder), output_path=output)
    second = ingest_consensus_notes("eurozone", str(folder), output_path=output)
    assert len(first) == len(second) == 1
