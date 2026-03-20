"""Tests for manual CSV loading and readiness checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.data.manual_loader import assess_manual_country_readiness, load_manual_csv


def test_load_manual_csv_accepts_valid_file(tmp_path: Path) -> None:
    """A valid manual CSV should load successfully."""
    file_path = tmp_path / "cpi.csv"
    file_path.write_text(
        "date,value,series_id\n2024-01-01,0.2,cpi\n2024-02-01,0.4,cpi\n",
        encoding="utf-8",
    )

    frame = load_manual_csv(file_path)

    assert list(frame.columns) == ["date", "value", "series_id"]
    assert len(frame) == 2


def test_load_manual_csv_rejects_invalid_file(tmp_path: Path) -> None:
    """Malformed manual CSVs should raise a clear error."""
    file_path = tmp_path / "bad.csv"
    file_path.write_text("date,value\n2024-01-01,abc\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_manual_csv(file_path)


def test_assess_manual_country_readiness_reports_missing_series(tmp_path: Path) -> None:
    """Readiness should report missing minimum manual series."""
    country_dir = tmp_path / "china"
    country_dir.mkdir(parents=True)
    (country_dir / "cpi.csv").write_text(
        "date,value,series_id\n2024-01-01,0.2,cpi\n",
        encoding="utf-8",
    )
    (country_dir / "pmi.csv").write_text(
        "date,value,series_id\n2024-01-01,49.2,pmi\n",
        encoding="utf-8",
    )

    result = assess_manual_country_readiness("china", base_dir=str(tmp_path))

    assert result["ready"] is False
    assert "policy_rate" in result["missing_series"]
