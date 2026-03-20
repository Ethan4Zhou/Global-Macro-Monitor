"""Tests for feature engineering helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.factors.features import build_us_macro_features


def _write_series_csv(
    base_dir: Path,
    series_id: str,
    dates: pd.DatetimeIndex,
    values: list[float],
) -> None:
    """Write one mocked raw FRED series CSV."""
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "series_id": series_id,
        }
    )
    frame.to_csv(base_dir / f"{series_id}.csv", index=False)


def test_build_us_macro_features_from_mocked_csvs(tmp_path: Path) -> None:
    """Feature builder should align series and compute monthly transformations."""
    raw_dir = tmp_path / "raw"
    processed_path = tmp_path / "us_macro_features.csv"
    raw_dir.mkdir()

    dates = pd.date_range("2023-01-01", periods=15, freq="MS")
    _write_series_csv(raw_dir, "CPIAUCSL", dates, [100 + i for i in range(15)])
    _write_series_csv(raw_dir, "CPILFESL", dates, [200 + 2 * i for i in range(15)])
    _write_series_csv(raw_dir, "UNRATE", dates, [4.5, 4.4, 4.3, 4.2, 4.2, 4.1, 4.0, 4.0, 3.9, 3.9, 3.8, 3.8, 3.7, 3.7, 3.6])
    _write_series_csv(raw_dir, "FEDFUNDS", dates, [5.5, 5.5, 5.5, 5.4, 5.3, 5.2, 5.1, 5.0, 4.9, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4])
    _write_series_csv(raw_dir, "GS10", dates, [4.0, 4.1, 4.2, 4.1, 4.0, 3.9, 3.8, 3.8, 3.7, 3.6, 3.6, 3.5, 3.4, 3.4, 3.3])
    _write_series_csv(raw_dir, "M2SL", dates, [1000 + 10 * i for i in range(15)])

    result = build_us_macro_features(
        input_dir=str(raw_dir),
        output_path=str(processed_path),
    )

    last_row = result.iloc[-1]
    assert processed_path.exists()
    assert "cpi_yoy" in result.columns
    assert "unrate_3m_avg" in result.columns
    assert "fedfunds_diff_3m" in result.columns
    assert round(float(last_row["cpi_yoy"]), 2) == 11.76
    assert round(float(last_row["unrate_3m_avg"]), 2) == 3.67
    assert round(float(last_row["fedfunds_diff_3m"]), 2) == -0.30
