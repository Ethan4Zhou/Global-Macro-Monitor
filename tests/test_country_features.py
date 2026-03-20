"""Tests for country-specific feature generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.factors.features import build_country_macro_features


def _write_manual_series(
    base_dir: Path,
    series_id: str,
    dates: pd.DatetimeIndex,
    values: list[float],
) -> None:
    """Write one mocked manual country series CSV."""
    frame = pd.DataFrame({"date": dates, "value": values, "series_id": series_id})
    frame.to_csv(base_dir / f"{series_id}.csv", index=False)


def test_build_country_macro_features_for_manual_country(tmp_path: Path) -> None:
    """Country features should build from manual CSV inputs."""
    manual_dir = tmp_path / "manual" / "china"
    manual_dir.mkdir(parents=True)
    output_path = tmp_path / "china_macro_features.csv"

    dates = pd.date_range("2024-01-01", periods=15, freq="MS")
    _write_manual_series(manual_dir, "cpi", dates, [100 + i for i in range(15)])
    _write_manual_series(manual_dir, "core_cpi", dates, [102 + i for i in range(15)])
    _write_manual_series(manual_dir, "unrate", dates, [5.2, 5.1, 5.0, 4.9, 4.9, 4.8, 4.8, 4.7, 4.7, 4.6, 4.6, 4.5, 4.5, 4.4, 4.4])
    _write_manual_series(manual_dir, "policy_rate", dates, [2.5] * 15)
    _write_manual_series(manual_dir, "yield_10y", dates, [2.7, 2.7, 2.8, 2.8, 2.8, 2.9, 2.9, 2.9, 3.0, 3.0, 3.0, 3.0, 3.1, 3.1, 3.1])
    _write_manual_series(manual_dir, "m2", dates, [200 + 5 * i for i in range(15)])

    result = build_country_macro_features(
        country="china",
        manual_base_dir=str(tmp_path / "manual"),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert set(["date", "country", "cpi_yoy", "policy_rate_level", "yield_10y_level"]).issubset(result.columns)
    assert set(result["country"]) == {"china"}


def test_build_country_macro_features_falls_back_from_api_to_manual(tmp_path: Path) -> None:
    """When API raw files are absent, manual fallback should still work."""
    manual_dir = tmp_path / "manual" / "eurozone"
    manual_dir.mkdir(parents=True)
    output_path = tmp_path / "eurozone_macro_features.csv"

    dates = pd.date_range("2024-01-01", periods=15, freq="MS")
    _write_manual_series(manual_dir, "cpi", dates, [100 + i for i in range(15)])
    _write_manual_series(manual_dir, "core_cpi", dates, [101 + i for i in range(15)])
    _write_manual_series(manual_dir, "unrate", dates, [6.5 - 0.05 * i for i in range(15)])
    _write_manual_series(manual_dir, "policy_rate", dates, [4.0] * 15)
    _write_manual_series(manual_dir, "yield_10y", dates, [2.2 + 0.02 * i for i in range(15)])
    _write_manual_series(manual_dir, "m2", dates, [200 + 2 * i for i in range(15)])
    _write_manual_series(manual_dir, "pmi", dates, [48 + 0.2 * i for i in range(15)])

    result = build_country_macro_features(
        country="eurozone",
        manual_base_dir=str(tmp_path / "manual"),
        api_base_dir=str(tmp_path / "api"),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert result["country"].unique().tolist() == ["eurozone"]
    assert result["pmi_level"].notna().any()


def test_build_country_macro_features_prefers_china_normalized_api_over_manual(tmp_path: Path) -> None:
    """China feature builder should use normalized API files before manual fallback."""
    manual_dir = tmp_path / "manual" / "china"
    api_dir = tmp_path / "api" / "china" / "normalized"
    manual_dir.mkdir(parents=True)
    api_dir.mkdir(parents=True)
    output_path = tmp_path / "china_macro_features.csv"

    dates = pd.date_range("2024-01-01", periods=15, freq="MS")
    _write_manual_series(manual_dir, "cpi", dates, [100 + i for i in range(15)])
    _write_manual_series(manual_dir, "pmi", dates, [49 + 0.1 * i for i in range(15)])
    _write_manual_series(manual_dir, "policy_rate", dates, [2.4] * 15)
    _write_manual_series(manual_dir, "yield_10y", dates, [2.6] * 15)

    pd.DataFrame(
        {
            "date": dates,
            "value": [200 + i for i in range(15)],
            "series_id": ["cpi"] * 15,
            "country": ["china"] * 15,
            "source": ["china_nbs"] * 15,
            "frequency": ["monthly"] * 15,
            "release_date": [pd.NaT] * 15,
            "ingested_at": pd.to_datetime(["2024-03-01"] * 15),
        }
    ).to_csv(api_dir / "cpi.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "value": [50 + 0.1 * i for i in range(15)],
            "series_id": ["pmi"] * 15,
            "country": ["china"] * 15,
            "source": ["china_nbs"] * 15,
            "frequency": ["monthly"] * 15,
            "release_date": [pd.NaT] * 15,
            "ingested_at": pd.to_datetime(["2024-03-01"] * 15),
        }
    ).to_csv(api_dir / "pmi.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "value": [2.1] * 15,
            "series_id": ["policy_rate"] * 15,
            "country": ["china"] * 15,
            "source": ["china_rates"] * 15,
            "frequency": ["monthly"] * 15,
            "release_date": [pd.NaT] * 15,
            "ingested_at": pd.to_datetime(["2024-03-01"] * 15),
        }
    ).to_csv(api_dir / "policy_rate.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "value": [2.3] * 15,
            "series_id": ["yield_10y"] * 15,
            "country": ["china"] * 15,
            "source": ["china_rates"] * 15,
            "frequency": ["monthly"] * 15,
            "release_date": [pd.NaT] * 15,
            "ingested_at": pd.to_datetime(["2024-03-01"] * 15),
        }
    ).to_csv(api_dir / "yield_10y.csv", index=False)

    result = build_country_macro_features(
        country="china",
        manual_base_dir=str(tmp_path / "manual"),
        api_base_dir=str(tmp_path / "api"),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert result["cpi_level"].iloc[0] == 200


def test_build_country_macro_features_prefers_eurozone_normalized_api_over_manual(tmp_path: Path) -> None:
    """Eurozone feature builder should prefer normalized API files before manual fallback."""
    manual_dir = tmp_path / "manual" / "eurozone"
    api_dir = tmp_path / "api" / "eurozone" / "normalized"
    manual_dir.mkdir(parents=True)
    api_dir.mkdir(parents=True)
    output_path = tmp_path / "eurozone_macro_features.csv"

    dates = pd.date_range("2024-01-01", periods=15, freq="MS")
    _write_manual_series(manual_dir, "cpi", dates, [100 + i for i in range(15)])
    _write_manual_series(manual_dir, "pmi", dates, [48 + 0.1 * i for i in range(15)])
    _write_manual_series(manual_dir, "policy_rate", dates, [4.0] * 15)
    _write_manual_series(manual_dir, "yield_10y", dates, [2.5] * 15)

    for series_id, value, source in [
        ("cpi", [200 + i for i in range(15)], "eurostat"),
        ("growth_proxy", [50 + 0.1 * i for i in range(15)], "eurostat"),
        ("policy_rate", [3.0] * 15, "ecb"),
        ("yield_10y", [2.0] * 15, "ecb"),
    ]:
        pd.DataFrame(
            {
                "date": dates,
                "value": value,
                "series_id": [series_id] * 15,
                "country": ["eurozone"] * 15,
                "source": [source] * 15,
                "frequency": ["monthly"] * 15,
                "release_date": [pd.NaT] * 15,
                "ingested_at": pd.to_datetime(["2024-03-01"] * 15),
            }
        ).to_csv(api_dir / f"{series_id}.csv", index=False)

    result = build_country_macro_features(
        country="eurozone",
        manual_base_dir=str(tmp_path / "manual"),
        api_base_dir=str(tmp_path / "api"),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert result["cpi_level"].iloc[0] == 200
    assert result["pmi_level"].iloc[0] == 50
