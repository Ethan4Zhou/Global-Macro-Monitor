"""Tests for China API ingestion and validation."""

from __future__ import annotations

from pathlib import Path
import types
from unittest.mock import Mock, patch

import pandas as pd

from app.data.china_ingestion import (
    canonicalize_china_series_id,
    fetch_china_api_bundle,
    rebuild_china_normalized_data,
    validate_china_data,
)
from app.data.sources.china_akshare_client import fetch_china_akshare_series
from app.data.sources.china_nbs_client import fetch_china_nbs_series
from app.data.sources.china_nbs_client import (
    extract_core_cpi_from_release_text,
    extract_unrate_from_release_text,
)
from app.data.sources.china_rates_client import fetch_china_rates_series
from app.data.sources.imf_client import fetch_imf_series


def test_china_akshare_adapter_normalizes_payload(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """AkShare adapter should normalize canonical China series ids and columns."""
    fake_ak = types.SimpleNamespace(
        macro_china_cpi=lambda: pd.DataFrame(
            {"月份": ["2024年01月份", "2024年02月份"], "全国-同比增长": [0.2, 0.4]}
        )
    )
    monkeypatch.setattr(
        "app.data.sources.china_akshare_client.importlib.import_module",
        lambda _name: fake_ak,
    )

    result = fetch_china_akshare_series("cpi", country="china", frequency="monthly")
    assert list(result.columns) == ["date", "value", "series_id", "country", "source", "frequency", "release_date", "ingested_at"]
    assert result["series_id"].unique().tolist() == ["cpi"]
    assert result["source"].unique().tolist() == ["china_akshare"]


@patch("app.data.sources.china_nbs_client.requests.get")
def test_china_nbs_adapter_normalizes_payload(mock_get: Mock) -> None:
    """China NBS adapter should normalize a mocked macro payload."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "data": [
            {"month": "202401", "value": "0.3"},
            {"month": "202402", "value": "0.7"},
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_china_nbs_series("cpi", country="china", frequency="monthly")
    assert result["series_id"].iloc[0] == "cpi"
    assert result["source"].iloc[0] == "china_nbs"
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_nbs_text_extraction_parses_core_cpi_and_unrate() -> None:
    """NBS English monthly release text should yield core CPI and unemployment."""
    text = (
        "In February 2026, the consumer price index excluding food and energy increased 0.6 percent year on year. "
        "In February 2026, the surveyed urban unemployment rate was 5.3 percent."
    )

    core_cpi = extract_core_cpi_from_release_text(text)
    unrate = extract_unrate_from_release_text(text)

    assert float(core_cpi.loc[0, "value"]) == 0.6
    assert float(unrate.loc[0, "value"]) == 5.3
    assert str(pd.Timestamp(core_cpi.loc[0, "date"]).date()) == "2026-02-01"


@patch("app.data.sources.china_rates_client.requests.get")
def test_china_rates_adapter_parses_csv(mock_get: Mock) -> None:
    """China rates adapter should normalize a mocked CSV payload."""
    mock_response = Mock()
    mock_response.text = "date,yield\n2024-01-01,2.4\n2024-02-01,2.5\n"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_china_rates_series("yield_10y", country="china", frequency="monthly")
    assert result["value"].tolist() == [2.4, 2.5]
    assert result["source"].iloc[0] == "china_rates"


@patch("app.data.sources.imf_client.requests.get")
def test_imf_adapter_normalizes_payload(mock_get: Mock) -> None:
    """IMF adapter should normalize a mocked fallback payload."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "observations": [
            {"date": "2024-01-01", "value": "5.1"},
            {"date": "2024-02-01", "value": "5.0"},
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_imf_series("unrate", country="china", frequency="monthly")
    assert result["source"].iloc[0] == "imf"
    assert result["country"].iloc[0] == "china"


def test_fetch_china_api_bundle_deduplicates_and_saves(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """China ingestion should deduplicate rows and save normalized outputs."""
    def _fake_tushare(*args, **kwargs):  # type: ignore[no-untyped-def]
        series_id = kwargs["source_series_id"]
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]),
                "value": [0.2, 0.2, 0.4],
                "series_id": [series_id, series_id, series_id],
                "country": ["china"] * 3,
                "source": ["tushare"] * 3,
                "frequency": ["monthly"] * 3,
                "release_date": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-02-20"]),
                "ingested_at": pd.to_datetime(["2024-03-01"] * 3),
            }
        )

    def _fake_nbs(*args, **kwargs):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]),
                "value": [0.2, 0.2, 0.4],
                "series_id": ["cpi", "cpi", "cpi"],
                "country": ["china"] * 3,
                "source": ["china_nbs"] * 3,
                "frequency": ["monthly"] * 3,
                "release_date": [pd.NaT] * 3,
                "ingested_at": pd.to_datetime(["2024-03-01"] * 3),
            }
        )

    def _fake_rates(*args, **kwargs):  # type: ignore[no-untyped-def]
        series_id = kwargs["source_series_id"]
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "value": [2.1, 2.2],
                "series_id": [series_id, series_id],
                "country": ["china", "china"],
                "source": ["china_rates", "china_rates"],
                "frequency": ["monthly", "monthly"],
                "release_date": [pd.NaT, pd.NaT],
                "ingested_at": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            }
        )

    monkeypatch.setattr("app.data.china_ingestion.fetch_china_nbs_series", _fake_nbs)
    monkeypatch.setattr("app.data.china_ingestion.fetch_china_rates_series", _fake_rates)
    monkeypatch.setattr("app.data.china_ingestion.fetch_imf_series", _fake_nbs)
    monkeypatch.setattr(
        "app.data.china_ingestion.CHINA_SOURCE_FETCHERS",
        {
            "tushare": (_fake_tushare, "tushare"),
            "china_nbs": (_fake_nbs, "nbs"),
            "china_rates": (_fake_rates, "rates"),
            "imf": (_fake_nbs, "imf"),
        },
    )

    summary = fetch_china_api_bundle(base_dir=str(tmp_path / "china"))
    assert not summary.empty
    assert (tmp_path / "china" / "normalized" / "cpi.csv").exists()
    assert (tmp_path / "china" / "tushare" / "cpi.csv").exists()
    cpi = pd.read_csv(tmp_path / "china" / "normalized" / "cpi.csv")
    assert len(cpi) == 2
    assert summary.loc[summary["series_id"] == "cpi", "source_used"].iloc[0] == "tushare"


def test_validate_china_data_reports_missing_required_series(tmp_path: Path) -> None:
    """Validation should flag missing minimum series for regime readiness."""
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir(parents=True)
    pd.DataFrame(
        {"date": ["2024-01-01"], "value": [0.2], "series_id": ["cpi"], "country": ["china"], "source": ["china_nbs"], "frequency": ["monthly"], "release_date": [""], "ingested_at": ["2024-03-01"]}
    ).to_csv(normalized_dir / "cpi.csv", index=False)
    pd.DataFrame(
        {"date": ["2024-01-01"], "value": [49.5], "series_id": ["pmi"], "country": ["china"], "source": ["china_nbs"], "frequency": ["monthly"], "release_date": [""], "ingested_at": ["2024-03-01"]}
    ).to_csv(normalized_dir / "pmi.csv", index=False)

    result = validate_china_data(base_dir=str(normalized_dir))
    assert result["regime_ready"] is False
    assert "policy_rate" in result["missing_required_series"]


def test_validate_china_data_reports_enrichment_richness(tmp_path: Path) -> None:
    """Validation should distinguish minimum inputs from enrichment inputs."""
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir(parents=True)
    base_rows = {
        "cpi": 0.2,
        "pmi": 49.5,
        "policy_rate": 2.1,
        "yield_10y": 2.4,
        "m2": 7.2,
        "industrial_production": 5.1,
    }
    for series_id, value in base_rows.items():
        pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "value": [value],
                "series_id": [series_id],
                "country": ["china"],
                "source": ["china_akshare"],
                "frequency": ["monthly"],
                "release_date": [""],
                "ingested_at": ["2024-03-01"],
            }
        ).to_csv(normalized_dir / f"{series_id}.csv", index=False)

    result = validate_china_data(base_dir=str(normalized_dir))
    assert result["regime_ready"] is True
    assert "m2" in result["enrichment_available_series"]
    assert result["scoring_richness_level"] in {"minimum", "enhanced", "enhanced_partially_stale", "rich"}


def test_canonicalize_china_series_id_maps_aliases() -> None:
    """Source-specific China series ids should map into canonical regime ids."""
    assert canonicalize_china_series_id("short_rate_proxy") == "policy_rate"
    assert canonicalize_china_series_id("cgb_10y_proxy") == "yield_10y"
    assert canonicalize_china_series_id("consumer_price_index") == "cpi"


def test_validator_recognizes_normalized_files_with_mapped_series_ids(tmp_path: Path) -> None:
    """Validator should not miss required inputs when normalized files use canonical ids."""
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir(parents=True)
    for series_id, value in {
        "cpi": 0.2,
        "pmi": 49.5,
        "policy_rate": 2.1,
        "yield_10y": 2.4,
    }.items():
        pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "value": [value],
                "series_id": [series_id],
                "country": ["china"],
                "source": ["china_nbs"],
                "frequency": ["monthly"],
                "release_date": [""],
                "ingested_at": ["2024-03-01"],
            }
        ).to_csv(normalized_dir / f"{series_id}.csv", index=False)

    result = validate_china_data(base_dir=str(normalized_dir))
    assert result["regime_ready"] is True
    assert result["missing_required_series"] == []
    assert sorted(result["series_ids_found"]) == ["cpi", "pmi", "policy_rate", "yield_10y"]


def test_rebuild_china_normalized_data_maps_source_files(tmp_path: Path) -> None:
    """Rebuild should normalize source-specific raw files into canonical ids."""
    rates_dir = tmp_path / "china" / "rates"
    rates_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "value": [2.2],
            "series_id": ["short_rate_proxy"],
            "country": ["china"],
            "source": ["china_rates"],
            "frequency": ["monthly"],
            "release_date": [""],
            "ingested_at": ["2024-03-01"],
        }
    ).to_csv(rates_dir / "policy_rate.csv", index=False)

    summary = rebuild_china_normalized_data(base_dir=str(tmp_path / "china"))
    normalized = pd.read_csv(tmp_path / "china" / "normalized" / "policy_rate.csv")
    assert "policy_rate" in summary["series_id"].tolist()
    assert normalized["series_id"].iloc[0] == "policy_rate"


def test_fetch_china_api_bundle_tolerates_fallback_failure(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Optional China fallbacks should not crash the whole ingestion run."""
    def _fake_tushare(*args, **kwargs):  # type: ignore[no-untyped-def]
        series_id = kwargs["source_series_id"]
        if series_id == "industrial_production":
            raise RuntimeError("primary failed")
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "value": [1.0],
                "series_id": [series_id],
                "country": ["china"],
                "source": ["tushare"],
                "frequency": ["monthly"],
                "release_date": pd.to_datetime(["2024-01-15"]),
                "ingested_at": pd.to_datetime(["2024-03-01"]),
            }
        )

    def _raise_fallback(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("fallback failed")

    monkeypatch.setattr(
        "app.data.china_ingestion.CHINA_SOURCE_FETCHERS",
        {
            "tushare": (_fake_tushare, "tushare"),
            "china_akshare": (_fake_tushare, "akshare"),
            "china_nbs": (_raise_fallback, "nbs"),
            "china_rates": (_fake_tushare, "rates"),
            "imf": (_raise_fallback, "imf"),
        },
    )

    summary = fetch_china_api_bundle(base_dir=str(tmp_path / "china"))
    assert not summary.empty
    industrial_row = summary.loc[summary["series_id"] == "industrial_production"].iloc[0]
    assert industrial_row["status"] == "missing"


def test_fetch_china_api_bundle_uses_public_site_fallback_series_id(monkeypatch, tmp_path: Path) -> None:
    """China ingestion should honor fallback_source_series_id for public-site fallbacks."""

    def _raise_primary(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("primary failed")

    def _public_fetch(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["source_series_id"] == "china_core_cpi_public"
        return pd.DataFrame(
            {
                "date": ["2026-02-01"],
                "value": [0.6],
                "series_id": ["core_cpi"],
                "country": ["china"],
                "source": ["tradingeconomics"],
                "frequency": ["monthly"],
                "release_date": ["2026-02-01"],
                "ingested_at": ["2026-03-22T00:00:00+00:00"],
            }
        )

    monkeypatch.setattr(
        "app.data.china_ingestion.get_country_indicators",
        lambda country, group: (
            [
                {
                    "key": "core_cpi",
                    "source": "china_nbs",
                    "source_series_id": "core_cpi",
                    "fallback_source": "public_site",
                    "fallback_source_series_id": "china_core_cpi_public",
                    "frequency": "monthly",
                    "required_for_minimum_regime": False,
                }
            ]
            if group == "macro"
            else []
        ),
    )
    monkeypatch.setattr(
        "app.data.china_ingestion.CHINA_SOURCE_FETCHERS",
        {
            "china_nbs": (_raise_primary, "nbs"),
            "public_site": (_public_fetch, "public"),
        },
    )

    summary = fetch_china_api_bundle(base_dir=str(tmp_path / "china"))
    row = summary.loc[summary["series_id"] == "core_cpi"].iloc[0]
    assert row["source_used"] == "public_site"
    assert row["status"] == "ready"
    assert (tmp_path / "china" / "normalized" / "core_cpi.csv").exists()


def test_fetch_china_api_bundle_prefers_newer_public_site_series(monkeypatch, tmp_path: Path) -> None:
    """China ingestion should switch to a newer public-site series when the primary source is stale."""

    def _stale_primary(*args, **kwargs):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            {
                "date": ["2025-08-01"],
                "value": [5.2],
                "series_id": ["industrial_production"],
                "country": ["china"],
                "source": ["tushare"],
                "frequency": ["monthly"],
                "release_date": ["2025-08-01"],
                "ingested_at": ["2026-03-22T00:00:00+00:00"],
            }
        )

    def _newer_public(*args, **kwargs):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            {
                "date": ["2026-02-01"],
                "value": [5.8],
                "series_id": ["industrial_production"],
                "country": ["china"],
                "source": ["tradingeconomics"],
                "frequency": ["monthly"],
                "release_date": ["2026-02-01"],
                "ingested_at": ["2026-03-22T00:00:00+00:00"],
            }
        )

    monkeypatch.setattr(
        "app.data.china_ingestion.get_country_indicators",
        lambda country, group: (
            [
                {
                    "key": "industrial_production",
                    "source": "tushare",
                    "source_series_id": "industrial_production",
                    "fallback_source": "public_site",
                    "fallback_source_series_id": "china_industrial_production_public",
                    "frequency": "monthly",
                    "required_for_minimum_regime": False,
                }
            ]
            if group == "macro"
            else []
        ),
    )
    monkeypatch.setattr(
        "app.data.china_ingestion.CHINA_SOURCE_FETCHERS",
        {
            "tushare": (_stale_primary, "tushare"),
            "public_site": (_newer_public, "public"),
        },
    )

    summary = fetch_china_api_bundle(base_dir=str(tmp_path / "china"))
    row = summary.loc[summary["series_id"] == "industrial_production"].iloc[0]
    assert row["source_used"] == "public_site"
    assert row["latest_date"] == pd.Timestamp("2026-02-01")
