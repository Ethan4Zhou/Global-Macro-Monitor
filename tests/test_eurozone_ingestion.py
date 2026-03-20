"""Tests for Eurozone ingestion and normalization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.data.eurozone_ingestion import (
    EUROZONE_SOURCE_FETCHERS,
    canonicalize_eurozone_series_id,
    fetch_eurozone_api_bundle,
    validate_eurozone_data,
)
from app.data.sources.eurozone_ecb_client import fetch_eurozone_ecb_series
from app.data.sources.eurozone_eurostat_client import fetch_eurozone_eurostat_series
from main import run_fetch_country_api_data


@patch("app.data.sources.ecb_client.requests.get")
def test_eurozone_ecb_adapter_normalizes_payload(mock_get: Mock) -> None:
    """ECB wrapper should return normalized Eurozone data."""
    mock_response = Mock()
    mock_response.headers = {"content-type": "text/csv"}
    mock_response.text = "TIME_PERIOD,OBS_VALUE\n2024-01,4.0\n2024-02,3.9\n"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_eurozone_ecb_series("TEST", country="eurozone", frequency="monthly")
    assert list(result.columns) == ["date", "value", "series_id", "country", "source", "frequency", "release_date", "ingested_at"]
    assert result["country"].unique().tolist() == ["eurozone"]


@patch("app.data.sources.ecb_client.requests.get")
def test_ecb_client_uses_current_data_portal_endpoint(mock_get: Mock) -> None:
    """ECB fetches should target the current Data Portal API endpoint."""
    mock_response = Mock()
    mock_response.headers = {"content-type": "text/csv"}
    mock_response.text = "TIME_PERIOD,OBS_VALUE\n2024-01,4.0\n"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    fetch_eurozone_ecb_series("TEST", country="eurozone", frequency="monthly")

    called_url = mock_get.call_args.args[0]
    assert called_url.startswith("https://data-api.ecb.europa.eu/service/data/")


@patch("app.data.sources.eurostat_client.requests.get")
def test_eurozone_eurostat_adapter_normalizes_payload(mock_get: Mock) -> None:
    """Eurostat wrapper should normalize JSON-stat data."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "dimension": {"time": {"category": {"index": {"2024M01": 0, "2024M02": 1}}}},
        "value": {"0": 2.8, "1": 2.7},
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_eurozone_eurostat_series("TEST", country="eurozone", frequency="monthly")
    assert result["source"].iloc[0] == "eurostat"
    assert len(result) == 2


@patch("app.data.sources.eurozone_eurostat_client.requests.get")
@patch("app.data.sources.eurozone_eurostat_client.fetch_eurostat_series")
def test_eurozone_eurostat_adapter_appends_flash_inflation_when_newer(
    mock_fetch_eurostat_series: Mock,
    mock_get: Mock,
) -> None:
    """Eurostat wrapper should append a newer official flash inflation observation."""
    mock_fetch_eurostat_series.return_value = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-01"]),
            "value": [2.0],
            "series_id": ["cpi"],
            "country": ["eurozone"],
            "source": ["eurostat"],
            "frequency": ["monthly"],
            "release_date": pd.to_datetime(["2026-02-06"]),
            "ingested_at": pd.to_datetime(["2026-03-20"]),
        }
    )
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = """
    <html><body>
    <table>
      <thead>
        <tr><th colspan=\"4\">Euro area annual inflation and its components (%)</th></tr>
        <tr><th></th><th>Annual rate</th><th>Annual rate</th><th>Monthly rate</th></tr>
        <tr><th></th><th>Jan 26</th><th>Feb 26</th><th>Feb 26</th></tr>
      </thead>
      <tbody>
        <tr><td>All-items HICP</td><td>1.7</td><td>1.9e</td><td>0.7e</td></tr>
      </tbody>
    </table>
    </body></html>
    """
    mock_get.return_value = mock_response

    result = fetch_eurozone_eurostat_series(
        "prc_hicp_manr?geo=EA20&coicop=CP00&unit=RCH_A&freq=M",
        country="eurozone",
        frequency="monthly",
        source_hint="cpi",
    )
    assert pd.Timestamp(result["date"].max()) == pd.Timestamp("2026-02-01")
    assert result.iloc[-1]["source"] == "eurostat_flash"


def test_eurozone_canonical_series_id_mapping() -> None:
    """Eurozone aliases should map to canonical ids."""
    assert canonicalize_eurozone_series_id("pmi") == "growth_proxy"
    assert canonicalize_eurozone_series_id("growth_proxy") == "growth_proxy"


def test_eurozone_diagnostics_built_from_actual_loaded_frame(tmp_path: Path) -> None:
    """Eurozone diagnostics should reflect actually loaded normalized data."""
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-02-01"],
            "value": [2.7],
            "series_id": ["cpi"],
            "country": ["eurozone"],
            "source": ["eurostat"],
            "frequency": ["monthly"],
            "release_date": ["2026-02-15"],
            "ingested_at": ["2026-03-01"],
        }
    ).to_csv(normalized_dir / "cpi.csv", index=False)

    result = validate_eurozone_data(base_dir=str(normalized_dir))
    cpi_row = result["series_status"].loc[result["series_status"]["series_id"] == "cpi"].iloc[0]
    missing_row = result["series_status"].loc[result["series_status"]["series_id"] == "policy_rate"].iloc[0]

    assert cpi_row["source_used"] == "eurostat"
    assert int(cpi_row["row_count"]) == 1
    assert missing_row["source_used"] == "No loaded data"
    assert int(missing_row["row_count"]) == 0
    assert result["loaded_data_path"] == str(normalized_dir)
    assert result["actual_sources_found"] == ["eurostat"]


def test_eurozone_diagnostics_detect_optional_loaded_series(tmp_path: Path) -> None:
    """Eurozone diagnostics should recognize loaded enrichment series from normalized data."""
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir(parents=True)
    for series_id, source in [("m3", "ecb"), ("core_cpi", "eurostat")]:
        pd.DataFrame(
            {
                "date": ["2026-01-01"],
                "value": [1.0],
                "series_id": [series_id],
                "country": ["eurozone"],
                "source": [source],
                "frequency": ["monthly"],
                "release_date": ["2026-01-15"],
                "ingested_at": ["2026-03-01"],
            }
        ).to_csv(normalized_dir / f"{series_id}.csv", index=False)

    result = validate_eurozone_data(base_dir=str(normalized_dir))
    assert "m3" in result["available_series"]
    assert "core_cpi" in result["available_series"]
    assert sorted(result["actual_sources_found"]) == ["ecb", "eurostat"]


def test_fetch_eurozone_bundle_writes_normalized_files(tmp_path: Path) -> None:
    """Eurozone fetch should write normalized files for the minimum four series."""
    with patch("app.data.eurozone_ingestion.get_country_indicators") as mock_get_country_indicators:
        mock_get_country_indicators.return_value = [
            {
                "key": "cpi",
                "source": "eurozone_eurostat",
                "source_series_id": "prc_hicp_manr?geo=EA20&coicop=CP00&unit=RCH_A&freq=M",
                "frequency": "monthly",
            },
            {
                "key": "growth_proxy",
                "source": "eurozone_eurostat",
                "source_series_id": "sts_inpr_m?geo=EA20&s_adj=SA&nace_r2=B-D&unit=I15&freq=M",
                "frequency": "monthly",
            },
            {
                "key": "policy_rate",
                "source": "eurozone_ecb",
                "source_series_id": "FM/D.U2.EUR.4F.KR.DFR.LEV",
                "frequency": "monthly",
            },
            {
                "key": "yield_10y",
                "source": "eurozone_ecb",
                "source_series_id": "FM/M.U2.EUR.4F.BB.U2_10Y.YLD",
                "frequency": "monthly",
            },
        ]
        mock_eurostat_fetch = Mock()
        mock_ecb_fetch = Mock()
        eurostat_frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01"],
                "value": [2.8, 2.7],
                "series_id": ["raw_series", "raw_series"],
                "country": ["eurozone", "eurozone"],
                "source": ["eurostat", "eurostat"],
                "frequency": ["monthly", "monthly"],
                "release_date": ["2024-01-15", "2024-02-15"],
                "ingested_at": ["2026-03-20", "2026-03-20"],
            }
        )
        ecb_frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01"],
                "value": [4.0, 3.9],
                "series_id": ["raw_series", "raw_series"],
                "country": ["eurozone", "eurozone"],
                "source": ["ecb", "ecb"],
                "frequency": ["monthly", "monthly"],
                "release_date": ["2024-01-01", "2024-02-01"],
                "ingested_at": ["2026-03-20", "2026-03-20"],
            }
        )
        mock_eurostat_fetch.return_value = eurostat_frame
        mock_ecb_fetch.return_value = ecb_frame
        with patch.dict(
            EUROZONE_SOURCE_FETCHERS,
            {
                "eurozone_eurostat": (mock_eurostat_fetch, "eurostat"),
                "eurozone_ecb": (mock_ecb_fetch, "ecb"),
            },
            clear=False,
        ):
            summary = fetch_eurozone_api_bundle(base_dir=str(tmp_path))
    normalized_dir = tmp_path / "normalized"

    assert sorted(summary.loc[summary["status"] == "ready", "series_id"].tolist()) == [
        "cpi",
        "growth_proxy",
        "policy_rate",
        "yield_10y",
    ]
    assert (normalized_dir / "cpi.csv").exists()
    assert (normalized_dir / "growth_proxy.csv").exists()
    assert (normalized_dir / "policy_rate.csv").exists()
    assert (normalized_dir / "yield_10y.csv").exists()


@patch("main.fetch_eurozone_api_bundle")
def test_fetch_country_api_data_hard_fails_on_empty_eurozone_output(mock_fetch: Mock) -> None:
    """Eurozone CLI fetch should fail clearly if no normalized series are produced."""
    mock_fetch.return_value = pd.DataFrame(
        [{"series_id": "cpi", "status": "missing", "row_count": 0, "latest_date": pd.NaT, "required_for_minimum_regime": True}]
    )

    with pytest.raises(SystemExit, match="Eurozone API fetch produced no normalized minimum-series output"):
        run_fetch_country_api_data("eurozone")
