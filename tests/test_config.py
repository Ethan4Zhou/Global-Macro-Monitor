"""Tests for config-driven country loading."""

from __future__ import annotations

from app.utils.config import get_country_config, get_country_indicators, get_supported_countries


def test_supported_countries_include_expected_markets() -> None:
    """The config should expose the three supported countries."""
    countries = get_supported_countries()
    assert "us" in countries
    assert "china" in countries
    assert "eurozone" in countries


def test_country_indicator_groups_are_config_driven() -> None:
    """Each configured country should expose macro and valuation groups."""
    us_config = get_country_config("us")
    us_macro = get_country_indicators("us", "macro")
    china_macro = get_country_indicators("china", "macro")

    assert us_config["display_name"] == "United States"
    assert any(item["key"] == "cpi" for item in us_macro)
    assert any(item["source"] == "tushare" for item in china_macro)
