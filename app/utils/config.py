"""Helpers for loading config-driven country metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_country_configs(path: str = "configs/countries.yaml") -> dict[str, dict[str, str]]:
    """Load configured countries."""
    payload = load_yaml(path)
    return payload.get("countries", {})


def load_indicator_configs(path: str = "configs/indicators.yaml") -> dict[str, Any]:
    """Load configured indicators."""
    payload = load_yaml(path)
    return payload.get("countries", {})


def get_supported_countries() -> list[str]:
    """Return the list of supported countries."""
    return list(load_country_configs().keys())


def get_country_config(country: str) -> dict[str, str]:
    """Return metadata for one configured country."""
    countries = load_country_configs()
    if country not in countries:
        raise ValueError(f"Unsupported country: {country}")
    return countries[country]


def get_country_indicators(country: str, group: str) -> list[dict[str, Any]]:
    """Return the configured indicators for one country and group."""
    countries = load_indicator_configs()
    if country not in countries:
        raise ValueError(f"Unsupported country: {country}")
    return countries[country].get(group, [])
