"""Feature engineering for country-level monthly macro data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.manual_loader import load_manual_csv
from app.utils.config import get_country_indicators

FEATURE_COLUMNS = [
    "date",
    "country",
    "pmi_level",
    "pmi_diff_3m",
    "pmi_3m_avg",
    "cpi_level",
    "cpi_mom_pct_change",
    "cpi_yoy",
    "cpi_3m_avg",
    "core_cpi_level",
    "core_cpi_mom_pct_change",
    "core_cpi_yoy",
    "core_cpi_3m_avg",
    "unrate_level",
    "unrate_diff_3m",
    "unrate_3m_avg",
    "policy_rate_level",
    "policy_rate_diff_3m",
    "policy_rate_3m_avg",
    "yield_10y_level",
    "yield_10y_diff_3m",
    "yield_10y_3m_avg",
    "m2_level",
    "m2_mom_pct_change",
    "m2_yoy",
    "m2_3m_avg",
    "m3_level",
    "m3_mom_pct_change",
    "m3_yoy",
    "m3_3m_avg",
    "industrial_production_level",
    "industrial_production_yoy",
    "industrial_production_days_stale",
    "m2_days_stale",
    "m3_days_stale",
    "core_cpi_days_stale",
    "unrate_days_stale",
    "sentiment_level",
    "sentiment_diff_3m",
    "sentiment_3m_avg",
    "sentiment_days_stale",
    "fedfunds_level",
    "fedfunds_diff_3m",
    "fedfunds_3m_avg",
    "gs10_level",
    "gs10_diff_3m",
    "gs10_3m_avg",
]


def _empty_feature_frame() -> pd.DataFrame:
    """Return an empty feature frame with explicit output columns."""
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _load_series_file(
    country: str,
    indicator: dict[str, object],
    manual_base_dir: str = "data/raw/manual",
    fred_base_dir: str = "data/raw/fred",
    api_base_dir: str = "data/raw/api",
) -> pd.DataFrame | None:
    """Load a raw series file from FRED, API cache, or the manual directory."""
    source = str(indicator.get("source", ""))
    series_id = str(indicator.get("series_id") or indicator.get("key") or "")
    source_series_id = str(indicator.get("source_series_id") or series_id)
    if source == "fred":
        path = Path(fred_base_dir) / f"{source_series_id}.csv"
    elif source in {"china_akshare", "china_nbs", "china_rates", "imf", "eurozone_ecb", "eurozone_eurostat", "eurozone_oecd"}:
        path = Path(api_base_dir) / country / "normalized" / f"{series_id}.csv"
    elif source in {"ecb", "eurostat", "oecd", "tushare"}:
        path = Path(api_base_dir) / country / f"{series_id}.csv"
    elif source == "manual":
        path = Path(manual_base_dir) / country / f"{series_id}.csv"
    else:
        return None

    if not path.exists():
        return None

    if source == "manual":
        frame = load_manual_csv(path)
    else:
        frame = pd.read_csv(path)
        required_columns = {"date", "value"}
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        keep_columns = [column for column in ["date", "value", "series_id"] if column in frame.columns]
        frame = frame.loc[:, keep_columns].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return frame.loc[:, ["date", "value"]]


def load_country_raw_series(
    country: str,
    manual_base_dir: str = "data/raw/manual",
    fred_base_dir: str = "data/raw/fred",
    api_base_dir: str = "data/raw/api",
) -> dict[str, pd.DataFrame]:
    """Load configured raw macro series for one country."""
    series_map: dict[str, pd.DataFrame] = {}
    for indicator in get_country_indicators(country, "macro"):
        series = _load_series_file(country=country, indicator=indicator, manual_base_dir=manual_base_dir, fred_base_dir=fred_base_dir, api_base_dir=api_base_dir)
        if series is None and indicator.get("fallback_source"):
            fallback_indicator = {**indicator, "source": indicator["fallback_source"]}
            series = _load_series_file(
                country=country,
                indicator=fallback_indicator,
                manual_base_dir=manual_base_dir,
                fred_base_dir=fred_base_dir,
                api_base_dir=api_base_dir,
            )
        if series is None and str(indicator.get("source", "")) != "manual":
            manual_indicator = {**indicator, "source": "manual"}
            if country == "eurozone" and str(indicator["key"]) == "growth_proxy":
                manual_indicator["series_id"] = "pmi"
            series = _load_series_file(
                country=country,
                indicator=manual_indicator,
                manual_base_dir=manual_base_dir,
                fred_base_dir=fred_base_dir,
                api_base_dir=api_base_dir,
            )
        if series is None:
            continue
        series_map[str(indicator["key"])] = series.loc[:, ["date", "value"]].rename(
            columns={"value": str(indicator["key"])}
        )
    return series_map


def align_monthly_panel(series_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align raw series into a monthly panel and carry forward latest monthly values."""
    panel: pd.DataFrame | None = None
    for key, frame in series_map.items():
        aligned = frame.copy()
        aligned[f"{key}__observed_date"] = aligned["date"]
        panel = aligned if panel is None else panel.merge(aligned, on="date", how="outer")
    if panel is None:
        return pd.DataFrame(columns=["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    panel = panel.groupby("date", as_index=False).last()
    full_range = pd.date_range(panel["date"].min(), panel["date"].max(), freq="MS")
    panel = panel.set_index("date").reindex(full_range).rename_axis("date").reset_index()
    value_columns = [column for column in panel.columns if column != "date"]
    if value_columns:
        panel[value_columns] = panel[value_columns].ffill()
    return panel


def _pct_change(series: pd.Series) -> pd.Series:
    """Compute one-period percentage change in percent units."""
    return series.pct_change(periods=1, fill_method=None) * 100.0


def _pct_change_12m(series: pd.Series) -> pd.Series:
    """Compute 12-month percentage change in percent units."""
    return series.pct_change(periods=12, fill_method=None) * 100.0


def _empty_series(index: pd.Index) -> pd.Series:
    """Create an all-NaN float series."""
    return pd.Series(float("nan"), index=index, dtype="float64")


def build_country_macro_features_frame(panel: pd.DataFrame, country: str) -> pd.DataFrame:
    """Compute standardized macro features for one country."""
    if panel.empty:
        return _empty_feature_frame()

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(panel["date"], errors="coerce"),
            "country": country,
        }
    )
    index = features.index

    pmi = panel["pmi"] if "pmi" in panel.columns else _empty_series(index)
    if "growth_proxy" in panel.columns and pmi.isna().all():
        pmi = panel["growth_proxy"]
    cpi = panel["cpi"] if "cpi" in panel.columns else _empty_series(index)
    core_cpi = panel["core_cpi"] if "core_cpi" in panel.columns else _empty_series(index)
    unrate = panel["unrate"] if "unrate" in panel.columns else _empty_series(index)
    policy_rate = panel["policy_rate"] if "policy_rate" in panel.columns else _empty_series(index)
    yield_10y = panel["yield_10y"] if "yield_10y" in panel.columns else _empty_series(index)
    m2 = panel["m2"] if "m2" in panel.columns else _empty_series(index)
    m3 = panel["m3"] if "m3" in panel.columns else _empty_series(index)
    industrial_production = panel["industrial_production"] if "industrial_production" in panel.columns else _empty_series(index)
    sentiment = panel["sentiment"] if "sentiment" in panel.columns else _empty_series(index)
    industrial_obs = pd.to_datetime(panel["industrial_production__observed_date"], errors="coerce") if "industrial_production__observed_date" in panel.columns else pd.Series(pd.NaT, index=index)
    m2_obs = pd.to_datetime(panel["m2__observed_date"], errors="coerce") if "m2__observed_date" in panel.columns else pd.Series(pd.NaT, index=index)
    m3_obs = pd.to_datetime(panel["m3__observed_date"], errors="coerce") if "m3__observed_date" in panel.columns else pd.Series(pd.NaT, index=index)
    core_cpi_obs = pd.to_datetime(panel["core_cpi__observed_date"], errors="coerce") if "core_cpi__observed_date" in panel.columns else pd.Series(pd.NaT, index=index)
    unrate_obs = pd.to_datetime(panel["unrate__observed_date"], errors="coerce") if "unrate__observed_date" in panel.columns else pd.Series(pd.NaT, index=index)
    sentiment_obs = pd.to_datetime(panel["sentiment__observed_date"], errors="coerce") if "sentiment__observed_date" in panel.columns else pd.Series(pd.NaT, index=index)

    features["pmi_level"] = pmi
    features["pmi_diff_3m"] = pmi - pmi.shift(3)
    features["pmi_3m_avg"] = pmi.rolling(window=3).mean()

    features["cpi_level"] = cpi
    features["cpi_mom_pct_change"] = _pct_change(cpi) if country != "china" else _empty_series(index)
    features["cpi_yoy"] = _pct_change_12m(cpi) if country != "china" else cpi
    features["cpi_3m_avg"] = cpi.rolling(window=3).mean()

    features["core_cpi_level"] = core_cpi
    features["core_cpi_mom_pct_change"] = _pct_change(core_cpi)
    features["core_cpi_yoy"] = _pct_change_12m(core_cpi)
    features["core_cpi_3m_avg"] = core_cpi.rolling(window=3).mean()

    features["unrate_level"] = unrate
    features["unrate_diff_3m"] = unrate - unrate.shift(3)
    features["unrate_3m_avg"] = unrate.rolling(window=3).mean()

    features["policy_rate_level"] = policy_rate
    features["policy_rate_diff_3m"] = policy_rate - policy_rate.shift(3)
    features["policy_rate_3m_avg"] = policy_rate.rolling(window=3).mean()

    features["yield_10y_level"] = yield_10y
    features["yield_10y_diff_3m"] = yield_10y - yield_10y.shift(3)
    features["yield_10y_3m_avg"] = yield_10y.rolling(window=3).mean()

    features["m2_level"] = m2
    features["m2_mom_pct_change"] = _pct_change(m2) if country != "china" else _empty_series(index)
    features["m2_yoy"] = _pct_change_12m(m2) if country != "china" else m2
    features["m2_3m_avg"] = m2.rolling(window=3).mean()
    features["m3_level"] = m3
    features["m3_mom_pct_change"] = _pct_change(m3)
    features["m3_yoy"] = _pct_change_12m(m3)
    features["m3_3m_avg"] = m3.rolling(window=3).mean()

    features["industrial_production_level"] = industrial_production
    features["industrial_production_yoy"] = (
        _pct_change_12m(industrial_production) if country != "china" else industrial_production
    )
    feature_dates = pd.to_datetime(features["date"], errors="coerce")
    features["industrial_production_days_stale"] = (feature_dates - industrial_obs).dt.days
    features["m2_days_stale"] = (feature_dates - m2_obs).dt.days
    features["m3_days_stale"] = (feature_dates - m3_obs).dt.days
    features["core_cpi_days_stale"] = (feature_dates - core_cpi_obs).dt.days
    features["unrate_days_stale"] = (feature_dates - unrate_obs).dt.days
    features["sentiment_level"] = sentiment
    features["sentiment_diff_3m"] = sentiment - sentiment.shift(3)
    features["sentiment_3m_avg"] = sentiment.rolling(window=3).mean()
    features["sentiment_days_stale"] = (feature_dates - sentiment_obs).dt.days

    # Legacy US aliases kept for backward compatibility.
    features["fedfunds_level"] = features["policy_rate_level"]
    features["fedfunds_diff_3m"] = features["policy_rate_diff_3m"]
    features["fedfunds_3m_avg"] = features["policy_rate_3m_avg"]
    features["gs10_level"] = features["yield_10y_level"]
    features["gs10_diff_3m"] = features["yield_10y_diff_3m"]
    features["gs10_3m_avg"] = features["yield_10y_3m_avg"]

    return features.loc[:, FEATURE_COLUMNS]


def save_country_macro_features(
    frame: pd.DataFrame,
    country: str,
    output_path: str | None = None,
) -> Path:
    """Save the country macro feature panel to CSV."""
    destination = Path(output_path or f"data/processed/{country}_macro_features.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def build_country_macro_features(
    country: str,
    manual_base_dir: str = "data/raw/manual",
    fred_base_dir: str = "data/raw/fred",
    api_base_dir: str = "data/raw/api",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Load configured raw series, compute features, and save them for one country."""
    series_map = load_country_raw_series(
        country=country,
        manual_base_dir=manual_base_dir,
        fred_base_dir=fred_base_dir,
        api_base_dir=api_base_dir,
    )
    panel = align_monthly_panel(series_map)
    features = build_country_macro_features_frame(panel=panel, country=country)
    save_country_macro_features(features, country=country, output_path=output_path)
    return features


def build_us_macro_features(
    input_dir: str = "data/raw/fred",
    output_path: str = "data/processed/us_macro_features.csv",
) -> pd.DataFrame:
    """Backward-compatible wrapper for the US feature pipeline."""
    return build_country_macro_features(
        country="us",
        fred_base_dir=input_dir,
        output_path=output_path,
    )
