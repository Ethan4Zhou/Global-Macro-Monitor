"""AkShare-first China macro and rates adapter."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
from typing import Any

import pandas as pd

NORMALIZED_COLUMNS = [
    "date",
    "value",
    "series_id",
    "country",
    "source",
    "frequency",
    "release_date",
    "ingested_at",
]

AKSHARE_SERIES_CONFIG: dict[str, dict[str, Any]] = {
    "cpi": {
        "function": "macro_china_cpi",
        "date_column": "月份",
        "value_column": "全国-同比增长",
    },
    "pmi": {
        "function": "macro_china_pmi",
        "date_column": "月份",
        "value_column": "制造业-指数",
    },
    "industrial_production": {
        "function": "macro_china_industrial_production_yoy",
        "date_column": "日期",
        "value_column": "今值",
    },
    "m2": {
        "function": "macro_china_m2_yearly",
        "date_column": "日期",
        "value_column": "今值",
    },
    "unrate": {
        "function": "macro_china_urban_unemployment",
        "custom_handler": "urban_unemployment",
    },
    "policy_rate": {
        "function": "repo_rate_hist",
        "custom_handler": "repo_fr007",
    },
    "yield_10y": {
        "function": "bond_china_yield",
        "custom_handler": "bond_10y",
    },
    "hs300_pe_proxy": {
        "function": "stock_index_pe_lg",
        "custom_handler": "hs300_pe",
    },
    "hs300_pb_proxy": {
        "function": "stock_index_pb_lg",
        "custom_handler": "hs300_pb",
    },
}


def _ingested_at() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _load_akshare() -> Any:
    """Import AkShare lazily so tests can mock it cleanly."""
    try:
        return importlib.import_module("akshare")
    except ModuleNotFoundError as exc:
        raise RuntimeError("AkShare is not installed. Run `pip install akshare`.") from exc


def _normalize_monthly_frame(
    frame: pd.DataFrame,
    *,
    series_id: str,
    date_column: str,
    value_column: str,
    country: str,
) -> pd.DataFrame:
    """Normalize an AkShare frame into the shared monthly schema."""
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    if date_column not in frame.columns or value_column not in frame.columns:
        raise ValueError(f"AkShare response missing {date_column} or {value_column}.")

    normalized = frame.loc[:, [date_column, value_column]].copy()
    normalized = normalized.rename(columns={date_column: "release_date", value_column: "value"})
    normalized["release_date"] = pd.to_datetime(normalized["release_date"], errors="coerce")
    normalized["date"] = normalized["release_date"].dt.to_period("M").dt.to_timestamp()
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized["series_id"] = series_id
    normalized["country"] = country
    normalized["source"] = "china_akshare"
    normalized["frequency"] = "monthly"
    normalized["ingested_at"] = _ingested_at()
    normalized = normalized.dropna(subset=["date", "value"])
    normalized = normalized.sort_values(["date", "release_date"]).drop_duplicates(
        subset=["series_id", "date"], keep="last"
    )
    return normalized.loc[:, NORMALIZED_COLUMNS].reset_index(drop=True)


def _normalize_china_month_label(value: object) -> pd.Timestamp:
    """Normalize Chinese month labels such as 2026年02月份 into month-start timestamps."""
    text = str(value).strip().replace("年", "-").replace("月份", "").replace("月", "")
    return pd.to_datetime(f"{text}-01", errors="coerce")


def _fetch_repo_fr007(ak: Any, country: str) -> pd.DataFrame:
    """Fetch FR007 repo history over rolling monthly windows and normalize it."""
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    chunks: list[pd.DataFrame] = []
    for offset in range(12):
        period_end = (end - pd.DateOffset(months=offset)).normalize()
        period_start = period_end.replace(day=1)
        frame = ak.repo_rate_hist(
            start_date=period_start.strftime("%Y%m%d"),
            end_date=period_end.strftime("%Y%m%d"),
        )
        chunks.append(frame)
    combined = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    return _normalize_monthly_frame(
        frame=combined.rename(columns={"date": "release_date", "FR007": "value"}),
        series_id="policy_rate",
        date_column="release_date",
        value_column="value",
        country=country,
    )


def _fetch_bond_10y(ak: Any, country: str) -> pd.DataFrame:
    """Fetch the China government 10Y yield curve and normalize it."""
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.DateOffset(months=12)
    frame = ak.bond_china_yield(
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
    if "曲线名称" in frame.columns:
        frame = frame.loc[frame["曲线名称"].astype(str).str.contains("国债", na=False)].copy()
    return _normalize_monthly_frame(
        frame=frame.rename(columns={"日期": "release_date", "10年": "value"}),
        series_id="yield_10y",
        date_column="release_date",
        value_column="value",
        country=country,
    )


def _fetch_urban_unemployment(ak: Any, country: str) -> pd.DataFrame:
    """Fetch China urban unemployment and normalize it to unrate."""
    frame = ak.macro_china_urban_unemployment()
    if "item" in frame.columns:
        frame = frame.loc[frame["item"].astype(str).str.contains("失业率", na=False)].copy()
    return _normalize_monthly_frame(
        frame=frame.rename(columns={"date": "release_date", "value": "value"}),
        series_id="unrate",
        date_column="release_date",
        value_column="value",
        country=country,
    )


def _first_available_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first available column from a candidate list."""
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"AkShare valuation response missing expected columns: {candidates}")


def _fetch_hs300_pe(ak: Any, country: str) -> pd.DataFrame:
    """Fetch HS300 PE proxy and normalize it."""
    frame = ak.stock_index_pe_lg(symbol="沪深300")
    date_column = _first_available_column(frame, ["日期", "date"])
    value_column = _first_available_column(
        frame,
        ["市盈率", "滚动市盈率", "平均市盈率", "等权平均市盈率", "pe"],
    )
    return _normalize_monthly_frame(
        frame=frame,
        series_id="hs300_pe_proxy",
        date_column=date_column,
        value_column=value_column,
        country=country,
    )


def _fetch_hs300_pb(ak: Any, country: str) -> pd.DataFrame:
    """Fetch HS300 PB proxy and normalize it."""
    frame = ak.stock_index_pb_lg(symbol="沪深300")
    date_column = _first_available_column(frame, ["日期", "date"])
    value_column = _first_available_column(
        frame,
        ["市净率", "平均市净率", "等权平均市净率", "pb"],
    )
    return _normalize_monthly_frame(
        frame=frame,
        series_id="hs300_pb_proxy",
        date_column=date_column,
        value_column=value_column,
        country=country,
    )


def fetch_china_akshare_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
) -> pd.DataFrame:
    """Fetch one canonical China series from AkShare."""
    del frequency, source_hint
    if source_series_id not in AKSHARE_SERIES_CONFIG:
        raise ValueError(f"Unsupported AkShare China series: {source_series_id}")

    ak = _load_akshare()
    config = AKSHARE_SERIES_CONFIG[source_series_id]
    if config.get("custom_handler") == "repo_fr007":
        return _fetch_repo_fr007(ak, country=country)
    if config.get("custom_handler") == "bond_10y":
        return _fetch_bond_10y(ak, country=country)
    if config.get("custom_handler") == "urban_unemployment":
        return _fetch_urban_unemployment(ak, country=country)
    if config.get("custom_handler") == "hs300_pe":
        return _fetch_hs300_pe(ak, country=country)
    if config.get("custom_handler") == "hs300_pb":
        return _fetch_hs300_pb(ak, country=country)

    function_name = str(config["function"])
    function = getattr(ak, function_name)
    frame = function()
    if str(config["date_column"]) == "月份":
        frame = frame.copy()
        frame["月份"] = frame["月份"].map(_normalize_china_month_label)
        frame = frame.rename(columns={"月份": "release_date", str(config["value_column"]): "value"})
        return _normalize_monthly_frame(
            frame=frame,
            series_id=source_series_id,
            date_column="release_date",
            value_column="value",
            country=country,
        )
    return _normalize_monthly_frame(
        frame=frame,
        series_id=source_series_id,
        date_column=str(config["date_column"]),
        value_column=str(config["value_column"]),
        country=country,
    )
