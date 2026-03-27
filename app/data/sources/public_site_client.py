"""Free public-site adapters for latest macro and market observations."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import re
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

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

PUBLIC_SITE_SERIES_CONFIG: dict[str, dict[str, Any]] = {
    "china_m2_pbc_public": {
        "url": "https://www.pbc.gov.cn/en/3688247/3688978/3709137/index.html",
        "parser": "pboc_financial_report_latest",
        "output_series_id": "m2",
        "source": "pboc",
        "frequency": "monthly",
        "report_title_pattern": r"Financial Statistics Report",
        "value_pattern": r"broad money supply \(M2\)[^.]{0,120}?(?:rising|rose|increasing|increased) by (?P<value>-?\d+(?:\.\d+)?)\s*percent",
        "period_pattern": r"Financial Statistics Report \((?P<period>[^)]+)\)",
    },
    "china_core_cpi_public": {
        "url": "https://tradingeconomics.com/china/core-inflation-rate",
        "parser": "tradingeconomics_latest",
        "output_series_id": "core_cpi",
        "source": "tradingeconomics",
        "frequency": "monthly",
        "regexes": [
            r"(?:core inflation(?: rate)?|core CPI)[^.]{0,160}?(?:to|at|of|was|stood at|came in at|rose to|eased to|accelerated to)\s+(?P<value>-?\d+(?:\.\d+)?)\s*(?:percent|%)?.*?(?:in|for)\s+(?P<period>[A-Za-z]+(?:\s+\d{1,2},)?\s+20\d{2})",
        ],
    },
    "china_unrate_public": {
        "url": "https://tradingeconomics.com/china/unemployment-rate",
        "parser": "tradingeconomics_latest",
        "output_series_id": "unrate",
        "source": "tradingeconomics",
        "frequency": "monthly",
        "regexes": [
            r"(?:unemployment rate|urban surveyed unemployment rate)[^.]{0,160}?(?:to|at|of|was|stood at|came in at|rose to|eased to)\s+(?P<value>-?\d+(?:\.\d+)?)\s*(?:percent|%)?.*?(?:in|for)\s+(?P<period>[A-Za-z]+(?:\s+\d{1,2},)?\s+20\d{2})",
        ],
    },
    "china_m2_public": {
        "url": "https://tradingeconomics.com/china/money-supply-m2",
        "parser": "tradingeconomics_latest",
        "output_series_id": "m2",
        "source": "tradingeconomics",
        "frequency": "monthly",
        "regexes": [
            r"(?:money supply M2|M2 money supply|M2)[^.]{0,200}?(?:grew|growth|rose|expanded|increased|was)\s+(?:by\s+)?(?P<value>-?\d+(?:\.\d+)?)\s*(?:percent|%)\s*(?:year-on-year|YoY)?[^.]{0,120}?(?:in|for)\s+(?P<period>[A-Za-z]+(?:\s+\d{1,2},)?\s+20\d{2})",
        ],
    },
    "china_industrial_production_public": {
        "url": "https://tradingeconomics.com/china/industrial-production",
        "parser": "tradingeconomics_latest",
        "output_series_id": "industrial_production",
        "source": "tradingeconomics",
        "frequency": "monthly",
        "regexes": [
            r"(?:industrial production|industrial output)[^.]{0,200}?(?:rose|increased|grew|expanded|was)\s+(?:by\s+)?(?P<value>-?\d+(?:\.\d+)?)\s*(?:percent|%)\s*(?:year-on-year|YoY)?[^.]{0,120}?(?:in|for)\s+(?P<period>[A-Za-z]+(?:\s*-\s*[A-Za-z]+)?\s+20\d{2}|[A-Za-z]+(?:\s+\d{1,2},)?\s+20\d{2})",
        ],
    },
    "eurozone_equity_pb_proxy_public": {
        "url": "https://ycharts.com/companies/FEZ",
        "parser": "ycharts_metric",
        "output_series_id": "equity_pb_proxy",
        "source": "ycharts",
        "frequency": "monthly",
        "metric_label": "Weighted Average Price to Book Ratio",
    },
    "china_equity_proxy_public": {
        "url": "https://tradingeconomics.com/china/stock-market",
        "parser": "tradingeconomics_latest",
        "output_series_id": "china_equity_proxy",
        "source": "tradingeconomics",
        "frequency": "daily",
        "regexes": [
            r"(?:stock market|shanghai composite|CSI 300|index)[^.]{0,120}?(?:to|at|closed at|traded at|fell to|rose to|was)\s+(?P<value>-?\d+(?:,\d{3})*(?:\.\d+)?)\s+points[^.]{0,120}?(?:on|in)\s+(?P<period>[A-Za-z]+(?:\s+\d{1,2},)?\s+20\d{2})",
        ],
    },
    "eurozone_equity_proxy_public": {
        "url": "https://tradingeconomics.com/euro-area/stock-market",
        "parser": "tradingeconomics_latest",
        "output_series_id": "eurostoxx50_proxy",
        "source": "tradingeconomics",
        "frequency": "daily",
        "regexes": [
            r"(?:stock market|euro stoxx|euro area index|index)[^.]{0,120}?(?:to|at|closed at|traded at|fell to|rose to|was)\s+(?P<value>-?\d+(?:,\d{3})*(?:\.\d+)?)\s+points[^.]{0,120}?(?:on|in)\s+(?P<period>[A-Za-z]+(?:\s+\d{1,2},)?\s+20\d{2})",
        ],
    },
    "gold_proxy_public": {
        "url": "https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart",
        "parser": "macrotrends_history",
        "output_series_id": "gold_proxy",
        "source": "macrotrends",
        "frequency": "daily",
    },
    "oil_proxy_public": {
        "url": "https://www.macrotrends.net/2516/wti-crude-oil-prices-10-year-daily-chart",
        "parser": "macrotrends_history",
        "output_series_id": "oil_proxy",
        "source": "macrotrends",
        "frequency": "daily",
    },
    "copper_proxy_public": {
        "url": "https://www.macrotrends.net/1476/copper-prices-historical-chart-data",
        "parser": "macrotrends_history",
        "output_series_id": "copper_proxy",
        "source": "macrotrends",
        "frequency": "daily",
    },
    "sp500_proxy_public": {
        "url": "https://stooq.com/q/d/l/?s=%5Espx&i=d",
        "parser": "stooq_csv",
        "output_series_id": "sp500_proxy",
        "source": "stooq",
        "frequency": "daily",
    },
}


def _ingested_at() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _normalize_html_text(html: str) -> str:
    """Collapse HTML into a regex-friendly plain-text string."""
    without_tags = re.sub(r"<[^>]+>", " ", html)
    without_entities = without_tags.replace("&nbsp;", " ").replace("&#37;", "%")
    return re.sub(r"\s+", " ", without_entities).strip()


def _parse_period_label(label: str, *, frequency: str) -> pd.Timestamp:
    """Parse a human-readable period label into a timestamp."""
    text = str(label or "").strip()
    text = re.sub(r"\s*-\s*", "-", text)
    monthly_candidates = [
        "%B %Y",
        "%b %Y",
    ]
    daily_candidates = [
        "%B %d, %Y",
        "%b %d, %Y",
    ]

    if "-" in text and frequency == "monthly":
        parts = text.split("-")
        text = f"{parts[-1].strip()}".replace("  ", " ")

    for fmt in daily_candidates + monthly_candidates:
        try:
            parsed = pd.Timestamp(datetime.strptime(text, fmt))
            if frequency == "monthly":
                return parsed.to_period("M").to_timestamp()
            return parsed.normalize()
        except ValueError:
            continue

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    parsed_ts = pd.Timestamp(parsed)
    if frequency == "monthly":
        return parsed_ts.to_period("M").to_timestamp()
    return parsed_ts.normalize()


def _build_single_observation(
    *,
    date: pd.Timestamp,
    release_date: pd.Timestamp,
    value: float,
    series_id: str,
    country: str,
    source: str,
    frequency: str,
) -> pd.DataFrame:
    """Build a single-row normalized observation frame."""
    return pd.DataFrame(
        {
            "date": [date],
            "value": [value],
            "series_id": [series_id],
            "country": [country],
            "source": [source],
            "frequency": [frequency],
            "release_date": [release_date],
            "ingested_at": [_ingested_at()],
        }
    ).loc[:, NORMALIZED_COLUMNS]


def _build_observation_frame(
    frame: pd.DataFrame,
    *,
    series_id: str,
    country: str,
    source: str,
    frequency: str,
) -> pd.DataFrame:
    """Normalize a multi-row observation frame into the shared schema."""
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    if normalized.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    normalized["series_id"] = series_id
    normalized["country"] = country
    normalized["source"] = source
    normalized["frequency"] = frequency
    normalized["release_date"] = normalized["date"]
    normalized["ingested_at"] = _ingested_at()
    return normalized.loc[:, NORMALIZED_COLUMNS]


def _parse_tradingeconomics_latest(html: str, config: dict[str, Any], country: str) -> pd.DataFrame:
    """Parse a latest observation from a public TradingEconomics page."""
    text = _normalize_html_text(html)
    for pattern in config["regexes"]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw_value = str(match.group("value")).replace(",", "")
        value = float(raw_value)
        release_date = _parse_period_label(str(match.group("period")), frequency=str(config["frequency"]))
        if pd.isna(release_date):
            continue
        return _build_single_observation(
            date=release_date if str(config["frequency"]) == "daily" else release_date.to_period("M").to_timestamp(),
            release_date=release_date,
            value=value,
            series_id=str(config["output_series_id"]),
            country=country,
            source=str(config["source"]),
            frequency=str(config["frequency"]),
        )
    return pd.DataFrame(columns=NORMALIZED_COLUMNS)


def _parse_ycharts_metric(html: str, config: dict[str, Any], country: str) -> pd.DataFrame:
    """Parse a latest ETF/fund metric from a public YCharts page."""
    text = _normalize_html_text(html)
    metric_label = str(config["metric_label"])
    match = re.search(
        rf"{re.escape(metric_label)}[^0-9-]*(?P<value>-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    value = float(match.group("value"))
    release_date = pd.Timestamp.utcnow().tz_localize(None).to_period("M").to_timestamp()
    return _build_single_observation(
        date=release_date,
        release_date=release_date,
        value=value,
        series_id=str(config["output_series_id"]),
        country=country,
        source=str(config["source"]),
        frequency=str(config["frequency"]),
    )


def _parse_macrotrends_history(html: str, config: dict[str, Any], country: str) -> pd.DataFrame:
    """Parse historical daily observations from a Macrotrends chart page."""
    rows = re.findall(
        r'date:\s*"(?P<date>\d{4}-\d{2}-\d{2})"[^}]*?close:\s*"?(?P<value>-?\d+(?:\.\d+)?)"?',
        html,
        flags=re.IGNORECASE,
    )
    if not rows:
        rows = re.findall(
            r'\[\s*"(?P<date>\d{4}-\d{2}-\d{2})"\s*,\s*"?(?P<value>-?\d+(?:\.\d+)?)"?\s*\]',
            html,
            flags=re.IGNORECASE,
        )
    frame = pd.DataFrame(rows, columns=["date", "value"])
    return _build_observation_frame(
        frame,
        series_id=str(config["output_series_id"]),
        country=country,
        source=str(config["source"]),
        frequency=str(config["frequency"]),
    )


def _parse_stooq_csv(csv_text: str, config: dict[str, Any], country: str) -> pd.DataFrame:
    """Parse a public Stooq CSV download into the shared schema."""
    frame = pd.read_csv(StringIO(csv_text))
    if {"Date", "Close"}.difference(frame.columns):
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    normalized = frame.rename(columns={"Date": "date", "Close": "value"}).loc[:, ["date", "value"]]
    return _build_observation_frame(
        normalized,
        series_id=str(config["output_series_id"]),
        country=country,
        source=str(config["source"]),
        frequency=str(config["frequency"]),
    )


def _parse_pboc_financial_report_latest(_html: str, config: dict[str, Any], country: str) -> pd.DataFrame:
    """Follow the latest PBOC financial statistics report and extract M2 growth."""
    index_response = requests.get(
        str(config["url"]),
        timeout=30,
        headers={"User-Agent": "global-macro-monitor/1.0"},
    )
    index_response.raise_for_status()
    index_html = index_response.text
    links = re.findall(
        rf'<a[^>]+href="([^"]+)"[^>]*>\s*(?P<title>{config["report_title_pattern"]}\s*\([^)]+\))\s*</a>',
        index_html,
        flags=re.IGNORECASE,
    )
    if not links:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    href, title = links[0]
    article_url = urljoin(str(config["url"]), href)
    article_response = requests.get(
        article_url,
        timeout=30,
        headers={"User-Agent": "global-macro-monitor/1.0"},
    )
    article_response.raise_for_status()
    text = _normalize_html_text(article_response.text)
    value_match = re.search(str(config["value_pattern"]), text, flags=re.IGNORECASE)
    period_match = re.search(str(config["period_pattern"]), title, flags=re.IGNORECASE)
    if not value_match or not period_match:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    value = float(value_match.group("value"))
    release_date = _parse_period_label(period_match.group("period"), frequency=str(config["frequency"]))
    if pd.isna(release_date):
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return _build_single_observation(
        date=release_date.to_period("M").to_timestamp(),
        release_date=release_date,
        value=value,
        series_id=str(config["output_series_id"]),
        country=country,
        source=str(config["source"]),
        frequency=str(config["frequency"]),
    )


def fetch_public_site_series(
    source_series_id: str,
    country: str,
    frequency: str,
    source_hint: str | None = None,
) -> pd.DataFrame:
    """Fetch one latest macro or market observation from a free public site."""
    del source_hint, frequency
    if source_series_id not in PUBLIC_SITE_SERIES_CONFIG:
        raise ValueError(f"Unsupported public-site series: {source_series_id}")

    config = PUBLIC_SITE_SERIES_CONFIG[source_series_id]
    response = requests.get(
        str(config["url"]),
        timeout=30,
        headers={"User-Agent": "global-macro-monitor/1.0"},
    )
    response.raise_for_status()
    html = response.text
    parser = str(config["parser"])

    if parser == "tradingeconomics_latest":
        return _parse_tradingeconomics_latest(html, config, country)
    if parser == "ycharts_metric":
        return _parse_ycharts_metric(html, config, country)
    if parser == "macrotrends_history":
        return _parse_macrotrends_history(html, config, country)
    if parser == "stooq_csv":
        return _parse_stooq_csv(html, config, country)
    if parser == "pboc_financial_report_latest":
        return _parse_pboc_financial_report_latest(html, config, country)
    raise ValueError(f"Unsupported public-site parser: {parser}")
