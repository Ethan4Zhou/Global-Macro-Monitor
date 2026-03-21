"""High-value alert generation for the macro monitor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.change_detection import HISTORY_DIR, build_mode_comparison
from app.regime.global_monitor import build_country_status
from app.regime.nowcast import build_global_nowcast_overlay
from app.utils.config import get_supported_countries

ALERTS_PATH = Path("data/processed/monitor_alerts.csv")
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in ["date", "summary_date", "snapshot_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _alert_row(
    *,
    selected_mode: str,
    date: pd.Timestamp | object,
    severity: str,
    alert_type: str,
    region: str,
    entity_name: str,
    old_value: object = pd.NA,
    new_value: object = pd.NA,
    metric_value: object = pd.NA,
    reason: str = "",
) -> dict[str, object]:
    """Build one structured alert row."""
    return {
        "selected_mode": selected_mode,
        "date": pd.to_datetime(date, errors="coerce"),
        "severity": severity,
        "alert_type": alert_type,
        "region": region,
        "entity_name": entity_name,
        "old_value": old_value,
        "new_value": new_value,
        "metric_value": metric_value,
        "reason": reason,
    }


def _latest_region_deviation(consensus_deviation: pd.DataFrame, region: str) -> pd.Series | None:
    """Return the latest consensus deviation row for one region."""
    if consensus_deviation.empty:
        return None
    matched = consensus_deviation.loc[consensus_deviation["region"] == region].copy()
    if matched.empty:
        return None
    if "snapshot_date" in matched.columns:
        matched = matched.sort_values("snapshot_date")
    return matched.iloc[-1]


def build_monitor_alerts(processed_dir: str = "data/processed") -> pd.DataFrame:
    """Build a compact alert table from high-value monitor changes."""
    processed_root = Path(processed_dir)
    summary = _load_csv(processed_root / "global_macro_summary.csv")
    consensus_deviation = _load_csv(processed_root / "consensus_deviation.csv")

    if summary.empty:
        output = pd.DataFrame(
            columns=[
                "selected_mode",
                "date",
                "severity",
                "alert_type",
                "region",
                "entity_name",
                "old_value",
                "new_value",
                "metric_value",
                "reason",
            ]
        )
        ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(ALERTS_PATH, index=False)
        return output

    rows: list[dict[str, object]] = []
    supported_countries = get_supported_countries()

    for mode in ["latest_available", "last_common_date"]:
        mode_summary = summary.loc[summary["as_of_mode"] == mode].copy()
        if mode_summary.empty:
            continue
        latest = mode_summary.sort_values("summary_date").iloc[-1]
        summary_date = latest.get("summary_date")

        if str(latest.get("global_regime")) == "partial_view" or float(latest.get("coverage_ratio", 0.0)) < 0.7:
            rows.append(
                _alert_row(
                    selected_mode=mode,
                    date=summary_date,
                    severity="high",
                    alert_type="partial_coverage",
                    region="global",
                    entity_name="global_regime",
                    metric_value=latest.get("coverage_ratio"),
                    reason=str(latest.get("coverage_warning", "")),
                )
            )

        comparison = build_mode_comparison(
            selected_mode=mode,
            processed_dir=processed_dir,
            history_dir=str(HISTORY_DIR),
        )
        if comparison["comparison_available"]:
            for item in comparison["regime_changes"]:
                if item["entity_name"] not in {"global_regime", "investment_clock"}:
                    continue
                rows.append(
                    _alert_row(
                        selected_mode=mode,
                        date=item.get("summary_date", summary_date),
                        severity="high" if item["entity_name"] == "global_regime" else "medium",
                        alert_type=item["entity_name"],
                        region="global",
                        entity_name=item["entity_name"],
                        old_value=item.get("old_value"),
                        new_value=item.get("new_value"),
                        reason=str(item.get("reason", "")),
                    )
                )

            for item in comparison["confidence_changes"]:
                if str(item.get("direction")) != "downgrade":
                    continue
                rows.append(
                    _alert_row(
                        selected_mode=mode,
                        date=item.get("summary_date", summary_date),
                        severity="medium",
                        alert_type="confidence_downgrade",
                        region="global",
                        entity_name=str(item.get("entity_name")),
                        old_value=item.get("old_value"),
                        new_value=item.get("new_value"),
                        reason=str(item.get("reason", "")),
                    )
                )

        status_table = build_country_status(mode=mode)
        if not status_table.empty:
            for _, row in status_table.iterrows():
                if str(row.get("staleness_status")) == "very_stale":
                    rows.append(
                        _alert_row(
                            selected_mode=mode,
                            date=summary_date,
                            severity="medium",
                            alert_type="very_stale_country",
                            region=str(row.get("country")),
                            entity_name=str(row.get("country")),
                            metric_value=row.get("days_stale"),
                            reason="very stale country data",
                        )
                    )
                if bool(row.get("country_ready")) and not bool(row.get("globally_usable_latest")):
                    rows.append(
                        _alert_row(
                            selected_mode=mode,
                            date=summary_date,
                            severity="low",
                            alert_type="country_not_usable",
                            region=str(row.get("country")),
                            entity_name=str(row.get("country")),
                            reason="ready locally but excluded from the selected global mode",
                        )
                    )

        country_regime_dates = {
            country: latest.get(f"{country}_latest_date") for country in supported_countries
        }
        overlay = build_global_nowcast_overlay(latest.get("summary_date"), country_regime_dates)
        overlay_score = float(overlay.get("overlay_score", 0.0))
        if overlay.get("overlay_direction") != "neutral" and abs(overlay_score) >= 0.5:
            rows.append(
                _alert_row(
                    selected_mode=mode,
                    date=overlay.get("freshest_market_date", summary_date),
                    severity="medium",
                    alert_type="nowcast_shift",
                    region="global",
                    entity_name=str(overlay.get("overlay_direction")),
                    metric_value=overlay_score,
                    reason=", ".join(str(item) for item in overlay.get("overlay_drivers", [])[:3]),
                )
            )

        for region in supported_countries:
            latest_region = _latest_region_deviation(consensus_deviation, region)
            if latest_region is None:
                continue
            deviation = latest_region.get("consensus_deviation_score")
            if pd.isna(deviation) or abs(float(deviation)) < 0.5:
                continue
            rows.append(
                _alert_row(
                    selected_mode=mode,
                    date=latest_region.get("snapshot_date", summary_date),
                    severity="medium",
                    alert_type="consensus_gap",
                    region=region,
                    entity_name=region,
                    metric_value=float(deviation),
                    reason=str(latest_region.get("deviation_summary", "")),
                )
            )

    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(
            by=["date", "severity", "region", "alert_type"],
            ascending=[False, True, True, True],
            key=lambda col: col.map(SEVERITY_ORDER) if col.name == "severity" else col,
        ).reset_index(drop=True)

    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(ALERTS_PATH, index=False)
    return output
