"""Persistent snapshot history and mode-specific comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.regime.global_monitor import COUNTRY_WEIGHTS

PREFERENCE_ORDER = {"cautious": 0, "neutral": 1, "bullish": 2}
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
SUMMARY_HISTORY_PATH = "data/processed/global_summary_history.csv"
ALLOCATION_HISTORY_PATH = "data/processed/global_allocation_history.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with normalized timestamp/date columns when available."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in ["date", "summary_date", "run_timestamp"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _schema_version(columns: list[str]) -> str:
    """Build a deterministic schema version from sorted column names."""
    return "|".join(sorted(columns))


def _append_history(frame: pd.DataFrame, history_path: str) -> pd.DataFrame:
    """Append a snapshot frame to a persistent history file."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = _load_csv(path)
    output = pd.concat([history, frame], ignore_index=True) if not history.empty else frame.copy()
    output.to_csv(path, index=False)
    return output


def append_global_summary_history(
    summary: pd.DataFrame,
    run_timestamp: pd.Timestamp,
    history_path: str = SUMMARY_HISTORY_PATH,
) -> pd.DataFrame:
    """Append the latest global summary snapshot to persistent history."""
    history_frame = summary.copy()
    history_frame["run_timestamp"] = run_timestamp
    history_frame["selected_mode"] = history_frame["as_of_mode"]
    history_frame["schema_version"] = _schema_version(history_frame.columns.tolist())
    return _append_history(history_frame, history_path)


def append_global_allocation_history(
    allocation: pd.DataFrame,
    run_timestamp: pd.Timestamp,
    history_path: str = ALLOCATION_HISTORY_PATH,
) -> pd.DataFrame:
    """Append the latest global allocation snapshot to persistent history."""
    history_frame = allocation.copy()
    history_frame["run_timestamp"] = run_timestamp
    history_frame["selected_mode"] = history_frame["as_of_mode"]
    history_frame["schema_version"] = _schema_version(history_frame.columns.tolist())
    return _append_history(history_frame, history_path)


def _comparison_reason(status: str) -> str:
    """Map comparison status to a user-facing explanation."""
    if status == "ready":
        return "Compared against the most recent prior comparable snapshot."
    if status == "schema_reset":
        return "Change history for this mode starts from the latest schema version."
    return "No prior snapshot is available yet for this mode."


def _direction(change_type: str, old_value: object, new_value: object) -> str:
    """Map ordered value changes into upgrade/downgrade style labels."""
    if change_type == "preference_change":
        return (
            "upgrade"
            if PREFERENCE_ORDER.get(str(new_value), 0) > PREFERENCE_ORDER.get(str(old_value), 0)
            else "downgrade"
        )
    if change_type == "confidence_change":
        return (
            "upgrade"
            if CONFIDENCE_ORDER.get(str(new_value), 0) > CONFIDENCE_ORDER.get(str(old_value), 0)
            else "downgrade"
        )
    return "changed"


def _history_for_mode(history: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return one mode's history sorted by timestamp."""
    if history.empty:
        return pd.DataFrame()
    matched = history.loc[history["as_of_mode"] == mode].copy()
    return matched.sort_values("run_timestamp").reset_index(drop=True)


def _latest_schema(history: pd.DataFrame) -> str | None:
    """Return the latest schema version present in a history frame."""
    if history.empty or "schema_version" not in history.columns:
        return None
    latest_timestamp = history["run_timestamp"].max()
    latest_rows = history.loc[history["run_timestamp"] == latest_timestamp]
    if latest_rows.empty:
        return None
    return str(latest_rows.iloc[-1]["schema_version"])


def _common_run_timestamps(
    summary_history: pd.DataFrame,
    allocation_history: pd.DataFrame,
    mode: str,
) -> tuple[list[pd.Timestamp], str]:
    """Find comparable common run timestamps for one mode."""
    summary_mode = _history_for_mode(summary_history, mode)
    allocation_mode = _history_for_mode(allocation_history, mode)
    if summary_mode.empty or allocation_mode.empty:
        return [], "no_prior_snapshot"

    summary_schema = _latest_schema(summary_mode)
    allocation_schema = _latest_schema(allocation_mode)
    comparable_summary = (
        summary_mode.loc[summary_mode["schema_version"] == summary_schema]
        if summary_schema is not None
        else pd.DataFrame()
    )
    comparable_allocation = (
        allocation_mode.loc[allocation_mode["schema_version"] == allocation_schema]
        if allocation_schema is not None
        else pd.DataFrame()
    )
    common = sorted(
        set(comparable_summary["run_timestamp"].tolist()).intersection(
            comparable_allocation["run_timestamp"].tolist()
        )
    )
    if len(common) >= 2:
        return common, "ready"
    if summary_schema is None or allocation_schema is None:
        return [], "no_prior_snapshot"
    if len(common) == 1:
        all_common = sorted(
            set(summary_mode["run_timestamp"].tolist()).intersection(
                allocation_mode["run_timestamp"].tolist()
            )
        )
        return common, "schema_reset" if len(all_common) > 1 else "no_prior_snapshot"
    return [], "schema_reset"


def _row_at_timestamp(history: pd.DataFrame, run_timestamp: pd.Timestamp) -> pd.Series | None:
    """Return a single summary row for a given run timestamp."""
    matched = history.loc[history["run_timestamp"] == run_timestamp]
    if matched.empty:
        return None
    return matched.iloc[-1]


def _snapshot_at_timestamp(history: pd.DataFrame, run_timestamp: pd.Timestamp) -> pd.DataFrame:
    """Return all allocation rows for a given run timestamp."""
    if history.empty:
        return pd.DataFrame()
    return history.loc[history["run_timestamp"] == run_timestamp].copy()


def _country_regime_changes(
    processed_dir: str,
    current_snapshot_timestamp: pd.Timestamp,
    prior_snapshot_timestamp: pd.Timestamp,
) -> list[dict[str, object]]:
    """Build country regime changes using summary-level country regime columns when available."""
    rows: list[dict[str, object]] = []
    summary_history = _load_csv(Path(processed_dir) / Path(SUMMARY_HISTORY_PATH).name)
    if summary_history.empty:
        return rows
    for mode in ["latest_available", "last_common_date"]:
        mode_history = _history_for_mode(summary_history, mode)
        current_row = _row_at_timestamp(mode_history, current_snapshot_timestamp)
        prior_row = _row_at_timestamp(mode_history, prior_snapshot_timestamp)
        if current_row is None or prior_row is None:
            continue
        for country in COUNTRY_WEIGHTS:
            column = f"{country}_regime"
            if column not in current_row.index or column not in prior_row.index:
                continue
            old_value = prior_row.get(column)
            new_value = current_row.get(column)
            if pd.isna(old_value) or pd.isna(new_value) or old_value == new_value:
                continue
            rows.append(
                {
                    "change_type": "regime_change",
                    "entity_type": "country_regime",
                    "entity_name": country,
                    "old_value": old_value,
                    "new_value": new_value,
                    "direction": "changed",
                    "reason": "regime transition",
                }
            )
    return rows


def _build_change_row(
    *,
    selected_mode: str,
    current_snapshot_timestamp: pd.Timestamp,
    prior_snapshot_timestamp: pd.Timestamp,
    summary_date: pd.Timestamp,
    change_type: str,
    entity_type: str,
    entity_name: str,
    old_value: object,
    new_value: object,
    reason: str,
) -> dict[str, object]:
    """Build one structured change row."""
    return {
        "selected_mode": selected_mode,
        "as_of_mode": selected_mode,
        "current_snapshot_timestamp": current_snapshot_timestamp,
        "prior_snapshot_timestamp": prior_snapshot_timestamp,
        "run_timestamp": current_snapshot_timestamp,
        "summary_date": summary_date,
        "date": summary_date,
        "change_type": change_type,
        "entity_type": entity_type,
        "entity_name": entity_name,
        "old_value": old_value,
        "new_value": new_value,
        "direction": _direction(change_type, old_value, new_value),
        "reason": reason,
    }


def _validate_comparison_result(result: dict[str, Any]) -> None:
    """Assert that a comparison result is internally consistent."""
    if not result["comparison_available"]:
        assert pd.isna(result["prior_snapshot_timestamp"])
        assert result["regime_change_count"] == 0
        assert result["preference_change_count"] == 0
        assert result["confidence_change_count"] == 0
        assert not result["regime_changes"]
        assert not result["preference_changes"]
        assert not result["confidence_changes"]
    if any(
        count > 0
        for count in [
            result["regime_change_count"],
            result["preference_change_count"],
            result["confidence_change_count"],
        ]
    ):
        assert result["comparison_available"] is True
        assert pd.notna(result["prior_snapshot_timestamp"])


def build_mode_comparison(
    selected_mode: str,
    processed_dir: str = "data/processed",
) -> dict[str, Any]:
    """Build one explicit comparison object for a selected mode from persistent histories."""
    summary_history = _load_csv(Path(processed_dir) / Path(SUMMARY_HISTORY_PATH).name)
    allocation_history = _load_csv(Path(processed_dir) / Path(ALLOCATION_HISTORY_PATH).name)
    common_timestamps, status = _common_run_timestamps(summary_history, allocation_history, selected_mode)

    if not common_timestamps:
        latest_available = sorted(
            set(_history_for_mode(summary_history, selected_mode)["run_timestamp"].tolist()).intersection(
                _history_for_mode(allocation_history, selected_mode)["run_timestamp"].tolist()
            )
        )
        current_snapshot_timestamp = latest_available[-1] if latest_available else pd.NaT
        result = {
            "selected_mode": selected_mode,
            "current_snapshot_timestamp": current_snapshot_timestamp,
            "prior_snapshot_timestamp": pd.NaT,
            "comparison_available": False,
            "comparison_reason": _comparison_reason(status),
            "regime_change_count": 0,
            "preference_change_count": 0,
            "confidence_change_count": 0,
            "regime_changes": [],
            "preference_changes": [],
            "confidence_changes": [],
            "why_it_changed": [],
        }
        _validate_comparison_result(result)
        return result

    current_snapshot_timestamp = common_timestamps[-1]
    prior_snapshot_timestamp = common_timestamps[-2] if len(common_timestamps) >= 2 else pd.NaT
    if pd.isna(prior_snapshot_timestamp):
        result = {
            "selected_mode": selected_mode,
            "current_snapshot_timestamp": current_snapshot_timestamp,
            "prior_snapshot_timestamp": pd.NaT,
            "comparison_available": False,
            "comparison_reason": _comparison_reason(status),
            "regime_change_count": 0,
            "preference_change_count": 0,
            "confidence_change_count": 0,
            "regime_changes": [],
            "preference_changes": [],
            "confidence_changes": [],
            "why_it_changed": [],
        }
        _validate_comparison_result(result)
        return result

    summary_mode = _history_for_mode(summary_history, selected_mode)
    allocation_mode = _history_for_mode(allocation_history, selected_mode)
    current_summary = _row_at_timestamp(summary_mode, current_snapshot_timestamp)
    prior_summary = _row_at_timestamp(summary_mode, prior_snapshot_timestamp)
    current_allocation = _snapshot_at_timestamp(allocation_mode, current_snapshot_timestamp)
    prior_allocation = _snapshot_at_timestamp(allocation_mode, prior_snapshot_timestamp)
    summary_date = (
        current_summary["summary_date"] if current_summary is not None else current_snapshot_timestamp
    )

    regime_changes: list[dict[str, object]] = []
    for entity_name, entity_type in [
        ("global_regime", "global_regime"),
        ("investment_clock", "investment_clock"),
    ]:
        old_value = prior_summary.get(entity_name) if prior_summary is not None else pd.NA
        new_value = current_summary.get(entity_name) if current_summary is not None else pd.NA
        if pd.isna(old_value) or pd.isna(new_value) or old_value == new_value:
            continue
        reason = "investment clock transition" if entity_name == "investment_clock" else "regime transition"
        regime_changes.append(
            _build_change_row(
                selected_mode=selected_mode,
                current_snapshot_timestamp=current_snapshot_timestamp,
                prior_snapshot_timestamp=prior_snapshot_timestamp,
                summary_date=summary_date,
                change_type="regime_change",
                entity_type=entity_type,
                entity_name=entity_name,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
            )
        )

    for row in _country_regime_changes(
        processed_dir=processed_dir,
        current_snapshot_timestamp=current_snapshot_timestamp,
        prior_snapshot_timestamp=prior_snapshot_timestamp,
    ):
        regime_changes.append(
            {
                **row,
                "selected_mode": selected_mode,
                "as_of_mode": selected_mode,
                "current_snapshot_timestamp": current_snapshot_timestamp,
                "prior_snapshot_timestamp": prior_snapshot_timestamp,
                "run_timestamp": current_snapshot_timestamp,
                "summary_date": summary_date,
                "date": summary_date,
            }
        )

    preference_changes: list[dict[str, object]] = []
    confidence_changes: list[dict[str, object]] = []
    for _, row in current_allocation.iterrows():
        previous = prior_allocation.loc[prior_allocation["asset"] == row["asset"]]
        if previous.empty:
            continue
        previous_row = previous.iloc[-1]
        if previous_row["preference"] != row["preference"]:
            preference_changes.append(
                _build_change_row(
                    selected_mode=selected_mode,
                    current_snapshot_timestamp=current_snapshot_timestamp,
                    prior_snapshot_timestamp=prior_snapshot_timestamp,
                    summary_date=summary_date,
                    change_type="preference_change",
                    entity_type="asset",
                    entity_name=row["asset"],
                    old_value=previous_row["preference"],
                    new_value=row["preference"],
                    reason="regime transition",
                )
            )
        if previous_row["confidence"] != row["confidence"]:
            confidence_changes.append(
                _build_change_row(
                    selected_mode=selected_mode,
                    current_snapshot_timestamp=current_snapshot_timestamp,
                    prior_snapshot_timestamp=prior_snapshot_timestamp,
                    summary_date=summary_date,
                    change_type="confidence_change",
                    entity_type="asset",
                    entity_name=row["asset"],
                    old_value=previous_row["confidence"],
                    new_value=row["confidence"],
                    reason=(
                        "stale country inputs"
                        if row["confidence"] == "low"
                        else "missing valuation inputs"
                    ),
                )
            )

    why_it_changed = []
    for frame in [regime_changes, preference_changes, confidence_changes]:
        for row in frame:
            reason = str(row["reason"])
            if reason and reason not in why_it_changed:
                why_it_changed.append(reason)

    result = {
        "selected_mode": selected_mode,
        "current_snapshot_timestamp": current_snapshot_timestamp,
        "prior_snapshot_timestamp": prior_snapshot_timestamp,
        "comparison_available": True,
        "comparison_reason": _comparison_reason("ready"),
        "regime_change_count": len(regime_changes),
        "preference_change_count": len(preference_changes),
        "confidence_change_count": len(confidence_changes),
        "regime_changes": regime_changes,
        "preference_changes": preference_changes,
        "confidence_changes": confidence_changes,
        "why_it_changed": why_it_changed,
    }
    _validate_comparison_result(result)
    return result


def build_global_change_log(processed_dir: str = "data/processed") -> pd.DataFrame:
    """Build and save a combined structured change log from comparison objects."""
    rows: list[dict[str, object]] = []
    for mode in ["latest_available", "last_common_date"]:
        comparison = build_mode_comparison(selected_mode=mode, processed_dir=processed_dir)
        rows.append(
            {
                "selected_mode": comparison["selected_mode"],
                "as_of_mode": comparison["selected_mode"],
                "current_snapshot_timestamp": comparison["current_snapshot_timestamp"],
                "prior_snapshot_timestamp": comparison["prior_snapshot_timestamp"],
                "run_timestamp": comparison["current_snapshot_timestamp"],
                "summary_date": pd.NaT,
                "date": pd.NaT,
                "change_type": "comparison_meta",
                "entity_type": "meta",
                "entity_name": "snapshot_comparison",
                "old_value": pd.NA,
                "new_value": pd.NA,
                "direction": pd.NA,
                "reason": comparison["comparison_reason"],
                "comparison_available": comparison["comparison_available"],
                "regime_change_count": comparison["regime_change_count"],
                "preference_change_count": comparison["preference_change_count"],
                "confidence_change_count": comparison["confidence_change_count"],
            }
        )
        for section in ["regime_changes", "preference_changes", "confidence_changes"]:
            for item in comparison[section]:
                rows.append(
                    {
                        **item,
                        "comparison_available": comparison["comparison_available"],
                        "regime_change_count": comparison["regime_change_count"],
                        "preference_change_count": comparison["preference_change_count"],
                        "confidence_change_count": comparison["confidence_change_count"],
                    }
                )

    output = pd.DataFrame(rows)
    output_path = Path(processed_dir) / "global_change_log.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output
