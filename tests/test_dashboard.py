"""Tests for dashboard-friendly formatting helpers."""

from __future__ import annotations

import pandas as pd

from app.dashboard.app import (
    ACTIVE_LANGUAGE,
    build_user_manual_markdown,
    comparison_message,
    format_change_sentence,
    format_country_list,
    format_display_value,
    format_entity_name,
    humanize_country_asset_label,
    humanize_global_asset_label,
    humanize_label,
    prepare_what_changed_sections,
    set_display_language,
    summarize_change_reasons,
    translate_runtime_text,
)


def test_format_display_value_shows_no_data_for_missing_values() -> None:
    """Dashboard formatting should make missing values explicit."""
    set_display_language("en")
    assert format_display_value(None) == "No data"
    assert format_display_value(float("nan")) == "No data"
    assert format_display_value("") == "No data"
    assert format_display_value("goldilocks") == "goldilocks"


def test_humanize_label_and_country_lists() -> None:
    """Display helpers should render internal labels nicely."""
    set_display_language("en")
    assert humanize_label("partial_view") == "Partial view"
    assert humanize_label("goldilocks") == "Goldilocks"
    assert humanize_label("bullish") == "Bullish"
    assert humanize_label("high") == "High"
    assert humanize_label("country_ready") == "Country Ready"
    assert humanize_label("globally_usable_latest") == "Usable in Selected Mode"
    assert format_entity_name("global_equities") == "Global Equities"
    assert format_entity_name("us_equities") == "US Equities"
    assert format_country_list("china,eurozone") == "China, Eurozone"
    assert humanize_label("very_stale") == "Very stale"


def test_change_sentence_and_reason_summary_helpers() -> None:
    """Change formatting helpers should stay readable and deduplicated."""
    set_display_language("en")
    assert (
        format_change_sentence("global_equities", "confidence", "medium", "low")
        == "Global Equities confidence changed from Medium to Low."
    )
    reasons = summarize_change_reasons(
        pd.DataFrame({"reason": ["stale country inputs", "stale country inputs", "missing valuation inputs"]})
    )
    assert reasons == ["stale country inputs", "missing valuation inputs"]


def test_prepare_what_changed_sections_uses_only_comparison_object() -> None:
    """Dashboard section prep should rely only on the provided comparison object."""
    set_display_language("en")
    comparison = {
        "comparison_available": False,
        "comparison_reason": "No prior snapshot is available yet for this mode.",
        "current_snapshot_timestamp": pd.NaT,
        "prior_snapshot_timestamp": pd.NaT,
        "regime_changes": [],
        "preference_changes": [],
        "confidence_changes": [],
        "why_it_changed": [],
        "regime_change_count": 0,
        "preference_change_count": 0,
        "confidence_change_count": 0,
    }
    sections = prepare_what_changed_sections(comparison)
    assert sections["available"] is False
    assert sections["message"] == "No prior snapshot is available yet for this mode."
    assert sections["regime_changes"].empty
    assert sections["preference_changes"].empty
    assert sections["confidence_changes"].empty


def test_comparison_message_handles_missing_meta() -> None:
    """Comparison message helper should explain missing prior snapshots."""
    set_display_language("en")
    assert comparison_message(None) == "No prior snapshot is available yet for this mode."


def test_dashboard_labels_support_chinese_display() -> None:
    """Key dashboard labels should localize cleanly in Chinese mode."""
    assert ACTIVE_LANGUAGE == "zh"
    set_display_language("zh")
    try:
        assert humanize_label("goldilocks") == "温和增长"
        assert humanize_label("fresh") == "最新"
        assert humanize_label("commodities") == "大宗商品"
        assert humanize_label("bullish") == "超配"
        assert humanize_label("cautious") == "低配"
        assert humanize_label("cheap") == "低估"
        assert humanize_label("policy_rate") == "政策利率"
        assert format_country_list("china,eurozone") == "中国, 欧元区"
        assert format_display_value(None) == "无数据"
        assert humanize_global_asset_label("duration") == "美元久期"
        assert humanize_country_asset_label("china", "dollar") == "美元"
        assert humanize_country_asset_label("eurozone", "dollar") == "美元"
    finally:
        set_display_language("en")


def test_runtime_reason_text_supports_chinese_display() -> None:
    """Dynamic rationale text should localize in Chinese mode."""
    set_display_language("zh")
    try:
        assert (
            translate_runtime_text(
                "Goldilocks and neutral liquidity still support global equities. Valuations look fair."
            )
            == "温和增长阶段与中性流动性环境仍支撑全球权益资产，估值大体合理。"
        )
        assert (
            translate_runtime_text("Neutral liquidity reduces support for the dollar.")
            == "中性流动性环境对美元的支撑有所减弱。"
        )
        assert (
            translate_runtime_text(
                "United States is in Goldilocks with neutral liquidity, so dollar duration remains neutral. Valuations look fair."
            )
            == "美国当前处于温和增长阶段，流动性环境中性，因此美元久期维持中性，估值大体合理。"
        )
        assert (
            translate_runtime_text("Model is more growth-positive than consensus.")
            == "模型对增长的判断比市场共识更偏乐观。"
        )
        assert (
            translate_runtime_text(
                "China is in Slowdown with neutral liquidity, which limits the equity view. Valuations look cheap."
            )
            == "中国当前处于增长放缓阶段，流动性环境中性，对权益资产判断形成约束，估值大体低估。"
        )
        assert (
            translate_runtime_text(
                "United States is in Goldilocks with neutral liquidity, but rich valuations keep the equity view from turning fully bullish. Valuations look expensive."
            )
            == "美国当前处于温和增长阶段，流动性环境中性，但偏贵的估值限制了权益资产进一步转向超配，估值大体高估。"
        )
        assert (
            translate_runtime_text("The macro backdrop is mixed for global equities. Valuations look expensive.")
            == "全球权益资产的宏观信号目前偏中性，估值大体高估。"
        )
        assert "最新可用模式" in translate_runtime_text(
            "Latest available compares each region on its own latest valid date: United States 2026-02-01 (fresh), China 2026-03-01 (fresh). Coverage is 100%."
        )
    finally:
        set_display_language("en")


def test_user_manual_supports_chinese_and_english() -> None:
    """The in-app manual should render both Chinese and English versions."""
    set_display_language("zh")
    zh_manual = build_user_manual_markdown()
    assert "四种宏观阶段先看这个" in zh_manual
    assert "典型超配" in zh_manual
    assert "温和增长" in zh_manual
    assert "共识偏差" in zh_manual
    assert "超配" in zh_manual
    assert "美元久期" in zh_manual

    set_display_language("en")
    en_manual = build_user_manual_markdown()
    assert "Start with the four macro states" in en_manual
    assert "Typical overweight" in en_manual
    assert "Goldilocks" in en_manual
    assert "consensus deviation" in en_manual.lower()
    assert "USD Duration" in en_manual
