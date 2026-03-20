"""Streamlit dashboard entry point."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.errors import EmptyDataError

# Streamlit can treat this file as the top-level `app` module, which conflicts
# with the project package named `app`. Ensure the project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from app.regime.change_detection import build_mode_comparison
from app.regime.global_monitor import build_country_status
from app.regime.nowcast import build_country_nowcast_overlay, build_global_nowcast_overlay
from app.utils.config import get_country_config, get_supported_countries
from app.data.china_ingestion import validate_china_data
from app.data.eurozone_ingestion import validate_eurozone_data


ACTIVE_LANGUAGE = "en"
LANGUAGE_OPTIONS = {"中文": "zh", "English": "en"}

LABEL_MAP_EN = {
    "latest_available": "Latest available",
    "last_common_date": "Last common date",
    "partial_view": "Partial view",
    "goldilocks": "Goldilocks",
    "reflation": "Reflation",
    "stagflation": "Stagflation",
    "slowdown": "Slowdown",
    "overheating": "Overheating",
    "disinflationary_growth": "Disinflationary growth",
    "neutral": "Neutral",
    "easy": "Easy",
    "tight": "Tight",
    "bullish": "Bullish",
    "cautious": "Cautious",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "fresh": "Fresh",
    "stale": "Stale",
    "very_stale": "Very stale",
    "ready": "Ready",
    "partial": "Partial",
    "missing": "Missing",
    "us": "United States",
    "china": "China",
    "eurozone": "Eurozone",
    "country_ready": "Country Ready",
    "globally_usable_latest": "Usable in Selected Mode",
    "days_stale": "Days Stale",
    "valuation_status": "Valuation Status",
    "global_equities": "Global Equities",
    "us_equities": "US Equities",
    "china_equities": "China Equities",
    "eurozone_equities": "Eurozone Equities",
    "global_regime": "Global Regime",
    "investment_clock": "Investment Clock",
    "growth_proxy": "Growth Proxy",
    "core_cpi": "Core CPI",
    "m3": "M3",
    "minimum": "Minimum",
    "enhanced": "Enhanced",
    "enhanced_partially_stale": "Enhanced (partially stale)",
    "rich": "Rich",
    "eurozone_ecb": "Eurozone ECB",
    "eurozone_eurostat": "Eurozone Eurostat",
    "eurozone_oecd": "Eurozone OECD",
    "strong_positive": "Strong positive",
    "strong_negative": "Strong negative",
    "positive": "Positive",
    "negative": "Negative",
    "strong_disinflation": "Strong disinflation",
    "disinflation": "Disinflation",
    "inflationary": "Inflationary",
    "strong_inflationary": "Strong inflationary",
    "dovish": "Dovish",
    "hawkish": "Hawkish",
    "strongly_dovish": "Strongly dovish",
    "strongly_hawkish": "Strongly hawkish",
    "unknown": "Unknown",
    "none": "none",
    "growth": "Growth",
    "inflation": "Inflation",
    "policy_bias": "Policy Bias",
    "equities": "Equities",
    "duration": "Duration",
    "gold": "Gold",
    "dollar": "Dollar",
    "policy_rate": "Policy Rate",
    "yield_10y": "Yield 10Y",
    "cpi": "CPI",
    "pmi": "PMI",
    "industrial_production": "Industrial Production",
    "unrate": "Unemployment Rate",
    "equity_pe_proxy": "Equity PE Proxy",
    "equity_pb_proxy": "Equity PB Proxy",
    "ecb": "ECB",
    "eurostat": "Eurostat",
    "eurostat_flash": "Eurostat Flash",
    "china_akshare": "China Akshare",
    "china_nbs": "China NBS",
    "china_rates": "China Rates",
    "normalized_api": "Normalized API",
    "No loaded data": "No loaded data",
}

LABEL_MAP_ZH = {
    "latest_available": "最新可得口径",
    "last_common_date": "共同日期口径",
    "partial_view": "局部视角",
    "goldilocks": "温和增长",
    "reflation": "再通胀",
    "stagflation": "滞胀",
    "slowdown": "增长放缓",
    "overheating": "过热",
    "disinflationary_growth": "低通胀扩张",
    "neutral": "中性",
    "easy": "宽松",
    "tight": "偏紧",
    "bullish": "超配",
    "cautious": "低配",
    "cheap": "低估",
    "fair": "合理",
    "expensive": "高估",
    "high": "高",
    "medium": "中",
    "low": "低",
    "fresh": "最新",
    "stale": "滞后",
    "very_stale": "严重滞后",
    "ready": "已就绪",
    "partial": "部分可用",
    "missing": "缺失",
    "us": "美国",
    "china": "中国",
    "eurozone": "欧元区",
    "country_ready": "本地区已就绪",
    "globally_usable_latest": "当前模式可用",
    "days_stale": "滞后天数",
    "valuation_status": "估值状态",
    "global_equities": "全球权益资产",
    "us_equities": "美国权益资产",
    "china_equities": "中国权益资产",
    "eurozone_equities": "欧元区权益资产",
    "global_regime": "全球宏观阶段",
    "investment_clock": "投资时钟",
    "growth_proxy": "增长代理",
    "core_cpi": "核心CPI",
    "m3": "M3",
    "minimum": "最低配置",
    "enhanced": "增强版",
    "enhanced_partially_stale": "增强版（部分滞后）",
    "rich": "丰富版",
    "eurozone_ecb": "欧央行",
    "eurozone_eurostat": "欧盟统计局",
    "eurozone_oecd": "经合组织",
    "strong_positive": "明显偏强",
    "strong_negative": "明显偏弱",
    "positive": "偏强",
    "negative": "偏弱",
    "strong_disinflation": "明显去通胀",
    "disinflation": "去通胀",
    "inflationary": "通胀压力升温",
    "strong_inflationary": "通胀压力明显升温",
    "dovish": "鸽派",
    "hawkish": "鹰派",
    "strongly_dovish": "明显鸽派",
    "strongly_hawkish": "明显鹰派",
    "unknown": "未知",
    "none": "无",
    "growth": "增长",
    "inflation": "通胀",
    "policy_bias": "政策倾向",
    "equities": "权益资产",
    "duration": "久期",
    "gold": "黄金",
    "dollar": "美元",
    "commodity": "大宗商品",
    "commodities": "大宗商品",
    "policy_rate": "政策利率",
    "yield_10y": "10年期收益率",
    "cpi": "CPI",
    "pmi": "PMI",
    "industrial_production": "工业增加值",
    "unrate": "失业率",
    "equity_pe_proxy": "权益市盈率代理",
    "equity_pb_proxy": "权益市净率代理",
    "ecb": "ECB",
    "eurostat": "Eurostat",
    "eurostat_flash": "Eurostat 快报",
    "china_akshare": "AkShare",
    "china_nbs": "国家统计局",
    "china_rates": "中国货币网/债券数据",
    "normalized_api": "标准化API数据",
    "No loaded data": "未加载数据",
}

TEXT_MAP_ZH = {
    "No data": "无数据",
    "View": "视图",
    "Language": "语言",
    "Global": "全球",
    "{country} Macro Monitor": "{country}宏观监测",
    "Region: {region} | Currency: {currency}": "地区：{region} | 货币：{currency}",
    "No regime data available for this country yet.": "该地区暂时没有可用的宏观阶段数据。",
    "{country} currently relies on very stale inputs. Latest local regime date: {date}.": "{country}当前依赖严重滞后的输入数据。本地最新阶段日期：{date}。",
    "Latest Regime": "当前宏观阶段",
    "Liquidity Overlay": "流动性环境",
    "Latest Date": "最新日期",
    "System Update Date": "系统更新日期",
    "Data Through Date": "数据截至日期",
    "Latest Market-sensitive Input": "最新市场敏感输入",
    "Nowcast Overlay": "实时偏移层",
    "Growth Score": "增长评分",
    "Inflation Score": "通胀评分",
    "Liquidity Score": "流动性评分",
    "Regime Confidence": "阶段置信度",
    "Raw Regime": "原始阶段判定",
    "Valuation Regime": "估值状态",
    "Valuation Score": "估值评分",
    "China Data Status": "中国数据状态",
    "Eurozone Data Status": "欧元区数据状态",
    "API Series Available": "API 可用序列数",
    "Regime Input Ready": "阶段输入已就绪",
    "Using API Data": "正在使用 API 数据",
    "Yes": "是",
    "No": "否",
    "Missing required {country} regime inputs: {items}": "{country}阶段判定仍缺少必需输入：{items}",
    "Minimum {country} regime inputs are satisfied.": "{country}阶段判定的最低输入已满足。",
    "Minimum inputs used: {items}": "最低输入使用项：{items}",
    "Enrichment Inputs Available": "增强输入可用项",
    "Scoring Richness": "评分丰富度",
    "Enrichment inputs used in scoring: {items}": "参与评分的增强输入：{items}",
    "Available but ignored due to staleness: {items}": "已获取但因滞后被忽略：{items}",
    "Classification works, richness reduced. Optional series still missing: {items}": "分类仍可运行，但信息丰富度下降。可选序列仍缺失：{items}",
    "Valuation Proxy Readiness": "估值代理已就绪",
    "Valuation Proxy Series Found": "已识别的估值代理序列",
    "Series": "序列",
    "Source": "数据源",
    "Row Count": "行数",
    "Status": "状态",
    "Asset": "资产",
    "Preference": "偏好",
    "Score": "评分",
    "Confidence": "置信度",
    "Reason": "理由",
    "Mode Context": "模式说明",
    "Dimension": "维度",
    "Consensus View": "市场共识",
    "Model View": "模型判断",
    "Deviation Score": "偏离评分",
    "Consensus Deviation Score": "总偏离评分",
    "Policy Deviation Score": "政策偏离评分",
    "Summary": "摘要",
    "From Regime": "起始阶段",
    "To Regime": "目标阶段",
    "Count": "次数",
    "Window": "窗口",
    "Average Forward Return": "平均前瞻收益",
    "Median Forward Return": "中位数前瞻收益",
    "Hit Ratio": "正收益概率",
    "Bucket": "分桶",
    "Valuation canonical ids found: {items}": "已识别的估值规范序列：{items}",
    "Valuation proxy inputs used: {items}": "实际使用的估值代理输入：{items}",
    "Valuation proxy inputs missing: {items}": "缺失的估值代理输入：{items}",
    "Valuation actual sources found: {items}": "估值实际数据源：{items}",
    "No normalized {country} API files are available yet. The current regime is coming from manual fallback data.": "尚未发现 {country} 的标准化 API 文件，当前阶段结果来自手工兜底数据。",
    "{country} can only join the latest global view after fresh API or fallback data exists for its minimum regime inputs.": "{country}只有在最低阶段输入具备更新的API或兜底数据后，才能进入最新全球视图。",
    "Consensus Deviation": "共识偏差",
    "No consensus deviation snapshot is available for this region yet.": "该地区暂时没有可用的共识偏差快照。",
    "Source Count": "来源数量",
    "Consensus Date": "共识日期",
    "Consensus Confidence": "共识置信度",
    "Latest consensus note date: {date}": "最新共识笔记日期：{date}",
    "Latest Asset Preferences": "最新资产偏好",
    "Allocation confidence: {level}. {note}": "资产偏好置信度：{level}。{note}",
    "Recent Asset Preference History": "近期资产偏好历史",
    "Global Macro Monitor": "全球宏观监测",
    "Weighted summary across US, China, and Eurozone.": "基于美国、中国和欧元区加权汇总的全球宏观监测。",
    "No global summary data available yet.": "暂时没有可用的全球汇总数据。",
    "Global mode": "全球对齐模式",
    "No global summary data available for the selected mode yet.": "当前所选模式下暂无全球汇总数据。",
    "Selected Mode": "当前模式",
    "Summary Date": "汇总日期",
    "System Update Date": "系统更新日期",
    "Data Through Date": "数据截至日期",
    "Latest Market-sensitive Input": "最新市场敏感输入",
    "Nowcast Overlay": "实时偏移层",
    "Latest Global Regime": "当前全球宏观阶段",
    "Investment Clock": "投资时钟",
    "Coverage Ratio": "覆盖率",
    "Available Countries": "可用地区",
    "Missing Countries": "缺失地区",
    "A country can be Ready locally but still not usable in the selected global view if it has no valid data on that mode's evaluation date.": "某个地区即使本地已就绪，也可能因在该模式评估日没有有效数据而无法纳入当前全球视图。",
    "Weighting": "权重设置",
    "Configured weights: {weights}": "配置权重：{weights}",
    "Effective weights: {weights}": "实际生效权重：{weights}",
    "Only one country is available in the selected mode, so the effective global view is entirely driven by that market.": "当前模式下只有一个地区可用，因此有效的全球视图完全由该市场驱动。",
    "Country Status": "地区状态",
    "Country": "地区",
    "Regime": "宏观阶段",
    "Country Ready": "本地区已就绪",
    "Usable in Selected Mode": "当前模式可用",
    "Days Stale": "滞后天数",
    "Staleness Status": "滞后状态",
    "Valuation Status": "估值状态",
    "Stale country data detected: {countries}": "检测到滞后地区数据：{countries}",
    "Global Allocation": "全球资产偏好",
    "No global allocation map available for the selected mode yet.": "当前所选模式下暂无全球资产偏好结果。",
    "Reason explains the asset view itself. Confidence falls when country coverage is thin, country data is stale, or valuation inputs are missing.": "理由解释的是资产偏好的核心依据。置信度会因地区覆盖不足、数据滞后或估值输入缺失而下降。",
    "No consensus deviation table": "暂无共识偏差表",
    "What Changed": "最新变化",
    "Current snapshot time: {time}": "当前快照时间：{time}",
    "Prior snapshot time: {time}": "上一可比快照时间：{time}",
    "Regime Changes": "阶段变化",
    "Preference Changes": "偏好变化",
    "Confidence Changes": "置信度变化",
    "Why It Changed": "变化原因",
    "No clear change driver was recorded.": "没有记录到明确的变化驱动因素。",
    "No regime changes": "没有阶段变化",
    "No preference changes": "没有偏好变化",
    "No confidence changes": "没有置信度变化",
    "Regime Evaluation": "阶段评估",
    "Regime Frequency": "阶段频次",
    "Transition Matrix": "阶段迁移矩阵",
    "Forward Return Summary": "前瞻收益汇总",
    "Confidence Bucket Summary": "置信度分桶汇总",
    "No regime frequency summary": "暂无阶段频次汇总",
    "No transition matrix": "暂无迁移矩阵",
    "No forward return summary": "暂无前瞻收益汇总",
    "No confidence bucket summary": "暂无置信度分桶汇总",
    "No prior snapshot is available yet for this mode.": "当前模式暂时还没有上一条可比快照。",
    "China valuation is proxy-based: HS300 PE/PB when available, plus real yield and term spread proxies.": "中国估值为代理指标口径：优先使用沪深300市盈率/市净率代理，并辅以实际收益率与期限利差代理。",
    "The system was refreshed on {system_date}. The macro regime is still based on data through {data_date}.": "系统已于 {system_date} 刷新，但宏观阶段仍基于截至 {data_date} 的数据。",
    "{country} has fresher market-sensitive inputs through {market_date} ({series}), while the macro regime still runs through {data_date}.": "{country} 的市场敏感输入已更新到 {market_date}（{series}），但宏观阶段仍截至 {data_date}。",
    "No fresher market-sensitive inputs were found beyond the current macro snapshot.": "当前没有比主模型更晚的市场敏感输入。",
    "Global macro state is based on data through {data_date}, while fresher market-sensitive inputs are available through {market_date} ({details}).": "全球宏观主状态截至 {data_date}，但更高频的市场敏感输入已更新到 {market_date}（{details}）。",
    "Global macro state and the latest market-sensitive inputs are aligned at the same date.": "全球宏观主状态与最新市场敏感输入日期一致。",
}
REGION_MAP_ZH = {"North America": "北美", "Europe": "欧洲", "Asia": "亚洲"}
FREE_TEXT_LABEL_MAP_ZH = {
    "Goldilocks": "温和增长",
    "Reflation": "再通胀",
    "Stagflation": "滞胀",
    "Slowdown": "增长放缓",
    "Partial view": "局部视角",
    "Unknown": "未知",
    "Easy": "宽松",
    "Neutral": "中性",
    "Tight": "偏紧",
    "easy": "宽松",
    "neutral": "中性",
    "tight": "偏紧",
    "Fair": "合理",
    "Cheap": "低估",
    "Expensive": "高估",
    "fair": "合理",
    "cheap": "低估",
    "expensive": "高估",
    "fresh": "最新",
    "stale": "滞后",
    "very stale": "严重滞后",
    "United States": "美国",
    "China": "中国",
    "Eurozone": "欧元区",
    "global equities": "全球权益资产",
    "equities": "权益资产",
    "duration": "久期",
    "gold": "黄金",
    "dollar": "美元",
    "commodities": "大宗商品",
    "commodity": "大宗商品",
}
CONSENSUS_TEXT_MAP_ZH = {
    "Model is broadly aligned with consensus.": "模型判断与当前市场共识大体一致。",
    "Growth, inflation, and policy-bias signals are close to current public narratives.": "增长、通胀和政策倾向信号与当前主流公开叙事大体接近。",
    "Model is more growth-positive than consensus.": "模型对增长的判断比市场共识更偏乐观。",
    "Model is more growth-negative than consensus.": "模型对增长的判断比市场共识更偏谨慎。",
    "The model reads stronger growth momentum than mainstream narratives.": "模型显示的增长动能强于主流叙事所反映的水平。",
    "The model reads weaker growth momentum than mainstream narratives.": "模型显示的增长动能弱于主流叙事所反映的水平。",
    "Model sees less inflation risk than current public narratives.": "模型判断的通胀风险低于当前主流公开叙事。",
    "Model sees more inflation risk than current public narratives.": "模型判断的通胀风险高于当前主流公开叙事。",
    "The model's inflation view is more benign than consensus.": "模型的通胀判断比市场共识更温和。",
    "The model's inflation view is less benign than consensus.": "模型的通胀判断比市场共识更偏不利。",
    "Model is more dovish than consensus.": "模型对政策环境的判断比市场共识更偏鸽派。",
    "Model is more hawkish than consensus.": "模型对政策环境的判断比市场共识更偏鹰派。",
    "Liquidity conditions look easier than the mainstream policy narrative suggests.": "模型所反映的流动性条件比主流政策叙事暗示的更为宽松。",
    "Liquidity conditions look tighter than the mainstream policy narrative suggests.": "模型所反映的流动性条件比主流政策叙事暗示的更为偏紧。",
    "No consensus snapshot is available for this region yet.": "该地区暂时还没有可用的市场共识快照。",
    "Add recent consensus notes before comparing the model with public narratives.": "请先补充近期共识笔记，再与模型判断进行比较。",
    "Regime is based on growth and inflation scores with sufficient distance from neutral.": "当前阶段判断基于增长与通胀评分，且与中性阈值保持足够距离。",
    "Regime sits near the neutral boundary and should be read with caution.": "当前阶段位于中性边界附近，解读时宜保持谨慎。",
    "The latest scores remain inside the neutral buffer, so the prior regime is retained.": "最新评分仍处于中性缓冲区内，因此延续上一期阶段判断。",
}


def set_display_language(language: str) -> None:
    """Persist the active dashboard language."""
    global ACTIVE_LANGUAGE
    ACTIVE_LANGUAGE = language
    try:
        st.session_state["_display_language"] = language
    except Exception:
        pass


def get_display_language(language: str | None = None) -> str:
    """Return the active dashboard language."""
    if language:
        return language
    try:
        return str(st.session_state.get("_display_language", ACTIVE_LANGUAGE))
    except Exception:
        return ACTIVE_LANGUAGE


def tr(text: str, language: str | None = None, **kwargs: object) -> str:
    """Translate a UI string when Chinese display is active."""
    lang = get_display_language(language)
    template = text if lang == "en" else TEXT_MAP_ZH.get(text, text)
    return template.format(**kwargs)


def localize_region(region: str, language: str | None = None) -> str:
    """Translate region labels for the page header."""
    if get_display_language(language) == "zh":
        return REGION_MAP_ZH.get(region, region)
    return region


def translate_runtime_text(text: object, language: str | None = None) -> str:
    """Translate runtime-generated English rationale text into Chinese when needed."""
    raw = format_display_value(text)
    if get_display_language(language) != "zh" or raw == "No data":
        return raw
    if raw in CONSENSUS_TEXT_MAP_ZH:
        return CONSENSUS_TEXT_MAP_ZH[raw]

    translated = raw
    for english, chinese in sorted(FREE_TEXT_LABEL_MAP_ZH.items(), key=lambda item: -len(item[0])):
        translated = translated.replace(english, chinese)

    latest_available_match = re.match(
        r"Latest available compares each region on its own latest valid date: (?P<body>.+)\. Coverage is (?P<coverage>[\d.]+%)\.",
        raw,
    )
    if latest_available_match:
        body = latest_available_match.group("body")
        coverage = latest_available_match.group("coverage")
        for english, chinese in sorted(FREE_TEXT_LABEL_MAP_ZH.items(), key=lambda item: -len(item[0])):
            body = body.replace(english, chinese)
        return f"最新可用模式按各地区自身最近有效日期进行比较：{body}。覆盖率为 {coverage}。"

    last_common_match = re.match(
        r"Last common date compares all contributing regions on (?P<date>\d{4}-\d{2}-\d{2}) across (?P<countries>.+)\. Coverage is (?P<coverage>[\d.]+%)\.",
        raw,
    )
    if last_common_match:
        countries = last_common_match.group("countries")
        coverage = last_common_match.group("coverage")
        for english, chinese in sorted(FREE_TEXT_LABEL_MAP_ZH.items(), key=lambda item: -len(item[0])):
            countries = countries.replace(english, chinese)
        return (
            f"共同最近日期模式以 {last_common_match.group('date')} 作为共同评估日，"
            f"纳入地区包括 {countries}。覆盖率为 {coverage}。"
        )

    latest_country_match = re.match(
        r"Latest available uses (?P<country>.+?) data through (?P<date>\d{4}-\d{2}-\d{2})\.",
        raw,
    )
    if latest_country_match:
        country = latest_country_match.group("country")
        for english, chinese in sorted(FREE_TEXT_LABEL_MAP_ZH.items(), key=lambda item: -len(item[0])):
            country = country.replace(english, chinese)
        return f"最新可用模式使用 {country} 截至 {latest_country_match.group('date')} 的数据。"

    exact_map = {
        "Cycle signals are not strong enough for a clear duration call.": "周期信号尚不足以支持明确的久期配置判断。",
        "The current macro backdrop reduces the need for gold protection.": "当前宏观环境降低了黄金防御配置的必要性。",
        "Commodity signals are mixed.": "大宗商品信号偏分化。",
        "Gold signals are mixed.": "黄金信号偏分化。",
        "The macro backdrop is mixed for the dollar.": "美元的宏观信号目前偏中性。",
        "The macro backdrop is mixed for global equities.": "全球权益资产的宏观信号目前偏中性。",
        "No countries were usable in the selected mode.": "当前所选模式下没有可用地区可参与计算。",
        "Global summary is based on incomplete country coverage.": "全球汇总基于不完整的地区覆盖，解读时需谨慎。",
    }
    if raw in exact_map:
        return exact_map[raw]

    pattern = re.match(
        r"(?P<regime>.+?) and (?P<liquidity>.+?) liquidity still support global equities\. Valuations look (?P<valuation>.+?)\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        tail = pattern.group("tail")
        result = (
            f"{translate_runtime_text(pattern.group('regime'))}阶段与{translate_runtime_text(pattern.group('liquidity'))}"
            f"流动性环境仍支撑全球权益资产，估值大体{translate_runtime_text(pattern.group('valuation'))}。"
        )
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"(?P<liquidity>.+?) liquidity reduces support for the dollar\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = f"{translate_runtime_text(pattern.group('liquidity'))}流动性环境对美元的支撑有所减弱。"
        tail = pattern.group("tail")
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"(?P<country>United States|China|Eurozone) is in (?P<regime>.+?) with (?P<liquidity>.+?) liquidity, which supports equities\. Valuations look (?P<valuation>.+?)\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = (
            f"{translate_runtime_text(pattern.group('country'))}当前处于{translate_runtime_text(pattern.group('regime'))}阶段，"
            f"流动性环境{translate_runtime_text(pattern.group('liquidity'))}，对权益资产形成支撑，"
            f"估值大体{translate_runtime_text(pattern.group('valuation'))}。"
        )
        tail = pattern.group("tail")
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"(?P<country>United States|China|Eurozone) is in (?P<regime>.+?) with (?P<liquidity>.+?) liquidity, which limits the equity view\. Valuations look (?P<valuation>.+?)\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = (
            f"{translate_runtime_text(pattern.group('country'))}当前处于{translate_runtime_text(pattern.group('regime'))}阶段，"
            f"流动性环境{translate_runtime_text(pattern.group('liquidity'))}，对权益资产判断形成约束，"
            f"估值大体{translate_runtime_text(pattern.group('valuation'))}。"
        )
        tail = pattern.group("tail")
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"United States is in (?P<regime>.+?) with (?P<liquidity>.+?) liquidity, so dollar duration remains neutral\. Valuations look (?P<valuation>.+?)\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = (
            f"美国当前处于{translate_runtime_text(pattern.group('regime'))}阶段，"
            f"流动性环境{translate_runtime_text(pattern.group('liquidity'))}，"
            f"因此美元久期维持中性，估值大体{translate_runtime_text(pattern.group('valuation'))}。"
        )
        tail = pattern.group("tail")
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"United States is in (?P<regime>.+?) with (?P<liquidity>.+?) liquidity, which supports dollar duration\. Valuations look (?P<valuation>.+?)\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = (
            f"美国当前处于{translate_runtime_text(pattern.group('regime'))}阶段，"
            f"流动性环境{translate_runtime_text(pattern.group('liquidity'))}，"
            f"对美元久期形成支撑，估值大体{translate_runtime_text(pattern.group('valuation'))}。"
        )
        tail = pattern.group("tail")
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"United States is in (?P<regime>.+?) with (?P<liquidity>.+?) liquidity, which limits the dollar-duration view\. Valuations look (?P<valuation>.+?)\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = (
            f"美国当前处于{translate_runtime_text(pattern.group('regime'))}阶段，"
            f"流动性环境{translate_runtime_text(pattern.group('liquidity'))}，"
            f"对美元久期判断形成约束，估值大体{translate_runtime_text(pattern.group('valuation'))}。"
        )
        tail = pattern.group("tail")
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"(?P<country>United States|China|Eurozone) macro signals are mixed for equities\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = f"{translate_runtime_text(pattern.group('country'))}的权益资产宏观信号目前偏中性。"
        tail = pattern.group("tail")
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"Confidence is reduced by (?P<issues>.+)\.",
        raw,
    )
    if pattern:
        issues = pattern.group("issues")
        issue_map = {
            "partial country coverage": "地区覆盖不足",
            "very stale country data": "地区数据严重滞后",
            "stale country data": "地区数据滞后",
            "the market is excluded from this mode": "该市场未被纳入当前模式",
            "very stale local data": "本地数据严重滞后",
            "stale local data": "本地数据滞后",
            "missing valuation input": "估值输入缺失",
            " and ": "、",
            ", and ": "、",
            ", ": "、",
        }
        for english, chinese in issue_map.items():
            issues = issues.replace(english, chinese)
        return f"置信度因{issues}而下调。"

    return translated


def humanize_label(value: object) -> str:
    """Convert internal labels into a user-friendly display form."""
    text = format_display_value(value)
    if text == tr("No data"):
        return text
    mapping = LABEL_MAP_EN if get_display_language() == "en" else LABEL_MAP_ZH
    return mapping.get(text, text.replace("_", " ").title())


def format_country_list(value: object) -> str:
    """Format comma-separated country codes into readable names."""
    text = format_display_value(value)
    if text == tr("No data"):
        return text
    parts = [item for item in text.split(",") if item]
    if not parts:
        return tr("No data")
    return ", ".join(humanize_label(item) for item in parts)


def format_weight_map(value: object) -> str:
    """Format weight strings into readable dashboard text."""
    text = format_display_value(value)
    if text == tr("No data"):
        return text
    parts = [item for item in text.split(",") if item]
    if not parts:
        return tr("No data")
    formatted: list[str] = []
    for part in parts:
        country, _, weight = part.partition(":")
        if not country:
            continue
        try:
            formatted.append(f"{humanize_label(country)} {float(weight):.0%}")
        except ValueError:
            formatted.append(f"{humanize_label(country)} {weight}")
    return ", ".join(formatted) if formatted else "No data"


def format_display_value(value: object, missing_label: str = "No data") -> str:
    """Format dashboard values while keeping missing data explicit."""
    missing_value = tr("No data") if missing_label == "No data" else missing_label
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return missing_value
    if isinstance(value, str) and not value.strip():
        return missing_value
    return str(value)


def _format_optional_date(value: object) -> str:
    """Format an optional timestamp or date for dashboard metrics."""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return tr("No data")
    return timestamp.date().isoformat()


def format_entity_name(value: object) -> str:
    """Format change-log entity names into readable display labels."""
    return humanize_label(value)


def humanize_global_asset_label(value: object) -> str:
    """Format global allocation asset labels with page-specific wording."""
    text = str(value)
    if get_display_language() == "zh" and text == "duration":
        return "美元久期"
    return humanize_label(text)


def humanize_country_asset_label(country: str, asset: object) -> str:
    """Format country-view asset labels with local-currency wording."""
    del country
    return humanize_label(asset)


def format_change_sentence(
    entity_name: object,
    field_label: str,
    old_value: object,
    new_value: object,
) -> str:
    """Build a user-friendly change sentence."""
    label = format_entity_name(entity_name)
    if get_display_language() == "zh":
        if field_label == "changed":
            return f"{label}由 {humanize_label(old_value)} 变为 {humanize_label(new_value)}。"
        field_map = {"confidence": "置信度", "preference": "偏好", "regime": "阶段"}
        field_text = field_map.get(field_label, field_label)
        return f"{label}的{field_text}由 {humanize_label(old_value)} 变为 {humanize_label(new_value)}。"
    if field_label == "changed":
        return f"{label} changed from {humanize_label(old_value)} to {humanize_label(new_value)}."
    return (
        f"{label} {field_label} changed from "
        f"{humanize_label(old_value)} to {humanize_label(new_value)}."
    )


def summarize_change_reasons(frame: pd.DataFrame) -> list[str]:
    """Return deduplicated reason labels for the latest change set."""
    if frame.empty or "reason" not in frame.columns:
        return []
    reasons = []
    for reason in frame["reason"].dropna().astype(str):
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def split_change_sections(
    mode_changes: pd.DataFrame,
    country_changes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split change log rows into mutually exclusive dashboard sections."""
    if "change_type" not in mode_changes.columns:
        mode_changes = pd.DataFrame(columns=["change_type"])
    return {
        "regime": mode_changes.loc[mode_changes["change_type"] == "regime_change"].copy(),
        "preference": mode_changes.loc[mode_changes["change_type"] == "preference_change"].copy(),
        "confidence": mode_changes.loc[mode_changes["change_type"] == "confidence_change"].copy(),
        "country": country_changes.copy(),
    }


def comparison_message(meta_row: pd.Series | None) -> str:
    """Return a user-facing comparison message for a mode."""
    if meta_row is None:
        return tr("No prior snapshot is available yet for this mode.")
    return tr(format_display_value(meta_row.get("reason")))


def prepare_what_changed_sections(comparison: dict[str, object]) -> dict[str, object]:
    """Normalize one comparison object into dashboard-ready sections."""
    return {
        "available": bool(comparison["comparison_available"]),
        "message": tr(str(comparison["comparison_reason"])),
        "current_snapshot_timestamp": comparison["current_snapshot_timestamp"],
        "prior_snapshot_timestamp": comparison["prior_snapshot_timestamp"],
        "regime_changes": pd.DataFrame(comparison["regime_changes"]),
        "preference_changes": pd.DataFrame(comparison["preference_changes"]),
        "confidence_changes": pd.DataFrame(comparison["confidence_changes"]),
        "why_it_changed": list(comparison["why_it_changed"]),
        "regime_change_count": int(comparison["regime_change_count"]),
        "preference_change_count": int(comparison["preference_change_count"]),
        "confidence_change_count": int(comparison["confidence_change_count"]),
    }


def _status_from_frame(frame: pd.DataFrame, required_columns: list[str]) -> str:
    """Summarize data availability for a processed frame."""
    if frame.empty:
        return "missing"
    available = sum(
        1 for column in required_columns if column in frame.columns and frame[column].notna().any()
    )
    if available == 0:
        return "missing"
    if available < len(required_columns):
        return "partial"
    return "ready"


def load_csv_with_dates(path: Path, build_hint: str) -> pd.DataFrame:
    """Load a processed CSV with a normalized date column."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run `{build_hint}` first.")

    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    for column in ["date", "summary_date", "run_timestamp"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    sort_column = "date" if "date" in frame.columns else ("summary_date" if "summary_date" in frame.columns else None)
    if sort_column is None:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_column).reset_index(drop=True)


def load_optional_csv(path: Path) -> pd.DataFrame:
    """Load an optional CSV file if it exists, otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    for column in ["date", "latest_date", "release_date", "ingested_at", "snapshot_date", "model_date", "latest_consensus_note_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def render_country_view(country: str) -> None:
    """Render the dashboard for one country."""
    country_meta = get_country_config(country)
    regime_data = load_csv_with_dates(
        Path(f"data/processed/{country}_macro_regimes.csv"),
        f"python main.py classify-country-regime --country {country}",
    )
    valuation_data = load_csv_with_dates(
        Path(f"data/processed/{country}_valuation_features.csv"),
        f"python main.py build-country-valuation --country {country}",
    )
    asset_data = load_csv_with_dates(
        Path(f"data/processed/{country}_asset_preferences.csv"),
        f"python main.py map-country-assets --country {country}",
    )
    consensus_deviation = load_optional_csv(Path("data/processed/consensus_deviation.csv"))

    country_label = humanize_label(country)
    st.title(tr("{country} Macro Monitor", country=country_label))
    st.caption(
        tr(
            "Region: {region} | Currency: {currency}",
            region=localize_region(str(country_meta["region"])),
            currency=country_meta["currency"],
        )
    )

    if regime_data.empty:
        st.warning(tr("No regime data available for this country yet."))
        return

    latest = regime_data.iloc[-1]
    latest_valuation = valuation_data.iloc[-1] if not valuation_data.empty else pd.Series(dtype="object")
    latest_assets = asset_data.iloc[-1] if not asset_data.empty else pd.Series(dtype="object")
    latest_date = pd.Timestamp(latest["date"])
    nowcast_overlay = build_country_nowcast_overlay(country, latest_date)
    system_update_date = _format_optional_date(nowcast_overlay["system_update_timestamp"])
    market_input_date = _format_optional_date(nowcast_overlay["freshest_market_date"])

    if (pd.Timestamp.utcnow().tz_localize(None) - latest_date).days > 180:
        st.warning(
            tr(
                "{country} currently relies on very stale inputs. Latest local regime date: {date}.",
                country=country_label,
                date=latest_date.date().isoformat(),
            )
        )

    top_left, top_mid, top_right, top_far_right = st.columns(4)
    top_left.metric(tr("Latest Regime"), humanize_label(latest.get("regime")))
    top_mid.metric(tr("Liquidity Overlay"), humanize_label(latest.get("liquidity_regime")))
    top_right.metric(tr("System Update Date"), system_update_date)
    top_far_right.metric(tr("Data Through Date"), pd.Timestamp(latest["date"]).date().isoformat())

    st.subheader(tr("Nowcast Overlay"))
    overlay_left, overlay_mid = st.columns(2)
    overlay_left.metric(tr("Latest Market-sensitive Input"), market_input_date)
    overlay_mid.metric(
        tr("Source"),
        ", ".join(humanize_label(item) for item in nowcast_overlay["freshest_market_series"])
        if nowcast_overlay["freshest_market_series"]
        else tr("No data"),
    )
    if nowcast_overlay["has_newer_market_input"]:
        st.info(
            tr(
                "{country} has fresher market-sensitive inputs through {market_date} ({series}), while the macro regime still runs through {data_date}.",
                country=country_label,
                market_date=market_input_date,
                series=", ".join(humanize_label(item) for item in nowcast_overlay["freshest_market_series"]),
                data_date=latest_date.date().isoformat(),
            )
        )
    else:
        st.caption(
            tr(
                "The system was refreshed on {system_date}. The macro regime is still based on data through {data_date}.",
                system_date=system_update_date,
                data_date=latest_date.date().isoformat(),
            )
        )
        st.caption(tr("No fresher market-sensitive inputs were found beyond the current macro snapshot."))

    score_left, score_mid, score_right = st.columns(3)
    score_left.metric(tr("Growth Score"), f"{latest.get('growth_score', float('nan')):.2f}")
    score_mid.metric(tr("Inflation Score"), f"{latest.get('inflation_score', float('nan')):.2f}")
    score_right.metric(tr("Liquidity Score"), f"{latest.get('liquidity_score', float('nan')):.2f}")
    if "regime_confidence" in latest.index:
        confidence_left, confidence_mid = st.columns(2)
        confidence_left.metric(tr("Regime Confidence"), humanize_label(latest.get("regime_confidence")))
        confidence_mid.metric(tr("Raw Regime"), humanize_label(latest.get("regime_raw", latest.get("regime"))))
    if "regime_note" in latest.index:
        st.caption(translate_runtime_text(latest.get("regime_note")))

    valuation_left, valuation_mid = st.columns(2)
    valuation_left.metric(
        tr("Valuation Regime"),
        humanize_label(latest_valuation.get("valuation_regime")),
    )
    valuation_mid.metric(
        tr("Valuation Score"),
        f"{latest_valuation.get('valuation_score', float('nan')):.2f}",
    )
    if country == "china" and not valuation_data.empty:
        st.caption(
            tr(
                "China valuation is proxy-based: HS300 PE/PB when available, plus real yield and term spread proxies."
            )
        )

    if country in {"china", "eurozone"}:
        title = tr("China Data Status") if country == "china" else tr("Eurozone Data Status")
        st.subheader(title)
        validation = validate_china_data() if country == "china" else validate_eurozone_data()
        source_left, source_mid, source_right = st.columns(3)
        source_left.metric(
            tr("API Series Available"),
            len(validation["available_series"]),
        )
        source_mid.metric(
            tr("Regime Input Ready"),
            tr("Yes") if validation["regime_ready"] else tr("No"),
        )
        source_right.metric(
            tr("Using API Data"),
            tr("Yes") if bool(validation["available_series"]) else tr("No"),
        )
        if validation["missing_required_series"]:
            st.info(
                tr(
                    "Missing required {country} regime inputs: {items}",
                    country=country_label,
                    items=", ".join(validation["missing_required_series"]),
                )
            )
        else:
            st.caption(tr("Minimum {country} regime inputs are satisfied.", country=country_label))
        st.caption(
            tr(
                "Minimum inputs used: {items}",
                items=", ".join(validation["minimum_inputs_used"]) if validation["minimum_inputs_used"] else humanize_label("none"),
            )
        )
        enrichment_left, enrichment_mid = st.columns(2)
        enrichment_left.metric(
            tr("Enrichment Inputs Available"),
            ", ".join(humanize_label(item) for item in validation["enrichment_available_series"]) if validation["enrichment_available_series"] else humanize_label("none"),
        )
        enrichment_mid.metric(
            tr("Scoring Richness"),
            humanize_label(validation["scoring_richness_level"]),
        )
        st.caption(
            tr(
                "Enrichment inputs used in scoring: {items}",
                items=", ".join(humanize_label(item) for item in validation["enrichment_inputs_used"]) if validation["enrichment_inputs_used"] else humanize_label("none"),
            )
        )
        if validation["enrichment_inputs_ignored_stale"]:
            st.caption(
                tr(
                    "Available but ignored due to staleness: {items}",
                    items=", ".join(humanize_label(item) for item in validation["enrichment_inputs_ignored_stale"]),
                )
            )
        if validation["optional_missing_series"]:
            st.caption(
                tr(
                    "Classification works, richness reduced. Optional series still missing: {items}",
                    items=", ".join(humanize_label(item) for item in validation["optional_missing_series"]),
                )
            )
        valuation_left, valuation_mid = st.columns(2)
        valuation_left.metric(
            tr("Valuation Proxy Readiness"),
            tr("Yes") if validation["valuation_proxy_readiness"] else tr("No"),
        )
        valuation_mid.metric(
            tr("Valuation Proxy Series Found"),
            ", ".join(humanize_label(item) for item in validation["valuation_proxy_series_found"]) if validation["valuation_proxy_series_found"] else humanize_label("none"),
        )
        st.caption(
            tr(
                "Valuation canonical ids found: {items}",
                items=", ".join(humanize_label(item) for item in validation["valuation_canonical_series_ids_found"])
                if validation["valuation_canonical_series_ids_found"]
                else humanize_label("none"),
            )
        )
        st.caption(
            tr(
                "Valuation proxy inputs used: {items}",
                items=", ".join(humanize_label(item) for item in validation["valuation_proxy_inputs_used"])
                if validation["valuation_proxy_inputs_used"]
                else humanize_label("none"),
            )
        )
        if validation["valuation_proxy_inputs_missing"]:
            st.caption(
                tr(
                    "Valuation proxy inputs missing: {items}",
                    items=", ".join(humanize_label(item) for item in validation["valuation_proxy_inputs_missing"]),
                )
            )
        st.caption(
            tr(
                "Valuation actual sources found: {items}",
                items=", ".join(humanize_label(item) for item in validation["valuation_actual_sources_found"])
                if validation["valuation_actual_sources_found"]
                else humanize_label("none"),
            )
        )
        if validation["stale_warning"]:
            st.warning(translate_runtime_text(validation["stale_warning"]))
        series_status = validation["series_status"].copy()
        if series_status.empty:
                st.caption(
                    tr(
                        "No normalized {country} API files are available yet. The current regime is coming from manual fallback data.",
                        country=country_label,
                    )
                )
        else:
            display_summary = series_status.copy()
            display_summary["series_id"] = display_summary["series_id"].apply(humanize_label)
            display_summary["source_used"] = display_summary["source_used"].apply(
                humanize_label
            )
            display_summary["latest_date"] = display_summary["latest_date"].apply(
                lambda value: format_display_value(
                    pd.Timestamp(value).date().isoformat() if pd.notna(value) else None
                )
            )
            display_summary["status"] = display_summary["status"].apply(humanize_label)
            display_summary = display_summary.rename(
                columns={
                    "series_id": tr("Series"),
                    "source_used": tr("Source"),
                    "row_count": tr("Row Count"),
                    "latest_date": tr("Latest Date"),
                    "status": tr("Status"),
                }
            )
            st.dataframe(
                display_summary.loc[:, [tr("Series"), tr("Source"), tr("Row Count"), tr("Latest Date"), tr("Status")]],
                use_container_width=True,
                hide_index=True,
            )
            if not validation["regime_ready"]:
                st.caption(
                    tr(
                        "{country} can only join the latest global view after fresh API or fallback data exists for its minimum regime inputs.",
                        country=country_label,
                    )
                )

    st.subheader(tr("Consensus Deviation"))
    region_consensus = consensus_deviation.loc[consensus_deviation["region"] == country].copy() if not consensus_deviation.empty else pd.DataFrame()
    if region_consensus.empty:
        st.info(tr("No consensus deviation snapshot is available for this region yet."))
    else:
        latest_consensus = region_consensus.sort_values("snapshot_date").iloc[-1]
        consensus_left, consensus_mid, consensus_right = st.columns(3)
        consensus_left.metric(tr("Source Count"), int(latest_consensus.get("source_count", 0)))
        consensus_mid.metric(
            tr("Consensus Date"),
            format_display_value(
                pd.Timestamp(latest_consensus["snapshot_date"]).date().isoformat()
                if pd.notna(latest_consensus.get("snapshot_date"))
                else None
            ),
        )
        consensus_right.metric(
            tr("Consensus Confidence"),
            humanize_label(latest_consensus.get("consensus_confidence")),
        )
        st.caption(
            tr(
                "Latest consensus note date: {date}",
                date=format_display_value(
                    pd.Timestamp(latest_consensus["latest_consensus_note_date"]).date().isoformat()
                    if pd.notna(latest_consensus.get("latest_consensus_note_date"))
                    else None
                ),
            )
        )
        st.caption(translate_runtime_text(latest_consensus.get("deviation_summary")))
        st.caption(translate_runtime_text(latest_consensus.get("deviation_reason")))
        comparison_table = pd.DataFrame(
            [
                {
                    tr("Dimension"): humanize_label("growth"),
                    tr("Consensus View"): humanize_label(latest_consensus.get("growth_consensus")),
                    tr("Model View"): humanize_label(latest_consensus.get("model_growth_view")),
                    tr("Deviation Score"): latest_consensus.get("growth_deviation_score"),
                },
                {
                    tr("Dimension"): humanize_label("inflation"),
                    tr("Consensus View"): humanize_label(latest_consensus.get("inflation_consensus")),
                    tr("Model View"): humanize_label(latest_consensus.get("model_inflation_view")),
                    tr("Deviation Score"): latest_consensus.get("inflation_deviation_score"),
                },
                {
                    tr("Dimension"): humanize_label("policy_bias"),
                    tr("Consensus View"): humanize_label(latest_consensus.get("policy_bias_consensus")),
                    tr("Model View"): humanize_label(latest_consensus.get("model_policy_bias_view")),
                    tr("Deviation Score"): latest_consensus.get("policy_deviation_score"),
                },
            ]
        )
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    st.subheader(tr("Latest Asset Preferences"))
    asset_snapshot = pd.DataFrame(
        [
            {tr("Asset"): humanize_country_asset_label(country, "equities"), tr("Preference"): humanize_label(latest_assets.get("equities")), tr("Score"): latest_assets.get("equities_score", float("nan"))},
            {tr("Asset"): humanize_country_asset_label(country, "duration"), tr("Preference"): humanize_label(latest_assets.get("duration")), tr("Score"): latest_assets.get("duration_score", float("nan"))},
            {tr("Asset"): humanize_country_asset_label(country, "gold"), tr("Preference"): humanize_label(latest_assets.get("gold")), tr("Score"): latest_assets.get("gold_score", float("nan"))},
            {tr("Asset"): humanize_country_asset_label(country, "dollar"), tr("Preference"): humanize_label(latest_assets.get("dollar")), tr("Score"): latest_assets.get("dollar_score", float("nan"))},
        ]
    )
    st.dataframe(asset_snapshot, use_container_width=True, hide_index=True)
    if "allocation_confidence" in latest_assets.index:
        st.caption(
            tr(
                "Allocation confidence: {level}. {note}",
                level=humanize_label(latest_assets.get("allocation_confidence")),
                note=translate_runtime_text(latest_assets.get("allocation_note")),
            )
        )

    st.subheader(tr("Growth Score"))
    st.plotly_chart(px.line(regime_data, x="date", y="growth_score"), use_container_width=True)
    st.subheader(tr("Inflation Score"))
    st.plotly_chart(px.line(regime_data, x="date", y="inflation_score"), use_container_width=True)
    st.subheader(tr("Liquidity Score"))
    st.plotly_chart(px.line(regime_data, x="date", y="liquidity_score"), use_container_width=True)

    if not valuation_data.empty and "valuation_score" in valuation_data.columns:
        st.subheader(tr("Valuation Score"))
        st.plotly_chart(
            px.line(valuation_data, x="date", y="valuation_score"),
            use_container_width=True,
        )

    st.subheader(tr("Recent Asset Preference History"))
    st.dataframe(asset_data.tail(12), use_container_width=True)


def render_global_view() -> None:
    """Render the dashboard for the global summary."""
    summary = load_csv_with_dates(
        Path("data/processed/global_macro_summary.csv"),
        "python main.py build-global-summary",
    )
    allocation = load_csv_with_dates(
        Path("data/processed/global_allocation_map.csv"),
        "python main.py build-global-allocation",
    )
    regime_frequency = load_csv_with_dates(
        Path("data/processed/regime_frequency_summary.csv"),
        "python main.py evaluate-regimes",
    )
    transition_matrix = load_csv_with_dates(
        Path("data/processed/regime_transition_matrix.csv"),
        "python main.py evaluate-regimes",
    )
    forward_summary = load_csv_with_dates(
        Path("data/processed/regime_forward_return_summary.csv"),
        "python main.py evaluate-regimes",
    )
    confidence_summary = load_csv_with_dates(
        Path("data/processed/confidence_bucket_summary.csv"),
        "python main.py evaluate-confidence",
    )
    consensus_deviation = load_optional_csv(Path("data/processed/consensus_deviation.csv"))
    st.title(tr("Global Macro Monitor"))
    st.caption(tr("Weighted summary across US, China, and Eurozone."))

    if summary.empty:
        st.warning(tr("No global summary data available yet."))
        return

    mode = st.selectbox(
        tr("Global mode"),
        options=["latest_available", "last_common_date"],
        format_func=humanize_label,
    )
    mode_summary = summary.loc[summary["as_of_mode"] == mode].reset_index(drop=True)
    if mode_summary.empty:
        st.warning(tr("No global summary data available for the selected mode yet."))
        return
    latest = mode_summary.iloc[-1]
    status_table = build_country_status(mode=mode)
    country_regime_dates = {
        country: latest.get(f"{country}_latest_date") for country in get_supported_countries()
    }
    global_overlay = build_global_nowcast_overlay(latest.get("summary_date"), country_regime_dates)
    system_update_date = _format_optional_date(global_overlay["system_update_timestamp"])
    market_input_date = _format_optional_date(global_overlay["freshest_market_date"])

    if latest.get("global_regime") == "partial_view":
        st.warning(translate_runtime_text(latest.get("coverage_warning")))
    top_left, top_mid, top_right, top_far_right = st.columns(4)
    top_left.metric(tr("Selected Mode"), humanize_label(mode))
    top_mid.metric(tr("System Update Date"), system_update_date)
    top_right.metric(tr("Data Through Date"), format_display_value(
        pd.Timestamp(latest["summary_date"]).date().isoformat()
        if pd.notna(latest.get("summary_date"))
        else None
    ))
    top_far_right.metric(tr("Latest Global Regime"), humanize_label(latest.get("global_regime")))
    metric_left, metric_mid, metric_right = st.columns(3)
    metric_left.metric(
        tr("Investment Clock"),
        humanize_label(latest.get("investment_clock", latest.get("global_investment_clock"))),
    )
    metric_mid.metric(tr("Coverage Ratio"), f"{float(latest.get('coverage_ratio', 0.0)):.2f}")
    metric_right.metric(tr("Latest Market-sensitive Input"), market_input_date)

    st.subheader(tr("Nowcast Overlay"))
    if global_overlay["countries_with_newer_inputs"]:
        st.info(
            tr(
                "Global macro state is based on data through {data_date}, while fresher market-sensitive inputs are available through {market_date} ({details}).",
                data_date=_format_optional_date(latest.get("summary_date")),
                market_date=market_input_date,
                details=", ".join(
                    f"{humanize_label(item.split(':', 1)[0])}:{humanize_label(item.split(':', 1)[1])}"
                    for item in global_overlay["freshest_market_sources"]
                ),
            )
        )
    else:
        st.caption(
            tr(
                "The system was refreshed on {system_date}. The macro regime is still based on data through {data_date}.",
                system_date=system_update_date,
                data_date=_format_optional_date(latest.get("summary_date")),
            )
        )
        st.caption(tr("Global macro state and the latest market-sensitive inputs are aligned at the same date."))

    coverage_left, coverage_mid = st.columns(2)
    coverage_left.metric(tr("Available Countries"), format_country_list(latest.get("countries_available")))
    coverage_mid.metric(tr("Missing Countries"), format_country_list(latest.get("countries_missing")))

    st.caption(tr("A country can be Ready locally but still not usable in the selected global view if it has no valid data on that mode's evaluation date."))

    st.subheader(tr("Weighting"))
    st.markdown(tr("Configured weights: {weights}", weights=format_weight_map(latest.get("configured_weights"))))
    st.markdown(tr("Effective weights: {weights}", weights=format_weight_map(latest.get("effective_weights"))))
    available_countries = [item for item in str(latest.get("countries_available", "")).split(",") if item]
    if len(available_countries) == 1:
        st.info(tr("Only one country is available in the selected mode, so the effective global view is entirely driven by that market."))

    st.subheader(tr("Country Status"))
    display_status = status_table.copy()
    if not display_status.empty:
        drop_columns = [column for column in ["summary_date"] if column in display_status.columns]
        if drop_columns:
            display_status = display_status.drop(columns=drop_columns)
        display_status["country"] = display_status["country"].apply(humanize_label)
        display_status["regime"] = display_status["regime"].apply(humanize_label)
        display_status["country_ready"] = display_status["country_ready"].map({True: "Yes", False: "No"})
        display_status["globally_usable_latest"] = display_status["globally_usable_latest"].map({True: "Yes", False: "No"})
        display_status["latest_date"] = display_status["latest_date"].apply(
            lambda value: format_display_value(
                pd.Timestamp(value).date().isoformat() if pd.notna(value) else None
            )
        )
        display_status["staleness_status"] = display_status["staleness_status"].apply(humanize_label)
        display_status["valuation_status"] = display_status["valuation_status"].apply(humanize_label)
        display_status = display_status.rename(
            columns={
                "country": tr("Country"),
                "regime": tr("Regime"),
                "country_ready": tr("Country Ready"),
                "globally_usable_latest": tr("Usable in Selected Mode"),
                "latest_date": tr("Latest Date"),
                "days_stale": tr("Days Stale"),
                "staleness_status": tr("Staleness Status"),
                "valuation_status": tr("Valuation Status"),
            }
        )
    st.dataframe(display_status, use_container_width=True, hide_index=True)

    stale_countries = display_status.loc[
        display_status[tr("Staleness Status")].isin([humanize_label("stale"), humanize_label("very_stale")]),
        tr("Country"),
    ].tolist() if not display_status.empty else []
    if stale_countries:
        st.warning(tr("Stale country data detected: {countries}", countries=", ".join(stale_countries)))

    st.subheader(tr("Global Allocation"))
    mode_allocation = allocation.loc[allocation["as_of_mode"] == mode].copy()
    if mode_allocation.empty:
        st.info(tr("No global allocation map available for the selected mode yet."))
        return
    latest_allocation_date = mode_allocation["date"].max()
    latest_allocation = mode_allocation.loc[mode_allocation["date"] == latest_allocation_date].copy()
    mode_context = (
        format_display_value(latest_allocation.iloc[0].get("mode_context"))
        if not latest_allocation.empty and "mode_context" in latest_allocation.columns
        else tr("No data")
    )
    if mode_context != tr("No data"):
        st.caption(translate_runtime_text(mode_context))
    latest_allocation["asset"] = latest_allocation["asset"].apply(humanize_global_asset_label)
    latest_allocation["preference"] = latest_allocation["preference"].apply(humanize_label)
    latest_allocation["confidence"] = latest_allocation["confidence"].apply(humanize_label)
    latest_allocation["score"] = latest_allocation["score"].map(lambda value: f"{float(value):.1f}")
    latest_allocation["reason"] = latest_allocation["reason"].apply(translate_runtime_text)
    latest_allocation = latest_allocation.rename(
        columns={
            "asset": tr("Asset"),
            "preference": tr("Preference"),
            "score": tr("Score"),
            "confidence": tr("Confidence"),
            "reason": tr("Reason"),
        }
    )
    st.dataframe(
        latest_allocation.loc[:, [tr("Asset"), tr("Preference"), tr("Score"), tr("Confidence"), tr("Reason")]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        tr(
            "Reason explains the asset view itself. Confidence falls when country coverage is thin, country data is stale, or valuation inputs are missing."
        )
    )

    st.subheader(tr("Consensus Deviation"))
    if consensus_deviation.empty:
        st.write(tr("No consensus deviation table"))
    else:
        deviation_table = consensus_deviation.copy()
        deviation_table["region"] = deviation_table["region"].apply(humanize_label)
        deviation_table["deviation_summary"] = deviation_table["deviation_summary"].apply(translate_runtime_text)
        for column in ["growth_deviation_score", "inflation_deviation_score", "policy_deviation_score", "consensus_deviation_score"]:
            if column in deviation_table.columns:
                deviation_table[column] = deviation_table[column].map(
                    lambda value: f"{float(value):.2f}" if pd.notna(value) else tr("No data")
                )
        deviation_table = deviation_table.rename(
            columns={
                "region": tr("Country"),
                "consensus_deviation_score": tr("Consensus Deviation Score"),
                "growth_deviation_score": tr("Growth Score"),
                "inflation_deviation_score": tr("Inflation Score"),
                "policy_deviation_score": tr("Policy Deviation Score"),
                "deviation_summary": tr("Summary"),
            }
        )
        st.dataframe(
            deviation_table.loc[:, [
                tr("Country"),
                tr("Consensus Deviation Score"),
                tr("Growth Score"),
                tr("Inflation Score"),
                tr("Policy Deviation Score"),
                tr("Summary"),
            ]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(tr("What Changed"))
    comparison = build_mode_comparison(selected_mode=mode)
    prepared = prepare_what_changed_sections(comparison)
    current_snapshot_time = (
        pd.Timestamp(prepared["current_snapshot_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        if pd.notna(prepared["current_snapshot_timestamp"])
        else tr("No data")
    )
    prior_snapshot_time = (
        pd.Timestamp(prepared["prior_snapshot_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        if pd.notna(prepared["prior_snapshot_timestamp"])
        else tr("No data")
    )
    st.caption(tr("Current snapshot time: {time}", time=current_snapshot_time))
    st.caption(tr("Prior snapshot time: {time}", time=prior_snapshot_time))
    st.caption(prepared["message"])

    regime_changes = prepared["regime_changes"]
    preference_changes = prepared["preference_changes"]
    confidence_changes = prepared["confidence_changes"]

    summary_left, summary_mid, summary_right = st.columns(3)
    summary_left.metric(tr("Regime Changes"), prepared["regime_change_count"])
    summary_mid.metric(tr("Preference Changes"), prepared["preference_change_count"])
    summary_right.metric(tr("Confidence Changes"), prepared["confidence_change_count"])

    st.markdown(f"**{tr('Why It Changed')}**")
    reason_summaries = prepared["why_it_changed"]
    if reason_summaries:
        for reason in reason_summaries:
            st.write(f"- {humanize_label(reason)}")
    else:
        st.write(f"- {tr('No clear change driver was recorded.')}")

    st.markdown(f"**{tr('Regime Changes')}**")
    if not prepared["available"] or regime_changes.empty:
        st.write(tr("No regime changes"))
    else:
        for _, row in regime_changes.iterrows():
            field_label = "regime" if row["entity_type"] == "country_regime" else "changed"
            st.write(f"- {format_change_sentence(row['entity_name'], field_label, row['old_value'], row['new_value'])}")

    st.markdown(f"**{tr('Preference Changes')}**")
    if not prepared["available"] or preference_changes.empty:
        st.write(tr("No preference changes"))
    else:
        for _, row in preference_changes.iterrows():
            st.write(
                f"- {format_change_sentence(row['entity_name'], 'preference', row['old_value'], row['new_value'])}"
            )

    st.markdown(f"**{tr('Confidence Changes')}**")
    if not prepared["available"] or confidence_changes.empty:
        st.write(tr("No confidence changes"))
    else:
        for _, row in confidence_changes.iterrows():
            st.write(
                f"- {format_change_sentence(row['entity_name'], 'confidence', row['old_value'], row['new_value'])}"
            )

    st.subheader(tr("Regime Evaluation"))
    mode_frequency = regime_frequency.loc[regime_frequency["selected_mode"] == mode].copy()
    mode_transitions = transition_matrix.loc[transition_matrix["selected_mode"] == mode].copy()
    mode_forward = forward_summary.loc[forward_summary["selected_mode"] == mode].copy()
    mode_confidence = confidence_summary.loc[confidence_summary["selected_mode"] == mode].copy()

    st.markdown(f"**{tr('Regime Frequency')}**")
    if mode_frequency.empty:
        st.write(tr("No regime frequency summary"))
    else:
        mode_frequency = mode_frequency.rename(
            columns={
                "regime": tr("Regime"),
                "count": tr("Count"),
                "selected_mode": tr("Selected Mode"),
            }
        )
        st.dataframe(mode_frequency, use_container_width=True, hide_index=True)

    st.markdown(f"**{tr('Transition Matrix')}**")
    if mode_transitions.empty:
        st.write(tr("No transition matrix"))
    else:
        pivot = mode_transitions.pivot(
            index="from_regime",
            columns="to_regime",
            values="count",
        ).fillna(0)
        pivot.index = [humanize_label(value) for value in pivot.index]
        pivot.columns = [humanize_label(value) for value in pivot.columns]
        st.dataframe(pivot, use_container_width=True)

    st.markdown(f"**{tr('Forward Return Summary')}**")
    if mode_forward.empty:
        st.write(tr("No forward return summary"))
    else:
        mode_forward = mode_forward.rename(
            columns={
                "asset": tr("Asset"),
                "regime": tr("Regime"),
                "window": tr("Window"),
                "count": tr("Count"),
                "average_forward_return": tr("Average Forward Return"),
                "median_forward_return": tr("Median Forward Return"),
                "hit_ratio": tr("Hit Ratio"),
            }
        )
        st.dataframe(mode_forward.head(24), use_container_width=True, hide_index=True)

    st.markdown(f"**{tr('Confidence Bucket Summary')}**")
    if mode_confidence.empty:
        st.write(tr("No confidence bucket summary"))
    else:
        mode_confidence = mode_confidence.rename(
            columns={
                "confidence": tr("Confidence"),
                "asset": tr("Asset"),
                "count": tr("Count"),
                "average_forward_return": tr("Average Forward Return"),
                "median_forward_return": tr("Median Forward Return"),
                "hit_ratio": tr("Hit Ratio"),
            }
        )
        st.dataframe(mode_confidence.head(24), use_container_width=True, hide_index=True)


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="Global Macro Monitor", layout="wide")
    current_language = st.sidebar.selectbox(
        tr("Language", language="en"),
        options=list(LANGUAGE_OPTIONS.values()),
        format_func=lambda value: "中文" if value == "zh" else "English",
        index=0 if get_display_language() == "zh" else 1,
    )
    set_display_language(current_language)
    options = ["global"] + get_supported_countries()
    selection = st.sidebar.selectbox(
        tr("View"),
        options=options,
        format_func=lambda value: tr("Global") if value == "global" else humanize_label(value),
    )

    try:
        if selection == "global":
            render_global_view()
        else:
            render_country_view(selection)
    except FileNotFoundError as exc:
        st.warning(str(exc))


if __name__ == "__main__":
    main()
