"""Streamlit dashboard entry point."""

from __future__ import annotations

import html
from pathlib import Path
import re
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from app.regime.change_detection import HISTORY_DIR as CHANGE_HISTORY_DIR
from app.regime.global_monitor import build_country_status
from app.regime.nowcast import build_country_nowcast_overlay, build_global_nowcast_overlay
from app.utils.config import get_country_config, get_supported_countries
from app.data.china_ingestion import validate_china_data
from app.data.eurozone_ingestion import validate_eurozone_data


ACTIVE_LANGUAGE = "zh"
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
    "dxy_proxy": "Dollar Index Proxy",
    "vix_proxy": "VIX Proxy",
    "gold_proxy": "Gold Proxy",
    "oil_proxy": "Oil Proxy",
    "copper_proxy": "Copper Proxy",
    "sp500_proxy": "S&P 500 Proxy",
    "eurostoxx50_proxy": "Euro Stoxx 50 Proxy",
    "china_equity_proxy": "China Equity Proxy",
    "risk_on": "Risk-on",
    "defensive": "Defensive",
    "stable": "Stable",
    "easing": "Easing",
    "tightening": "Tightening",
    "cheaper": "Cheaper",
    "richer": "Richer",
    "cooling": "Cooling",
    "reheating": "Reheating",
    "weaker_dollar": "Weaker dollar",
    "stronger_dollar": "Stronger dollar",
    "tighter_spreads": "Tighter spreads",
    "wider_spreads": "Wider spreads",
    "lower_volatility": "Lower volatility",
    "higher_volatility": "Higher volatility",
    "stronger": "Stronger",
    "weaker": "Weaker",
    "risk": "Risk",
    "rates": "Rates",
    "Risk Overlay": "Risk Overlay",
    "Rates Overlay": "Rates Overlay",
    "Inflation Overlay": "Inflation Overlay",
    "cpi": "CPI",
    "pmi": "PMI",
    "industrial_production": "Industrial Production",
    "unrate": "Unemployment Rate",
    "buffett_indicator": "Buffett Indicator",
    "earnings_yield_proxy": "Earnings Yield Proxy",
    "equity_pe_proxy": "Equity PE Proxy",
    "shiller_pe_proxy": "Shiller CAPE Proxy",
    "equity_pb_proxy": "Equity PB Proxy",
    "credit_spread_proxy": "Corporate OAS Proxy",
    "ecb": "ECB",
    "eurostat": "Eurostat",
    "eurostat_flash": "Eurostat Flash",
    "china_akshare": "China Akshare",
    "china_nbs": "China NBS",
    "china_rates": "China Rates",
    "normalized_api": "Normalized API",
    "siblis": "Siblis",
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
    "dxy_proxy": "美元指数代理",
    "vix_proxy": "波动率代理",
    "gold_proxy": "黄金价格代理",
    "oil_proxy": "原油价格代理",
    "copper_proxy": "铜价代理",
    "sp500_proxy": "标普500代理",
    "eurostoxx50_proxy": "欧元区蓝筹股代理",
    "china_equity_proxy": "中国权益资产代理",
    "risk_on": "偏风险偏好",
    "Risk-on": "偏风险偏好",
    "defensive": "偏防御",
    "Defensive": "偏防御",
    "stable": "稳定",
    "easing": "边际宽松",
    "tightening": "边际收紧",
    "cheaper": "估值走低",
    "richer": "估值走高",
    "cooling": "通胀降温",
    "reheating": "通胀再升温",
    "weaker_dollar": "美元走弱",
    "stronger_dollar": "美元走强",
    "tighter_spreads": "信用利差收窄",
    "wider_spreads": "信用利差走阔",
    "lower_volatility": "波动率回落",
    "higher_volatility": "波动率上升",
    "stronger": "走强",
    "weaker": "走弱",
    "risk": "风险偏好",
    "rates": "利率",
    "Risk Overlay": "风险偏好偏移",
    "Rates Overlay": "利率偏移",
    "Inflation Overlay": "通胀偏移",
    "cpi": "CPI",
    "pmi": "PMI",
    "industrial_production": "工业增加值",
    "unrate": "失业率",
    "buffett_indicator": "巴菲特指标",
    "earnings_yield_proxy": "盈利收益率代理",
    "equity_pe_proxy": "权益市盈率代理",
    "shiller_pe_proxy": "席勒CAPE代理",
    "equity_pb_proxy": "权益市净率代理",
    "credit_spread_proxy": "公司债OAS代理",
    "ecb": "ECB",
    "eurostat": "Eurostat",
    "eurostat_flash": "Eurostat 快报",
    "china_akshare": "AkShare",
    "china_nbs": "国家统计局",
    "china_rates": "中国货币网/债券数据",
    "normalized_api": "标准化API数据",
    "global_markets": "全球市场数据",
    "siblis": "Siblis",
    "No loaded data": "未加载数据",
}

TEXT_MAP_ZH = {
    "No data": "无数据",
    "View": "视图",
    "Language": "语言",
    "User Guide": "用户手册",
    "Open User Guide": "用户手册",
    "Back to Dashboard": "返回监控页面",
    "Global Snapshot": "全球快照",
    "Primary Signals": "核心信号",
    "Asset Tilt Board": "资产偏好总览",
    "Regional Quick Read": "地区速览",
    "Detailed Tables": "详细表格",
    "Macro regime and market overlay are aligned.": "宏观主状态与实时市场偏移方向大体一致。",
    "Macro regime and market overlay are diverging.": "宏观主状态与实时市场偏移方向出现分化，值得重点跟踪。",
    "Current core view": "当前核心判断",
    "Current Core View": "当前核心判断",
    "Current Global Macro Regime": "当前全球宏观阶段",
    "Open to understand how to read the model, scores, confidence, and allocation views.": "展开后可查看模型、评分、置信度与资产偏好的阅读方法。",
    "Detailed guide to the model, scoring system, confidence, consensus deviation, and asset mapping.": "这是一份关于模型、评分体系、置信度、共识偏差和资产映射逻辑的详细说明。",
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
    "Nowcast Score": "实时偏移评分",
    "Nowcast Direction": "实时偏移方向",
    "Driver Summary": "主要驱动",
    "Growth Score": "增长评分",
    "Inflation Score": "通胀评分",
    "Liquidity Score": "流动性评分",
    "Regime Confidence": "阶段置信度",
    "Raw Regime": "原始阶段判定",
    "Valuation Regime": "估值状态",
    "Valuation Score": "估值评分",
    "Valuation Confidence": "估值置信度",
    "Valuation Inputs Used": "估值已用输入",
    "Valuation Inputs Missing": "估值缺失输入",
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
    "Monitor Alerts": "重点提醒",
    "Severity": "级别",
    "Headline": "提醒",
    "Detail": "说明",
    "No active alerts were generated for the selected mode.": "当前所选模式下暂未生成需要重点关注的提醒。",
    "No active alerts were generated for this region.": "该地区当前暂无需要重点关注的提醒。",
    "Short reason": "简要理由",
    "Full rationale by asset": "分资产查看完整理由",
    "No detailed rationale is available.": "当前没有可展示的完整理由。",
    "State Type": "阶段类型",
    "State": "阶段值",
    "Share": "占比",
    "Read This First": "先看这个",
    "Four Macro States": "四种宏观阶段",
    "Core Meaning": "核心含义",
    "Typical Overweight": "典型超配",
    "Typical Neutral": "典型中性",
    "Typical Underweight": "典型低配",
    "How to Read Asset Preferences": "如何理解资产偏好",
    "How to Read Scores": "如何理解评分",
    "How to Read Confidence": "如何理解置信度",
    "How to Read Consensus Deviation": "如何理解共识偏差",
    "How to Read the Nowcast Overlay": "如何理解实时偏移层",
    "Practical Reading Order": "建议阅读顺序",
    "Core Scores": "核心评分",
    "Valuation": "估值概览",
    "Growth Momentum": "增长动能",
    "Inflation Pressure": "通胀压力",
    "Higher": "更高",
    "Lower": "更低",
    "This system is a macro monitoring and allocation support framework. It first answers where the cycle is, then asks which assets are more suitable to overweight, hold neutral, or underweight.": "这套系统是一个宏观监控与资产配置支持框架。它先回答当前周期大致处于什么位置，再回答哪些资产更适合超配、维持中性或低配。",
    "Growth is stable, inflation is soft, and liquidity is not obviously tightening.": "增长平稳，通胀温和，流动性未明显收紧。",
    "Global equities, US equities, Eurozone equities": "全球权益资产、美国权益资产、欧元区权益资产",
    "USD duration, commodities": "美元久期、大宗商品",
    "Gold, dollar": "黄金、美元",
    "Growth improves, inflation rises, and nominal activity re-accelerates.": "增长改善、通胀回升、名义增长抬升。",
    "Equities, commodities": "权益资产、大宗商品",
    "Dollar, gold": "美元、黄金",
    "USD duration": "美元久期",
    "Growth weakens, inflation softens, and risk appetite fades.": "增长回落、通胀回落、风险偏好下降。",
    "Equities, commodities": "权益资产、大宗商品",
    "Growth is weak while inflation remains firm; it is one of the least friendly combinations.": "增长走弱但通胀偏高，是最不友好的组合之一。",
    "Gold, some commodities": "黄金、部分大宗商品",
    "The macro-plus-valuation mix supports overweight positioning.": "宏观与估值组合支持超配判断。",
    "Signals are mixed, or macro and valuation offset each other.": "信号偏分化，或宏观与估值相互抵消。",
    "The macro-plus-valuation mix argues for underweight positioning.": "宏观与估值组合更支持低配判断。",
    "Higher means stronger growth.": "数值越高，代表增长越强。",
    "Higher means more inflation pressure.": "数值越高，代表通胀压力越大。",
    "Higher means easier financial conditions.": "数值越高，代表金融条件越宽松。",
    "Higher means cheaper and more supportive conditions.": "数值越高，代表估值越便宜、环境越支持风险资产。",
    "These are transparent standardized scores, not black-box probabilities.": "这些都是透明的标准化评分，不是黑箱概率。",
    "Broad coverage, fresh data, and stronger valuation support.": "覆盖较广、数据较新、估值支持更充分。",
    "Core pipeline works, but some valuation or enrichment inputs are incomplete or stale.": "主链路可用，但部分估值或增强输入仍不完整或偏滞后。",
    "Stale data, thin coverage, or weaker valuation and enrichment coverage.": "数据滞后、覆盖偏薄，或估值与增强输入覆盖较弱。",
    "Consensus deviation compares the model view versus mainstream public narrative across growth, inflation, and policy bias. A positive growth deviation means the model is more growth-positive than consensus; a negative one means the model is more growth-negative.": "共识偏差比较的是模型判断与主流公开叙事在增长、通胀和政策倾向三个维度上的差异。增长偏离为正，代表模型比共识更偏增长乐观；为负则代表模型更偏增长悲观。",
    "A positive inflation deviation means the model sees less inflation risk than consensus. A positive policy deviation means the model is more dovish than consensus.": "通胀偏离为正，代表模型认为通胀风险低于当前共识；政策偏离为正，则代表模型比当前共识更偏鸽派。",
    "Tracks risk-sensitive inputs such as equities, volatility, and the dollar.": "跟踪权益、波动率、美元等风险敏感输入。",
    "Tracks policy rates and longer-term yields.": "跟踪政策利率和长期收益率变化。",
    "Tracks inflation-sensitive inputs such as CPI-style data, gold, and oil.": "跟踪 CPI 类数据、黄金、原油等通胀敏感输入。",
    "The nowcast overlay does not replace the macro model. It is a higher-frequency layer that shows whether market-sensitive inputs are already moving ahead of the monthly macro regime.": "实时偏移层不会替代主宏观模型。它是一层更高频的补充，用来判断市场敏感输入是否已经先于月频宏观状态发生变化。",
    "The chart above is the base framework rather than a mechanical rule. Final preferences still depend on liquidity, valuation, and confidence.": "上图展示的是基础框架，不是机械指令。最终资产偏好仍会受到流动性环境、估值状态和数据置信度影响。",
    "Positive score: higher-frequency inputs are leaning more risk-on / easier than the monthly macro regime.": "正值：代表更高频的市场输入比月频宏观状态更偏向风险偏好或更偏宽松。",
    "Negative score: higher-frequency inputs are leaning more defensive / tighter than the monthly macro regime.": "负值：代表更高频的市场输入比月频宏观状态更偏向防御或更偏收紧。",
    "Near zero: signals are mixed and do not yet point to a clear short-term tilt.": "接近 0：代表高频信号仍偏分化，尚未指向明确的短期偏移方向。",
    "Check whether the macro regime changed.": "先看宏观阶段有没有变化。",
    "Check whether scores crossed the neutral zone.": "再看评分是否跨过中性区间。",
    "Check whether the nowcast overlay shows a higher-frequency shift.": "然后看实时偏移层是否已经先给出更高频的变化信号。",
    "Check whether valuation supports or offsets the macro view.": "再确认估值是在支持还是抵消宏观判断。",
    "Check whether asset preferences changed direction.": "再看资产偏好有没有改变方向。",
    "Check whether the model materially diverges from consensus.": "最后看模型判断是否与市场共识出现明显偏离。",
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
    "Valuation confidence: {level}": "估值置信度：{level}",
    "Valuation inputs used: {items}": "估值已用输入：{items}",
    "Valuation inputs still missing: {items}": "估值仍缺失输入：{items}",
    "Macro and valuation inputs are both available with strong coverage.": "宏观与估值输入均已具备，且估值覆盖较完整。",
    "Macro and valuation inputs are available, but valuation coverage is still partial.": "宏观与估值输入已具备，但估值覆盖仍然不完整。",
    "Valuation is missing, so local allocation is still produced with reduced confidence.": "估值输入缺失，因此本地资产偏好仍会生成，但置信度会下调。",
    "The system was refreshed on {system_date}. The macro regime is still based on data through {data_date}.": "系统已于 {system_date} 刷新，但宏观阶段仍基于截至 {data_date} 的数据。",
    "{country} has fresher market-sensitive inputs through {market_date} ({series}), while the macro regime still runs through {data_date}.": "{country} 的市场敏感输入已更新到 {market_date}（{series}），但宏观阶段仍截至 {data_date}。",
    "No fresher market-sensitive inputs were found beyond the current macro snapshot.": "当前没有比主模型更晚的市场敏感输入。",
    "Global macro state is based on data through {data_date}, while fresher market-sensitive inputs are available through {market_date} ({details}).": "全球宏观主状态截至 {data_date}，但更高频的市场敏感输入已更新到 {market_date}（{details}）。",
    "Global macro state and the latest market-sensitive inputs are aligned at the same date.": "全球宏观主状态与最新市场敏感输入日期一致。",
    "Nowcast overlay is neutral.": "实时偏移层目前保持中性。",
    "Nowcast overlay tilts risk-on.": "实时偏移层目前偏向风险偏好。",
    "Nowcast overlay tilts defensive.": "实时偏移层目前偏向防御。",
}
REGION_MAP_ZH = {"North America": "北美", "Europe": "欧洲", "Asia": "亚洲"}
FREE_TEXT_LABEL_MAP_ZH = {
    "Goldilocks": "温和增长",
    "Reflation": "再通胀",
    "Stagflation": "滞胀",
    "Slowdown": "增长放缓",
    "Risk-on": "偏风险偏好",
    "Defensive": "偏防御",
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


def inject_dashboard_styles() -> None:
    """Inject lightweight dashboard styles for card-based sections."""
    st.markdown(
        """
        <style>
        :root {
            --terminal-bg: #0b0d12;
            --terminal-panel: #12161d;
            --terminal-panel-2: #171c24;
            --terminal-border: rgba(214, 154, 46, 0.28);
            --terminal-border-soft: rgba(255, 255, 255, 0.08);
            --terminal-accent: #f0b90b;
            --terminal-accent-soft: rgba(240, 185, 11, 0.16);
            --terminal-text: #eef2f7;
            --terminal-muted: #9aa5b5;
            --terminal-negative: #d3725f;
            --terminal-positive: #8dbb6a;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(240,185,11,0.06), transparent 16%),
                linear-gradient(180deg, #0b0d12 0%, #0f1319 100%);
        }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(240,185,11,0.07), transparent 10%),
                linear-gradient(180deg, #11151b 0%, #0c1016 100%);
            border-right: 1px solid rgba(240,185,11,0.10);
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #0e1218 !important;
            border: 1px solid rgba(240,185,11,0.22) !important;
            border-radius: 14px !important;
        }
        h1, h2, h3 {
            color: var(--terminal-text);
            letter-spacing: 0.01em;
        }
        h2, h3 {
            padding-top: 0.2rem;
            border-top: 1px solid rgba(240,185,11,0.14);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(20,24,31,0.98), rgba(13,16,22,0.98));
            border: 1px solid var(--terminal-border-soft);
            border-top: 2px solid rgba(240,185,11,0.50);
            border-radius: 14px;
            padding: 0.95rem 1rem 0.8rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        [data-testid="stMetricLabel"] {
            color: var(--terminal-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: var(--terminal-text);
            font-family: "SFMono-Regular", "Menlo", "Monaco", monospace;
            letter-spacing: -0.02em;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(21,25,32,0.96), rgba(12,15,21,0.96));
            border: 1px solid var(--terminal-border-soft);
            border-radius: 16px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding-top: 0.1rem;
        }
        div[data-testid="stCaptionContainer"] p {
            color: var(--terminal-muted);
        }
        div[data-testid="stMarkdownContainer"] p {
            line-height: 1.55;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--terminal-border-soft);
            border-radius: 14px;
            overflow: hidden;
        }
        [data-testid="stDataFrame"] [role="grid"] {
            background: #0f1319;
        }
        [data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid rgba(240,185,11,0.18);
        }
        .stButton > button {
            background: linear-gradient(180deg, #13171e, #0f1319);
            color: var(--terminal-text);
            border: 1px solid rgba(240,185,11,0.28);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_user_manual_markdown(language: str | None = None) -> str:
    """Return a professional in-app manual for reading the monitor."""
    lang = get_display_language(language)
    if lang == "zh":
        return """
这套系统首先回答两个问题：  
第一，当前宏观阶段处在什么位置。  
第二，这种宏观与估值组合下，哪些资产更适合超配、维持中性或低配。

## 一、四种宏观阶段先看这个

| 宏观阶段 | 核心含义 | 典型超配 | 典型中性 | 典型低配 |
| --- | --- | --- | --- | --- |
| 温和增长 | 增长平稳、通胀温和、流动性不明显收紧 | 全球权益资产、美国权益资产、欧元区权益资产 | 美元久期、大宗商品 | 黄金、美元 |
| 再通胀 | 增长改善、通胀回升、名义增长抬升 | 权益资产、大宗商品 | 美元、黄金 | 美元久期 |
| 增长放缓 | 增长回落、通胀回落、风险偏好下降 | 美元久期 | 黄金、美元 | 权益资产、大宗商品 |
| 滞胀 | 增长走弱但通胀偏高，是最不友好的组合之一 | 黄金、部分大宗商品 | 美元 | 权益资产、美元久期 |

上表是典型框架，不是机械指令。  
最终页面里的资产偏好还会受到流动性环境、估值状态和数据置信度影响。

## 二、资产偏好标签怎么读

### 超配
代表当前宏观阶段、流动性和估值组合对该资产相对有利。  
这不是价格保证，而是相对配置倾向更偏正面。

### 中性
代表信号并不集中，或者宏观结论与估值结论互相抵消。  
中性通常意味着不急于做强方向判断。

### 低配
代表当前宏观阶段、流动性或估值组合对该资产相对不利。  
低配也不是一定下跌，而是相对吸引力较弱。

## 三、全球页里每个资产到底代表什么

### 全球权益资产
用于表达全球汇总口径下的权益偏好，是全球风险资产方向的核心指标。

### 美元久期
全球页里的久期采用美国本地利率与久期框架判断，因为全球久期定价通常仍以美元利率周期为主导。

### 黄金
主要反映通胀、防御、滞胀和避险逻辑，不完全等同于单一地区的本地判断。

### 美元
表示在当前宏观与流动性背景下，对美元资产或美元敞口的偏好。

### 大宗商品
主要反映增长、再通胀和供给冲击的综合判断。

### 美国 / 中国 / 欧元区权益资产
表示在各地区本地宏观框架下，对当地权益资产的配置倾向。

## 四、地区页应该怎么读

每个地区页都建议按同样顺序看：

1. 先看当前宏观阶段
2. 再看增长、通胀、流动性三项评分
3. 再看估值状态和估值置信度
4. 最后看资产偏好表和理由

如果阶段判断和估值结论同向，资产信号通常更强。  
如果阶段支持但估值偏贵，资产偏好可能只会停留在中性。

## 五、评分体系逻辑

### 增长评分
- 分数越高：增长越偏强
- 分数越低：增长越偏弱

### 通胀评分
- 分数越高：通胀压力越强
- 分数越低：越偏去通胀

### 流动性评分
- 分数越高：金融条件越宽松
- 分数越低：金融条件越偏紧

### 估值评分
- 分数越高：越低估，或折现环境越友好
- 分数越低：越高估，或折现压力越大

这些分数是规则驱动的标准化结果。  
它们是为了把不同国家的宏观与估值信号映射到同一套阅读框架里，而不是黑盒概率模型。

## 六、实时偏移层怎么理解

实时偏移层不是替代主模型。  
它的作用是回答一个更高频的问题：

- 月频宏观主状态还没切换
- 但利率、收益率、估值等市场敏感输入是否已经开始提前变化

因此它更像一个日频或高频的补充观察层，用来提示主模型可能正在面临的边际变化。

### 实时偏移评分
- 分数为正：更偏风险偏好
- 分数为负：更偏防御
- 越接近 0：说明高频市场信号分化，暂时没有形成明确方向

### 实时偏移方向
- `偏风险偏好`：高频市场信号整体更支持风险资产
- `中性`：高频信号尚未形成明确共振
- `偏防御`：高频市场信号更偏向防御或收紧交易

### 三条子评分怎么读

#### 风险偏好偏移
主要看估值类或风险资产敏感变量的边际变化。  
如果估值代理回落、风险补偿改善，通常会把这条评分推向正值。

#### 利率偏移
主要看政策利率、长端收益率等变量的最新变化。  
如果利率下行、市场开始交易更宽松的环境，这条评分通常会上升。  
如果利率继续上行、市场交易更紧的金融条件，这条评分通常会下降。

#### 通胀偏移
主要看 CPI 或核心通胀这类高频可更新通胀输入的边际变化。  
如果通胀读数回落，通常更有利于风险资产和宽松预期；  
如果通胀重新抬头，则通常对市场形成约束。

### 怎么把实时偏移层和主模型一起看

最实用的方法不是只看一个分数，而是把它当成“边际变化提示”：

- 主模型稳定，但实时偏移层明显转强：说明市场可能已经在提前交易改善
- 主模型稳定，但实时偏移层明显转弱：说明市场可能已经开始提前交易压力
- 主模型与实时偏移层同向：说明当前主线更一致，解读可信度通常更高
- 主模型与实时偏移层反向：说明当前处在过渡阶段，更需要谨慎

## 七、估值层怎么理解

### 美国
更接近正式投研口径，综合使用 Buffett 指标、PE、CAPE、PB、实际利率、期限利差、股权风险补偿与信用利差。

### 中国
采用研究代理口径，重点看 HS300 PE、HS300 PB、China CAPE、实际利率、期限利差和股权风险补偿。

### 欧元区
采用研究代理口径，重点看 Europe PE、Europe CAPE、实际利率、期限利差和股权风险补偿。

估值层不要只看一个标签。  
最有用的是一起看：
- `估值状态`
- `估值评分`
- `估值置信度`
- `估值已用输入`

## 八、共识偏差怎么看

共识偏差不是拿新闻预测市场。  
它是在比较：模型现在的判断，与主流公开叙事是否一致。

比较维度只有三个：
- 增长
- 通胀
- 政策倾向

### 增长偏离评分
- 为正：模型比市场共识更偏增长乐观
- 为负：模型比市场共识更偏增长谨慎

### 通胀偏离评分
- 为正：模型判断的通胀风险低于主流叙事
- 为负：模型判断的通胀风险高于主流叙事

### 政策偏离评分
- 为正：模型比主流叙事更偏鸽派或更偏宽松
- 为负：模型比主流叙事更偏鹰派或更偏偏紧

### 总偏离评分
- 越接近 0：模型与共识越一致
- 绝对值越大：模型与共识偏离越明显

如果共识偏差很大，说明模型和市场主流叙事站在不同一边，值得重点复核。

## 九、置信度怎么读

### 高
数据覆盖较完整，数据较新，估值输入较充分，结论相对更稳。

### 中
主链路可用，但部分估值或增强输入不完整，结论需要结合理由一起读。

### 低
通常意味着数据滞后、覆盖不足、估值输入缺失较多，结论只能谨慎使用。

置信度不是方向判断，它是在告诉你：这条结论有多扎实。

## 十、为什么理由一定要看

理由是在解释：  
当前这条资产偏好，究竟是由宏观阶段驱动，还是由流动性条件驱动，还是被估值明显抵消。

因此建议始终一起看：
- 偏好
- 评分
- 置信度
- 理由

如果只看“超配 / 中性 / 低配”，很容易误读。

## 十一、两种全球口径的区别

### 最新可得口径
每个地区使用各自最近一期有效数据。  
优点是更贴近实时监控。  
缺点是各地区日期不一定完全一致。

### 共同日期口径
所有地区退回到最近共同日期。  
优点是横向比较更整齐。  
缺点是会牺牲一部分时效性。

## 十二、系统更新日期和数据截至日期为什么不同

这两个日期不同是正常的：

- `系统更新日期`：程序今天已经刷新
- `数据截至日期`：该模块实际能拿到的最新有效数据日期

很多官方宏观数据本来就不是日更，所以不能把“今天刷新了”理解成“所有宏观数据都更新到今天”。

## 十三、最实用的阅读顺序

建议你每次都按这个顺序：

1. 先看宏观阶段有没有切换
2. 再看增长、通胀、流动性评分是否靠近或越过中性区
3. 再看实时偏移层是否已经出现更高频的边际变化
4. 再看估值是在支持还是抵消宏观结论
5. 再看资产偏好是否发生方向变化
6. 最后看共识偏差，判断模型是否和市场主流叙事站在不同一边

这套系统最适合用作：
- 宏观监控框架
- 投研讨论底稿
- 资产配置辅助系统

不建议把它直接理解成自动交易引擎。
""".strip()

    return """
This system answers two questions first:
what macro regime are we in, and what should that imply for asset allocation.

## 1. Start with the four macro states

| Macro state | Core meaning | Typical overweight | Typical neutral | Typical underweight |
| --- | --- | --- | --- | --- |
| Goldilocks | Growth is resilient, inflation is softer, liquidity is not a major drag | Global equities, US equities, Eurozone equities | USD duration, commodities | Gold, dollar |
| Reflation | Growth improves and inflation rises | Equities, commodities | Dollar, gold | USD duration |
| Slowdown | Growth weakens and inflation softens | USD duration | Gold, dollar | Equities, commodities |
| Stagflation | Growth weakens but inflation remains firm | Gold, some commodities | Dollar | Equities, USD duration |

The table above is the base framework rather than a mechanical rule.  
Final preferences still depend on liquidity, valuation, and confidence.

## 2. How to read asset preferences

### Bullish
The macro-plus-valuation mix supports overweight positioning.

### Neutral
Signals are mixed, or macro and valuation are offsetting each other.

### Cautious
The macro-plus-valuation mix argues for underweight positioning.

Asset preferences are directional allocation tilts, not price guarantees.

## 3. What each asset means

### On the global page
- `Global Equities`: the aggregate equity view across the monitored regions
- `USD Duration`: a US-led duration view for the global rate cycle
- `Gold`: inflation, defense, and stagflation hedge exposure
- `Dollar`: dollar exposure under the current liquidity and risk backdrop
- `Commodities`: the combined growth-plus-inflation cycle view
- `US / China / Eurozone Equities`: local regional equity views

### On country pages
- `Equities`: local equity preference
- `Duration`: local duration preference
- `Gold`: local macro view on gold
- `Dollar`: local macro view on dollar exposure

## 4. How to read scores

- `Growth Score`: higher means stronger growth
- `Inflation Score`: higher means more inflation pressure
- `Liquidity Score`: higher means easier financial conditions
- `Valuation Score`: higher means cheaper / more supportive conditions

These are transparent standardized scores, not black-box probabilities.

## 5. How to read the nowcast overlay

The nowcast overlay does not replace the main macro model.  
It answers a higher-frequency question:

- the monthly macro regime may not have changed yet
- but are market-sensitive inputs already moving ahead of it

### Nowcast score
- positive: the high-frequency backdrop is leaning more risk-on
- negative: the high-frequency backdrop is leaning more defensive
- near zero: signals are mixed and do not yet point to a clear tilt

### Three sub-scores

#### Risk overlay
Tracks valuation-sensitive and risk-sensitive inputs.  
If valuation proxies cheapen or risk compensation improves, this sub-score usually rises.

#### Rates overlay
Tracks policy rates and longer-term yields.  
Falling rates tend to support this score; rising rates tend to weaken it.

#### Inflation overlay
Tracks the latest movement in CPI-style inflation inputs.  
Cooling inflation usually supports risk assets and easier policy expectations, while reheating inflation tends to work the other way.

### How to use it
- if the macro regime is stable but the overlay turns stronger, markets may be front-running improvement
- if the macro regime is stable but the overlay turns weaker, markets may be front-running deterioration
- if macro and overlay point in the same direction, the current narrative is more coherent
- if they diverge, the system is likely in a transition phase

## 6. How to read valuation

### United States
Closer to institutional valuation practice:
- Buffett Indicator
- PE
- CAPE
- PB
- real yield
- term spread
- equity risk premium
- credit spread

### China
Research-proxy framework:
- HS300 PE
- HS300 PB
- China CAPE
- real yield
- term spread
- equity risk compensation

### Eurozone
Research-proxy framework:
- Europe PE
- Europe CAPE
- real yield
- term spread
- equity risk compensation

Always read:
- valuation regime
- valuation score
- valuation confidence

## 7. How to read consensus deviation

Consensus deviation compares:

model view  
versus  
mainstream public narrative

across:
- growth
- inflation
- policy bias

- positive growth deviation: model is more growth-positive than consensus
- negative growth deviation: model is more growth-negative than consensus
- positive inflation deviation: model sees less inflation risk than consensus
- negative inflation deviation: model sees more inflation risk than consensus
- positive policy deviation: model is more dovish than consensus
- negative policy deviation: model is more hawkish than consensus

## 8. How to read confidence

### High
- broad coverage
- fresh enough data
- stronger valuation coverage

### Medium
- core pipeline works
- some valuation or enrichment inputs are incomplete or stale

### Low
- stale data
- thin coverage
- weaker valuation or enrichment coverage

## 9. Why the reason field matters

The `Reason` field explains whether the asset call is driven by:
- macro regime
- liquidity
- valuation
- or a trade-off between them

So the best practice is to read:
- preference
- score
- confidence
- reason

together.

## 10. Two global modes

### Latest available
Each region uses its own latest valid observation.  
Better for live monitoring, but dates may differ by region.

### Last common date
All regions align to the latest shared date.  
Better for cross-region comparability, but less timely.

## 11. System update date vs data-through date

These are intentionally different:
- system update date = when the monitor refreshed
- data-through date = the latest valid date behind that module

Official macro data is not daily, so a refreshed system can still have a lagged macro cutoff date.

## 12. Practical usage

The most useful reading order is:

1. check whether the macro regime changed
2. check whether scores crossed the neutral zone
3. check whether the nowcast overlay is already showing a higher-frequency shift
4. check whether valuation supports or offsets the macro view
5. check whether asset preferences changed direction
6. check whether the model materially diverges from consensus

Use this system as:
- a macro monitoring framework
- a research discussion tool
- an asset allocation support tool

Do not treat it as a fully automated trading engine.
""".strip()


def localize_region(region: str, language: str | None = None) -> str:
    """Translate region labels for the page header."""
    if get_display_language(language) == "zh":
        return REGION_MAP_ZH.get(region, region)
    return region


def translate_runtime_text(text: object, language: str | None = None) -> str:
    """Translate runtime-generated English rationale text into Chinese when needed."""
    raw = format_display_value(text).strip()
    if get_display_language(language) != "zh" or raw == "No data":
        return raw
    if raw in LABEL_MAP_ZH:
        return LABEL_MAP_ZH[raw]
    if re.fullmatch(r"[A-Za-z0-9_/-]+", raw):
        direct_label = humanize_label(raw)
    else:
        direct_label = raw
    if direct_label != raw:
        return direct_label
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
        "The macro backdrop is mixed for global equities. Valuations look expensive.": "全球权益资产的宏观信号目前偏中性，估值大体高估。",
        "The macro backdrop is mixed for global equities. Valuations look fair.": "全球权益资产的宏观信号目前偏中性，估值大体合理。",
        "The macro backdrop is mixed for global equities. Valuations look cheap.": "全球权益资产的宏观信号目前偏中性，估值大体低估。",
        "United States is in Goldilocks with neutral liquidity, so dollar duration remains neutral. Valuations look expensive.": "美国当前处于温和增长阶段，流动性环境中性，因此美元久期维持中性，估值大体高估。",
        "Neutral liquidity reduces support for the dollar.": "中性流动性环境对美元的支撑有所减弱。",
        "United States is in Goldilocks with neutral liquidity, but rich valuations keep the equity view from turning fully bullish. Valuations look expensive.": "美国当前处于温和增长阶段，流动性环境中性，但偏贵的估值限制了权益资产进一步转向超配，估值大体高估。",
        "China is in Slowdown with neutral liquidity, which limits the equity view. Valuations look fair.": "中国当前处于增长放缓阶段，流动性环境中性，对权益资产判断形成约束，估值大体合理。",
        "China is in Slowdown with easy liquidity, which limits the equity view. Valuations look fair.": "中国当前处于增长放缓阶段，流动性环境宽松，对权益资产判断形成约束，估值大体合理。",
        "Eurozone is in Goldilocks with neutral liquidity, but rich valuations keep the equity view from turning fully bullish. Valuations look expensive.": "欧元区当前处于温和增长阶段，流动性环境中性，但偏贵的估值限制了权益资产进一步转向超配，估值大体高估。",
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
        r"(?P<country>United States|China|Eurozone) is in (?P<regime>.+?) with (?P<liquidity>.+?) liquidity, but rich valuations keep the equity view from turning fully bullish\. Valuations look (?P<valuation>.+?)\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = (
            f"{translate_runtime_text(pattern.group('country'))}当前处于{translate_runtime_text(pattern.group('regime'))}阶段，"
            f"流动性环境{translate_runtime_text(pattern.group('liquidity'))}，"
            "但偏贵的估值限制了权益资产进一步转向超配，"
            f"估值大体{translate_runtime_text(pattern.group('valuation'))}。"
        )
        tail = pattern.group("tail")
        return result if not tail else result + " " + translate_runtime_text(tail)

    pattern = re.match(
        r"(?P<country>United States|China|Eurozone) is in (?P<regime>.+?) with (?P<liquidity>.+?) liquidity, but cheaper valuations soften the macro headwind for equities\. Valuations look (?P<valuation>.+?)\.(?: (?P<tail>.+))?$",
        raw,
    )
    if pattern:
        result = (
            f"{translate_runtime_text(pattern.group('country'))}当前处于{translate_runtime_text(pattern.group('regime'))}阶段，"
            f"流动性环境{translate_runtime_text(pattern.group('liquidity'))}，"
            "但较低的估值在一定程度上缓和了权益资产面临的宏观压力，"
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

def render_user_manual_view() -> None:
    """Render the standalone in-app user guide page."""
    st.header(tr("User Guide"))
    st.caption(tr("Detailed guide to the model, scoring system, confidence, consensus deviation, and asset mapping."))

    with st.container(border=True):
        st.caption(tr("Current core view"))
        st.markdown(f"### {tr('Read This First')}")
        st.write(
            tr(
                "This system is a macro monitoring and allocation support framework. It first answers where the cycle is, then asks which assets are more suitable to overweight, hold neutral, or underweight."
            )
        )

    st.subheader(tr("Four Macro States"))
    st.plotly_chart(build_macro_cycle_guide_chart(), use_container_width=True)
    st.caption(tr("The chart above is the base framework rather than a mechanical rule. Final preferences still depend on liquidity, valuation, and confidence."))

    st.subheader(tr("How to Read Asset Preferences"))
    pref_left, pref_mid, pref_right = st.columns(3)
    with pref_left:
        render_html_card(tr("Preference"), humanize_label("bullish"), note=tr("The macro-plus-valuation mix supports overweight positioning."))
    with pref_mid:
        render_html_card(tr("Preference"), humanize_label("neutral"), note=tr("Signals are mixed, or macro and valuation offset each other."))
    with pref_right:
        render_html_card(tr("Preference"), humanize_label("cautious"), note=tr("The macro-plus-valuation mix argues for underweight positioning."))

    st.subheader(tr("How to Read Scores"))
    score_cols = st.columns(4)
    score_cards = [
        (tr("Growth Score"), tr("Higher means stronger growth.")),
        (tr("Inflation Score"), tr("Higher means more inflation pressure.")),
        (tr("Liquidity Score"), tr("Higher means easier financial conditions.")),
        (tr("Valuation Score"), tr("Higher means cheaper and more supportive conditions.")),
    ]
    for idx, (label, note) in enumerate(score_cards):
        with score_cols[idx]:
            render_html_card(label, "0.00", note=note)
    st.caption(tr("These are transparent standardized scores, not black-box probabilities."))

    st.subheader(tr("How to Read Confidence"))
    confidence_cols = st.columns(3)
    confidence_cards = [
        (humanize_label("high"), tr("Broad coverage, fresh data, and stronger valuation support.")),
        (humanize_label("medium"), tr("Core pipeline works, but some valuation or enrichment inputs are incomplete or stale.")),
        (humanize_label("low"), tr("Stale data, thin coverage, or weaker valuation and enrichment coverage.")),
    ]
    for idx, (label, note) in enumerate(confidence_cards):
        with confidence_cols[idx]:
            render_html_card(tr("Confidence"), label, note=note)

    st.subheader(tr("How to Read Consensus Deviation"))
    st.write(
        tr(
            "Consensus deviation compares the model view versus mainstream public narrative across growth, inflation, and policy bias. A positive growth deviation means the model is more growth-positive than consensus; a negative one means the model is more growth-negative."
        )
    )
    st.write(
        tr(
            "A positive inflation deviation means the model sees less inflation risk than consensus. A positive policy deviation means the model is more dovish than consensus."
        )
    )

    st.subheader(tr("How to Read the Nowcast Overlay"))
    overlay_cols = st.columns(3)
    overlay_cards = [
        (overlay_metric_label("risk"), tr("Tracks risk-sensitive inputs such as equities, volatility, and the dollar.")),
        (overlay_metric_label("rates"), tr("Tracks policy rates and longer-term yields.")),
        (overlay_metric_label("inflation"), tr("Tracks inflation-sensitive inputs such as CPI-style data, gold, and oil.")),
    ]
    for idx, (label, note) in enumerate(overlay_cards):
        with overlay_cols[idx]:
            render_html_card(label, "0.00", note=note)
    explain_left, explain_mid, explain_right = st.columns(3)
    with explain_left:
        st.caption(tr("Positive score: higher-frequency inputs are leaning more risk-on / easier than the monthly macro regime."))
    with explain_mid:
        st.caption(tr("Negative score: higher-frequency inputs are leaning more defensive / tighter than the monthly macro regime."))
    with explain_right:
        st.caption(tr("Near zero: signals are mixed and do not yet point to a clear short-term tilt."))
    st.caption(
        tr(
            "The nowcast overlay does not replace the macro model. It is a higher-frequency layer that shows whether market-sensitive inputs are already moving ahead of the monthly macro regime."
        )
    )

    st.subheader(tr("Practical Reading Order"))
    st.markdown(
        "\n".join(
            [
                f"1. {tr('Check whether the macro regime changed.')}",
                f"2. {tr('Check whether scores crossed the neutral zone.')}",
                f"3. {tr('Check whether the nowcast overlay shows a higher-frequency shift.')}",
                f"4. {tr('Check whether valuation supports or offsets the macro view.')}",
                f"5. {tr('Check whether asset preferences changed direction.')}",
                f"6. {tr('Check whether the model materially diverges from consensus.')}",
            ]
        )
    )


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


def format_nowcast_driver(driver: dict[str, object], country: str | None = None) -> str:
    """Format a compact nowcast driver explanation for dashboard display."""
    series_label = humanize_label(driver.get("series_id"))
    driver_label = humanize_label(driver.get("driver"))
    if country is not None:
        return f"{humanize_label(country)}：{series_label} {driver_label}"
    return f"{series_label} {driver_label}"


def format_alert_headline(row: pd.Series) -> str:
    """Render one alert headline in the active display language."""
    alert_type = str(row.get("alert_type", ""))
    region = humanize_label(row.get("region"))
    entity = humanize_label(row.get("entity_name"))
    old_value = humanize_label(row.get("old_value"))
    new_value = humanize_label(row.get("new_value"))
    metric = row.get("metric_value")
    if get_display_language() == "zh":
        if alert_type == "partial_coverage":
            coverage = f"{float(metric):.0%}" if pd.notna(metric) else tr("No data")
            return f"全球覆盖不足，当前覆盖率为 {coverage}。"
        if alert_type == "global_regime":
            return f"全球宏观阶段由 {old_value} 变为 {new_value}。"
        if alert_type == "investment_clock":
            return f"全球投资时钟由 {old_value} 变为 {new_value}。"
        if alert_type == "confidence_downgrade":
            return f"{entity} 的置信度由 {old_value} 下调至 {new_value}。"
        if alert_type == "very_stale_country":
            days = int(metric) if pd.notna(metric) else 0
            return f"{region} 数据已严重滞后，当前滞后 {days} 天。"
        if alert_type == "country_not_usable":
            return f"{region} 虽然本地已就绪，但未纳入当前全球口径。"
        if alert_type == "consensus_gap":
            return f"{region} 的模型判断与市场共识偏离较大。"
        if alert_type == "nowcast_shift":
            return f"实时偏移层目前转向 {entity}。"
        return f"{region} 出现新的监控提醒。"
    if alert_type == "partial_coverage":
        coverage = f"{float(metric):.0%}" if pd.notna(metric) else tr("No data")
        return f"Global coverage is thin at {coverage}."
    if alert_type == "global_regime":
        return f"Global regime changed from {old_value} to {new_value}."
    if alert_type == "investment_clock":
        return f"Investment clock changed from {old_value} to {new_value}."
    if alert_type == "confidence_downgrade":
        return f"{entity} confidence was downgraded from {old_value} to {new_value}."
    if alert_type == "very_stale_country":
        days = int(metric) if pd.notna(metric) else 0
        return f"{region} data is very stale at {days} days."
    if alert_type == "country_not_usable":
        return f"{region} is ready locally but excluded from the selected global mode."
    if alert_type == "consensus_gap":
        return f"{region} shows a large model-versus-consensus gap."
    if alert_type == "nowcast_shift":
        return f"The nowcast overlay has shifted {entity}."
    return f"{region} has a new monitor alert."


def format_alert_detail(row: pd.Series) -> str:
    """Render one alert detail line in the active display language."""
    alert_type = str(row.get("alert_type", ""))
    reason = translate_runtime_text(row.get("reason"))
    metric = row.get("metric_value")
    if get_display_language() == "zh":
        if alert_type == "partial_coverage":
            return reason or "当前全球汇总可靠性下降，解读时应更谨慎。"
        if alert_type == "consensus_gap" and pd.notna(metric):
            return f"当前总偏离评分为 {float(metric):.2f}。{reason}"
        if alert_type == "nowcast_shift" and pd.notna(metric):
            return f"当前实时偏移评分为 {float(metric):.2f}。{reason}"
        return reason or "该提醒来自最新监控结果。"
    if alert_type == "consensus_gap" and pd.notna(metric):
        return f"Current total deviation score is {float(metric):.2f}. {reason}"
    if alert_type == "nowcast_shift" and pd.notna(metric):
        return f"Current nowcast overlay score is {float(metric):.2f}. {reason}"
    return reason or "This alert was generated from the latest monitor state."


def _pill_class(preference: object) -> str:
    """Map a preference label to a visual pill style."""
    normalized = str(preference).lower()
    if normalized in {"bullish", "超配"}:
        return "pill-positive"
    if normalized in {"cautious", "低配"}:
        return "pill-negative"
    return "pill-neutral"


def render_html_card(label: str, value: str, note: str = "", pill: str | None = None, pill_class: str = "pill-neutral") -> None:
    """Render a compact dashboard card."""
    del pill_class
    with st.container(border=True):
        if pill:
            st.caption(pill)
        st.markdown(f"**{label}**")
        st.markdown(f"### {value}")
        if note:
            st.caption(note)


def render_alert_banner(title: str, body: str) -> None:
    """Render the top hero banner for the global page."""
    with st.container(border=True):
        st.caption(tr("Global Snapshot"))
        st.markdown(f"### {title}")
        st.write(body)


def render_alert_box(severity: str, headline: str, detail: str) -> None:
    """Render one alert callout box."""
    with st.container(border=True):
        st.caption(severity)
        st.markdown(f"**{headline}**")
        st.write(detail)


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


def manual_asset_list(items: list[str]) -> str:
    """Format an asset list for the manual chart using stable display labels."""
    formatted: list[str] = []
    for item in items:
        if item in {"global_equities", "us_equities", "china_equities", "eurozone_equities", "duration"}:
            formatted.append(humanize_global_asset_label(item))
        else:
            formatted.append(humanize_label(item))
    return "、".join(formatted)


def manual_asset_block(items: list[str]) -> str:
    """Format an asset list for quadrant annotations using wrapped lines."""
    formatted: list[str] = []
    for item in items:
        if item in {"global_equities", "us_equities", "china_equities", "eurozone_equities", "duration"}:
            formatted.append(humanize_global_asset_label(item))
        else:
            formatted.append(humanize_label(item))
    return "<br>".join(formatted)


def summarize_reason_text(text: object, max_chars: int = 42) -> str:
    """Shorten long rationale text for compact table display."""
    value = format_display_value(text)
    if value == tr("No data"):
        return value
    first_sentence = re.split(r"[。.!?]", value, maxsplit=1)[0].strip()
    candidate = first_sentence or value
    if len(candidate) <= max_chars:
        return candidate
    return candidate[: max_chars - 1].rstrip() + "…"


def build_macro_cycle_guide_chart() -> go.Figure:
    """Render a Merrill-style four-quadrant macro guide for the manual page."""
    figure = go.Figure()
    language = get_display_language()

    quadrant_styles = [
        {"x0": 0, "x1": 1, "y0": 1, "y1": 2, "fill": "rgba(190, 95, 80, 0.14)"},
        {"x0": 1, "x1": 2, "y0": 1, "y1": 2, "fill": "rgba(240, 185, 11, 0.12)"},
        {"x0": 0, "x1": 1, "y0": 0, "y1": 1, "fill": "rgba(90, 120, 180, 0.16)"},
        {"x0": 1, "x1": 2, "y0": 0, "y1": 1, "fill": "rgba(132, 187, 90, 0.14)"},
    ]
    for style in quadrant_styles:
        figure.add_shape(
            type="rect",
            x0=style["x0"],
            x1=style["x1"],
            y0=style["y0"],
            y1=style["y1"],
            line={"color": "rgba(255,255,255,0.08)", "width": 1},
            fillcolor=style["fill"],
        )

    manual_points = [
        {
            "x": 0.08,
            "y": 1.92,
            "title": humanize_label("stagflation"),
            "meaning": tr("Growth is weak while inflation remains firm; it is one of the least friendly combinations."),
            "overweight": manual_asset_block(["gold", "commodities"]),
            "neutral": manual_asset_block(["dollar"]),
            "underweight": manual_asset_block(["global_equities", "duration"]),
        },
        {
            "x": 1.08,
            "y": 1.92,
            "title": humanize_label("reflation"),
            "meaning": tr("Growth improves, inflation rises, and nominal activity re-accelerates."),
            "overweight": manual_asset_block(["global_equities", "commodities"]),
            "neutral": manual_asset_block(["dollar", "gold"]),
            "underweight": manual_asset_block(["duration"]),
        },
        {
            "x": 0.08,
            "y": 0.92,
            "title": humanize_label("slowdown"),
            "meaning": tr("Growth weakens, inflation softens, and risk appetite fades."),
            "overweight": manual_asset_block(["duration"]),
            "neutral": manual_asset_block(["gold", "dollar"]),
            "underweight": manual_asset_block(["global_equities", "commodities"]),
        },
        {
            "x": 1.08,
            "y": 0.92,
            "title": humanize_label("goldilocks"),
            "meaning": tr("Growth is stable, inflation is soft, and liquidity is not obviously tightening."),
            "overweight": manual_asset_block(["global_equities", "us_equities", "eurozone_equities"]),
            "neutral": manual_asset_block(["duration", "commodities"]),
            "underweight": manual_asset_block(["gold", "dollar"]),
        },
    ]

    for point in manual_points:
        text = (
            f"<b>{point['title']}</b><br>"
            f"{point['meaning']}<br><br>"
            f"<b>{tr('Typical Overweight')}</b>：{point['overweight']}<br>"
            f"<b>{tr('Typical Neutral')}</b>：{point['neutral']}<br>"
            f"<b>{tr('Typical Underweight')}</b>：{point['underweight']}"
        )
        figure.add_annotation(
            x=point["x"],
            y=point["y"],
            text=text,
            showarrow=False,
            align="left",
            font={"size": 13 if language == "zh" else 12, "color": "#eef2f7"},
            bgcolor="rgba(10, 13, 18, 0.78)",
            bordercolor="rgba(240,185,11,0.22)",
            borderpad=8,
            xanchor="left",
            yanchor="top",
        )

    figure.update_xaxes(
        range=[0, 2],
        tickvals=[0.5, 1.5],
        ticktext=[tr("Lower"), tr("Higher")],
        title=tr("Growth Momentum"),
        showgrid=False,
        zeroline=False,
        color="#9aa5b5",
    )
    figure.update_yaxes(
        range=[0, 2],
        tickvals=[0.5, 1.5],
        ticktext=[tr("Lower"), tr("Higher")],
        title=tr("Inflation Pressure"),
        showgrid=False,
        zeroline=False,
        color="#9aa5b5",
    )
    figure.update_layout(
        margin={"l": 30, "r": 30, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=640,
    )
    return figure


def _wrap_svg_text(text: str, max_chars: int) -> list[str]:
    """Wrap text for SVG rendering with simple CJK-friendly width limits."""
    normalized = text.replace("\n", " ").strip()
    if not normalized:
        return []
    if " " not in normalized:
        return [normalized[i : i + max_chars] for i in range(0, len(normalized), max_chars)]

    words = normalized.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            lines.append(word[:max_chars])
            current = word[max_chars:]
    if current:
        lines.append(current)
    return lines


def _svg_text_block(x: int, y: int, lines: list[str], font_size: int, *, color: str = "#eef2f7", weight: str = "400", line_height: float = 1.45) -> str:
    """Build one SVG text block with tspans for wrapped lines."""
    if not lines:
        return ""
    dy = int(font_size * line_height)
    spans = [f'<tspan x="{x}" y="{y + index * dy}">{html.escape(line)}</tspan>' for index, line in enumerate(lines)]
    return f'<text font-size="{font_size}" font-weight="{weight}" fill="{color}" font-family="Inter, PingFang SC, Noto Sans SC, sans-serif">{"".join(spans)}</text>'


def build_macro_cycle_guide_svg() -> str:
    """Build a high-resolution SVG version of the four-state macro guide."""
    width = 1280
    height = 980
    left = 135
    top = 78
    quadrant_w = 500
    quadrant_h = 350
    axis_summary = f"{tr('Growth Momentum')}：{tr('Lower')} → {tr('Higher')}；{tr('Inflation Pressure')}：{tr('Lower')} → {tr('Higher')}"

    cards = [
        {
            "title": humanize_label("stagflation"),
            "meaning": tr("Growth is weak while inflation remains firm; it is one of the least friendly combinations."),
            "overweight": "、".join(manual_asset_block(["gold", "commodities"]).split("<br>")),
            "neutral": "、".join(manual_asset_block(["dollar"]).split("<br>")),
            "underweight": "、".join(manual_asset_block(["global_equities", "duration"]).split("<br>")),
            "fill": "rgba(190, 95, 80, 0.14)",
            "x": left,
            "y": top,
        },
        {
            "title": humanize_label("reflation"),
            "meaning": tr("Growth improves, inflation rises, and nominal activity re-accelerates."),
            "overweight": "、".join(manual_asset_block(["global_equities", "commodities"]).split("<br>")),
            "neutral": "、".join(manual_asset_block(["dollar", "gold"]).split("<br>")),
            "underweight": "、".join(manual_asset_block(["duration"]).split("<br>")),
            "fill": "rgba(240, 185, 11, 0.12)",
            "x": left + quadrant_w,
            "y": top,
        },
        {
            "title": humanize_label("slowdown"),
            "meaning": tr("Growth weakens, inflation softens, and risk appetite fades."),
            "overweight": "、".join(manual_asset_block(["duration"]).split("<br>")),
            "neutral": "、".join(manual_asset_block(["gold", "dollar"]).split("<br>")),
            "underweight": "、".join(manual_asset_block(["global_equities", "commodities"]).split("<br>")),
            "fill": "rgba(90, 120, 180, 0.16)",
            "x": left,
            "y": top + quadrant_h,
        },
        {
            "title": humanize_label("goldilocks"),
            "meaning": tr("Growth is stable, inflation is soft, and liquidity is not obviously tightening."),
            "overweight": "、".join(manual_asset_block(["global_equities", "us_equities", "eurozone_equities"]).split("<br>")),
            "neutral": "、".join(manual_asset_block(["duration", "commodities"]).split("<br>")),
            "underweight": "、".join(manual_asset_block(["gold", "dollar"]).split("<br>")),
            "fill": "rgba(132, 187, 90, 0.14)",
            "x": left + quadrant_w,
            "y": top + quadrant_h,
        },
    ]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b0e14"/>',
        f'<text x="{left}" y="36" font-size="20" font-weight="600" fill="#9aa5b5" font-family="Inter, PingFang SC, Noto Sans SC, sans-serif">{html.escape(axis_summary)}</text>',
        f'<text x="52" y="{top + quadrant_h + 18}" transform="rotate(-90 52 {top + quadrant_h + 18})" font-size="20" font-weight="600" fill="#eef2f7" font-family="Inter, PingFang SC, Noto Sans SC, sans-serif">{html.escape(tr("Inflation Pressure"))}</text>',
        f'<text x="{left + 180}" y="{height - 30}" font-size="20" font-weight="600" fill="#eef2f7" font-family="Inter, PingFang SC, Noto Sans SC, sans-serif">{html.escape(tr("Growth Momentum"))}</text>',
        f'<text x="{left - 62}" y="{top + 96}" font-size="18" font-weight="500" fill="#eef2f7" font-family="Inter, PingFang SC, Noto Sans SC, sans-serif">{html.escape(tr("Higher"))}</text>',
        f'<text x="{left - 62}" y="{top + quadrant_h + 96}" font-size="18" font-weight="500" fill="#eef2f7" font-family="Inter, PingFang SC, Noto Sans SC, sans-serif">{html.escape(tr("Lower"))}</text>',
        f'<text x="{left + 190}" y="{height - 68}" font-size="18" font-weight="500" fill="#eef2f7" font-family="Inter, PingFang SC, Noto Sans SC, sans-serif">{html.escape(tr("Lower"))}</text>',
        f'<text x="{left + quadrant_w + 190}" y="{height - 68}" font-size="18" font-weight="500" fill="#eef2f7" font-family="Inter, PingFang SC, Noto Sans SC, sans-serif">{html.escape(tr("Higher"))}</text>',
    ]

    for card in cards:
        x = card["x"]
        y = card["y"]
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{quadrant_w}" height="{quadrant_h}" fill="{card["fill"]}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>'
        )
        inner_x = x + 28
        inner_y = y + 44
        svg_parts.append(
            f'<rect x="{inner_x - 12}" y="{inner_y - 20}" width="{quadrant_w - 56}" height="{quadrant_h - 56}" rx="16" ry="16" fill="rgba(10,13,18,0.78)" stroke="rgba(240,185,11,0.22)" stroke-width="1.1"/>'
        )
        svg_parts.append(_svg_text_block(inner_x, inner_y, [card["title"]], 26, weight="700"))
        svg_parts.append(_svg_text_block(inner_x, inner_y + 52, _wrap_svg_text(str(card["meaning"]), 21), 15, color="#d7dde7"))
        svg_parts.append(_svg_text_block(inner_x, inner_y + 126, _wrap_svg_text(f"{tr('Typical Overweight')}：{card['overweight']}", 21), 15))
        svg_parts.append(_svg_text_block(inner_x, inner_y + 192, _wrap_svg_text(f"{tr('Typical Neutral')}：{card['neutral']}", 21), 15))
        svg_parts.append(_svg_text_block(inner_x, inner_y + 258, _wrap_svg_text(f"{tr('Typical Underweight')}：{card['underweight']}", 21), 15))

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def overlay_metric_label(dimension: str) -> str:
    """Return a stable localized label for nowcast overlay sub-scores."""
    labels = {
        "risk": {"zh": "风险偏好偏移", "en": "Risk Overlay"},
        "rates": {"zh": "利率偏移", "en": "Rates Overlay"},
        "inflation": {"zh": "通胀偏移", "en": "Inflation Overlay"},
    }
    language = get_display_language()
    entry = labels.get(str(dimension), {"zh": str(dimension), "en": str(dimension)})
    return entry["zh"] if language == "zh" else entry["en"]


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
    monitor_alerts = load_optional_csv(Path("data/processed/monitor_alerts.csv"))

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

    hero_title = tr("Current Core View")
    hero_body = (
        f"{tr('Latest Regime')}：{humanize_label(latest.get('regime'))}；"
        f"{tr('Liquidity Overlay')}：{humanize_label(latest.get('liquidity_regime'))}。"
    )
    render_alert_banner(hero_title, hero_body)
    st.caption(
        " · ".join(
            [
                f"{tr('System Update Date')}：{system_update_date}",
                f"{tr('Data Through Date')}：{pd.Timestamp(latest['date']).date().isoformat()}",
                f"{tr('Latest Market-sensitive Input')}：{market_input_date}",
            ]
        )
    )

    st.subheader(tr("Primary Signals"))
    top_left, top_mid, top_right, top_far_right = st.columns(4)
    top_left.metric(tr("Latest Regime"), humanize_label(latest.get("regime")))
    top_mid.metric(tr("Liquidity Overlay"), humanize_label(latest.get("liquidity_regime")))
    top_right.metric(tr("Regime Confidence"), humanize_label(latest.get("regime_confidence")))
    top_far_right.metric(tr("Valuation Regime"), humanize_label(latest_valuation.get("valuation_regime")))

    st.subheader(tr("Nowcast Overlay"))
    overlay_left, overlay_mid, overlay_right = st.columns(3)
    overlay_left.metric(tr("Latest Market-sensitive Input"), market_input_date)
    overlay_mid.metric(tr("Nowcast Score"), f"{float(nowcast_overlay['overlay_score']):.2f}")
    overlay_right.metric(tr("Nowcast Direction"), humanize_label(nowcast_overlay["overlay_direction"]))
    st.caption(
        f"{tr('Source')}："
        ", ".join(humanize_label(item) for item in nowcast_overlay["freshest_market_series"])
        if nowcast_overlay["freshest_market_series"]
        else tr("No data")
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
    st.caption(f"{tr('Confidence')}：{humanize_label(nowcast_overlay['overlay_confidence'])}")
    if nowcast_overlay["signal_drivers"]:
        st.caption(
            f"{tr('Driver Summary')}："
            + "；".join(format_nowcast_driver(driver) for driver in nowcast_overlay["signal_drivers"][:4])
        )
    else:
        neutral_message = {
            "risk_on": tr("Nowcast overlay tilts risk-on."),
            "defensive": tr("Nowcast overlay tilts defensive."),
            "neutral": tr("Nowcast overlay is neutral."),
        }.get(str(nowcast_overlay["overlay_direction"]), tr("Nowcast overlay is neutral."))
        st.caption(neutral_message)
    if nowcast_overlay["ignored_drivers"]:
        st.caption(
            tr(
                "Available but ignored due to staleness: {items}",
                items=", ".join(humanize_label(item["series_id"]) for item in nowcast_overlay["ignored_drivers"][:4]),
            )
        )
    dim_left, dim_mid, dim_right = st.columns(3)
    dim_left.metric(overlay_metric_label("risk"), f"{float(nowcast_overlay['dimension_scores']['risk']):.2f}")
    dim_mid.metric(overlay_metric_label("rates"), f"{float(nowcast_overlay['dimension_scores']['rates']):.2f}")
    dim_right.metric(overlay_metric_label("inflation"), f"{float(nowcast_overlay['dimension_scores']['inflation']):.2f}")

    st.subheader(tr("Core Scores"))
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

    st.subheader(tr("Valuation"))
    valuation_left, valuation_mid, valuation_right = st.columns(3)
    valuation_left.metric(
        tr("Valuation Regime"),
        humanize_label(latest_valuation.get("valuation_regime")),
    )
    valuation_mid.metric(
        tr("Valuation Score"),
        f"{latest_valuation.get('valuation_score', float('nan')):.2f}",
    )
    valuation_right.metric(
        tr("Valuation Confidence"),
        humanize_label(latest_valuation.get("valuation_confidence")),
    )
    st.caption(
        tr(
            "Valuation inputs used: {items}",
            items=", ".join(
                humanize_label(item)
                for item in str(latest_valuation.get("valuation_inputs_used", "")).split(",")
                if item
            )
            or humanize_label("none"),
        )
    )
    missing_valuation_inputs = [
        humanize_label(item)
        for item in str(latest_valuation.get("valuation_inputs_missing", "")).split(",")
        if item
    ]
    if missing_valuation_inputs:
        st.caption(
            tr(
                "Valuation inputs still missing: {items}",
                items=", ".join(missing_valuation_inputs),
            )
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

    st.subheader(tr("Monitor Alerts"))
    region_alerts = monitor_alerts.loc[monitor_alerts["region"] == country].copy() if not monitor_alerts.empty else pd.DataFrame()
    if region_alerts.empty:
        st.caption(tr("No active alerts were generated for this region."))
    else:
        region_alerts["severity"] = region_alerts["severity"].apply(humanize_label)
        region_alerts[tr("Headline")] = region_alerts.apply(format_alert_headline, axis=1)
        region_alerts[tr("Detail")] = region_alerts.apply(format_alert_detail, axis=1)
        region_alerts = region_alerts.rename(columns={"severity": tr("Severity")})
        for _, row in region_alerts.head(4).iterrows():
            render_alert_box(row[tr("Severity")], row[tr("Headline")], row[tr("Detail")])

    st.subheader(tr("Latest Asset Preferences"))
    asset_cards = [
        ("equities", latest_assets.get("equities"), latest_assets.get("equities_score", float("nan"))),
        ("duration", latest_assets.get("duration"), latest_assets.get("duration_score", float("nan"))),
        ("gold", latest_assets.get("gold"), latest_assets.get("gold_score", float("nan"))),
        ("dollar", latest_assets.get("dollar"), latest_assets.get("dollar_score", float("nan"))),
    ]
    asset_cols = st.columns(4)
    for index, (asset_name, preference, score) in enumerate(asset_cards):
        with asset_cols[index]:
            render_html_card(
                humanize_country_asset_label(country, asset_name),
                humanize_label(preference),
                note=f"{tr('Score')}：{float(score):.1f}" if pd.notna(score) else f"{tr('Score')}：{tr('No data')}",
                pill=tr("Latest Asset Preferences"),
                pill_class=_pill_class(humanize_label(preference)),
            )
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
    history_frame = asset_data.tail(12).copy()
    if "date" in history_frame.columns:
        history_frame["date"] = history_frame["date"].apply(
            lambda value: format_display_value(
                pd.Timestamp(value).date().isoformat() if pd.notna(value) else None
            )
        )
    translation_columns = [
        "equities",
        "duration",
        "gold",
        "dollar",
        "allocation_confidence",
    ]
    for column in translation_columns:
        if column in history_frame.columns:
            history_frame[column] = history_frame[column].apply(humanize_label)
    rename_map = {
        "date": tr("Date"),
        "equities": humanize_country_asset_label(country, "equities"),
        "equities_score": f"{humanize_country_asset_label(country, 'equities')} {tr('Score')}",
        "duration": humanize_country_asset_label(country, "duration"),
        "duration_score": f"{humanize_country_asset_label(country, 'duration')} {tr('Score')}",
        "gold": humanize_country_asset_label(country, "gold"),
        "gold_score": f"{humanize_country_asset_label(country, 'gold')} {tr('Score')}",
        "dollar": humanize_country_asset_label(country, "dollar"),
        "dollar_score": f"{humanize_country_asset_label(country, 'dollar')} {tr('Score')}",
        "allocation_confidence": tr("Confidence"),
    }
    history_frame = history_frame.rename(columns=rename_map)
    display_columns = [
        column
        for column in [
            tr("Date"),
            humanize_country_asset_label(country, "equities"),
            f"{humanize_country_asset_label(country, 'equities')} {tr('Score')}",
            humanize_country_asset_label(country, "duration"),
            f"{humanize_country_asset_label(country, 'duration')} {tr('Score')}",
            humanize_country_asset_label(country, "gold"),
            f"{humanize_country_asset_label(country, 'gold')} {tr('Score')}",
            humanize_country_asset_label(country, "dollar"),
            f"{humanize_country_asset_label(country, 'dollar')} {tr('Score')}",
            tr("Confidence"),
        ]
        if column in history_frame.columns
    ]
    st.dataframe(history_frame.loc[:, display_columns], use_container_width=True, hide_index=True)


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
    monitor_alerts = load_optional_csv(Path("data/processed/monitor_alerts.csv"))
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
    overlay_alignment_text = (
        tr("Macro regime and market overlay are aligned.")
        if global_overlay["overlay_direction"] == "neutral" or humanize_label(latest.get("global_regime")) == humanize_label(global_overlay["overlay_direction"])
        else tr("Macro regime and market overlay are diverging.")
    )
    hero_title = tr("Current Core View")
    hero_body = (
        f"{tr('Current Global Macro Regime')}：{humanize_label(latest.get('global_regime'))}；"
        f"{tr('Investment Clock')}：{humanize_label(latest.get('investment_clock', latest.get('global_investment_clock')))}。"
        f"{overlay_alignment_text}"
    )
    render_alert_banner(hero_title, hero_body)
    st.caption(
        " · ".join(
            [
                f"{tr('System Update Date')}：{system_update_date}",
                f"{tr('Data Through Date')}：{_format_optional_date(latest.get('summary_date'))}",
                f"{tr('Latest Market-sensitive Input')}：{market_input_date}",
                f"{tr('Coverage Ratio')}：{float(latest.get('coverage_ratio', 0.0)):.0%}",
            ]
        )
    )

    if latest.get("global_regime") == "partial_view":
        st.warning(translate_runtime_text(latest.get("coverage_warning")))

    st.subheader(tr("Primary Signals"))
    top_left, top_mid = st.columns(2)
    with top_left:
        render_html_card(
            tr("Latest Global Regime"),
            humanize_label(latest.get("global_regime")),
            note=f"{tr('Selected Mode')}：{humanize_label(mode)}",
        )
    with top_mid:
        render_html_card(
            tr("Investment Clock"),
            humanize_label(latest.get("investment_clock", latest.get("global_investment_clock"))),
            note=f"{tr('Coverage Ratio')}：{float(latest.get('coverage_ratio', 0.0)):.0%}",
        )

    st.subheader(tr("Nowcast Overlay"))
    overlay_left, overlay_mid, overlay_right = st.columns(3)
    overlay_left.metric(tr("Nowcast Score"), f"{float(global_overlay['overlay_score']):.2f}")
    overlay_mid.metric(tr("Nowcast Direction"), humanize_label(global_overlay["overlay_direction"]))
    overlay_right.metric(tr("Confidence"), humanize_label(global_overlay["overlay_confidence"]))
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
    if global_overlay["overlay_drivers"]:
        formatted_drivers = []
        for item in global_overlay["overlay_drivers"][:5]:
            country, series_id, driver, _dimension = item.split(":", 3)
            formatted_drivers.append(
                format_nowcast_driver({"series_id": series_id, "driver": driver}, country=country)
            )
        st.caption(f"{tr('Driver Summary')}：" + "；".join(formatted_drivers))
    dim_left, dim_mid, dim_right = st.columns(3)
    dim_left.metric(overlay_metric_label("risk"), f"{float(global_overlay['dimension_scores']['risk']):.2f}")
    dim_mid.metric(overlay_metric_label("rates"), f"{float(global_overlay['dimension_scores']['rates']):.2f}")
    dim_right.metric(overlay_metric_label("inflation"), f"{float(global_overlay['dimension_scores']['inflation']):.2f}")

    coverage_left, coverage_mid = st.columns(2)
    coverage_left.metric(tr("Available Countries"), format_country_list(latest.get("countries_available")))
    coverage_mid.metric(tr("Missing Countries"), format_country_list(latest.get("countries_missing")))

    st.caption(tr("A country can be Ready locally but still not usable in the selected global view if it has no valid data on that mode's evaluation date."))

    st.subheader(tr("Monitor Alerts"))
    mode_alerts = monitor_alerts.loc[monitor_alerts["selected_mode"] == mode].copy() if not monitor_alerts.empty else pd.DataFrame()
    if mode_alerts.empty:
        st.caption(tr("No active alerts were generated for the selected mode."))
    else:
        mode_alerts["severity"] = mode_alerts["severity"].apply(humanize_label)
        mode_alerts[tr("Headline")] = mode_alerts.apply(format_alert_headline, axis=1)
        mode_alerts[tr("Detail")] = mode_alerts.apply(format_alert_detail, axis=1)
        mode_alerts = mode_alerts.rename(columns={"severity": tr("Severity")})
        for _, row in mode_alerts.head(4).iterrows():
            render_alert_box(row[tr("Severity")], row[tr("Headline")], row[tr("Detail")])

    st.caption(
        " · ".join(
            [
                tr("Configured weights: {weights}", weights=format_weight_map(latest.get("configured_weights"))),
                tr("Effective weights: {weights}", weights=format_weight_map(latest.get("effective_weights"))),
            ]
        )
    )
    available_countries = [item for item in str(latest.get("countries_available", "")).split(",") if item]
    if len(available_countries) == 1:
        st.info(tr("Only one country is available in the selected mode, so the effective global view is entirely driven by that market."))

    st.subheader(tr("Regional Quick Read"))
    quick_cols = st.columns(3)
    for index, country in enumerate(get_supported_countries()):
        regime_path = Path(f"data/processed/{country}_macro_regimes.csv")
        valuation_path = Path(f"data/processed/{country}_valuation_features.csv")
        regime_frame = load_optional_csv(regime_path)
        valuation_frame = load_optional_csv(valuation_path)
        latest_regime = regime_frame.iloc[-1] if not regime_frame.empty else pd.Series(dtype=object)
        latest_valuation = valuation_frame.iloc[-1] if not valuation_frame.empty else pd.Series(dtype=object)
        status_row = status_table.loc[status_table["country"] == country].iloc[0] if not status_table.empty and country in status_table["country"].values else pd.Series(dtype=object)
        quick_note = " / ".join(
            [
                f"{tr('Latest Date')}：{_format_optional_date(status_row.get('latest_date'))}",
                f"{tr('Confidence')}：{humanize_label(latest_regime.get('regime_confidence'))}",
                f"{tr('Valuation Status')}：{humanize_label(latest_valuation.get('valuation_regime', status_row.get('valuation_status')))}",
            ]
        )
        with quick_cols[index]:
            render_html_card(
                humanize_label(country),
                humanize_label(status_row.get("regime", latest_regime.get("regime"))),
                note=quick_note,
                pill=humanize_label(status_row.get("staleness_status")),
                pill_class="pill-negative" if status_row.get("staleness_status") == "very_stale" else "pill-neutral",
            )

    st.subheader(tr("Asset Tilt Board"))
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
    allocation_cards = st.columns(4)
    top_assets = latest_allocation = allocation.loc[allocation["as_of_mode"] == mode].copy()
    latest_allocation_date = top_assets["date"].max()
    latest_allocation = top_assets.loc[top_assets["date"] == latest_allocation_date].copy()
    lead_assets = ["global_equities", "duration", "gold", "dollar"]
    card_rows = latest_allocation.loc[latest_allocation["asset"].isin(lead_assets)].copy()
    for idx, asset in enumerate(lead_assets):
        row = card_rows.loc[card_rows["asset"] == asset]
        if row.empty:
            continue
        latest_row = row.iloc[0]
        with allocation_cards[idx]:
            render_html_card(
                humanize_global_asset_label(asset),
                humanize_label(latest_row["preference"]),
                note=translate_runtime_text(latest_row["reason"]),
                pill=f"{tr('Score')} {float(latest_row['score']):.1f} / {tr('Confidence')} {humanize_label(latest_row['confidence'])}",
                pill_class=_pill_class(humanize_label(latest_row["preference"])),
            )

    st.subheader(tr("Detailed Tables"))
    st.subheader(tr("Country Status"))
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
    latest_allocation["short_reason"] = latest_allocation["reason"].apply(summarize_reason_text)
    latest_allocation = latest_allocation.rename(
        columns={
            "asset": tr("Asset"),
            "preference": tr("Preference"),
            "score": tr("Score"),
            "confidence": tr("Confidence"),
        }
    )
    st.dataframe(
        latest_allocation.loc[:, [tr("Asset"), tr("Preference"), tr("Score"), tr("Confidence")]],
        use_container_width=True,
        hide_index=True,
    )
    with st.expander(tr("Full rationale by asset")):
        detailed_reasons = latest_allocation.loc[:, [tr("Asset"), "reason"]].rename(columns={"reason": tr("Reason")})
        if detailed_reasons.empty:
            st.caption(tr("No detailed rationale is available."))
        else:
            for _, row in detailed_reasons.iterrows():
                st.markdown(f"**{row[tr('Asset')]}**")
                st.write(row[tr("Reason")])
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
    comparison = build_mode_comparison(selected_mode=mode, history_dir=str(CHANGE_HISTORY_DIR))
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
        if "selected_mode" in mode_frequency.columns:
            mode_frequency["selected_mode"] = mode_frequency["selected_mode"].apply(humanize_label)
        if "state_type" in mode_frequency.columns:
            mode_frequency["state_type"] = mode_frequency["state_type"].apply(humanize_label)
        if "state" in mode_frequency.columns:
            mode_frequency["state"] = mode_frequency["state"].apply(humanize_label)
        if "share" in mode_frequency.columns:
            mode_frequency["share"] = mode_frequency["share"].map(
                lambda value: f"{float(value):.0%}" if pd.notna(value) else tr("No data")
            )
        mode_frequency = mode_frequency.rename(
            columns={
                "state_type": tr("State Type"),
                "state": tr("State"),
                "count": tr("Count"),
                "share": tr("Share"),
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
    if "_show_manual_page" not in st.session_state:
        st.session_state["_show_manual_page"] = False
    default_language = st.session_state.get("_display_language", ACTIVE_LANGUAGE)
    current_language = st.sidebar.selectbox(
        tr("Language", language=default_language),
        options=list(LANGUAGE_OPTIONS.values()),
        format_func=lambda value: "中文" if value == "zh" else "English",
        index=0 if default_language == "zh" else 1,
    )
    set_display_language(current_language)
    inject_dashboard_styles()
    options = ["global"] + get_supported_countries()
    selection = st.sidebar.selectbox(
        tr("View"),
        options=options,
        format_func=lambda value: tr("Global") if value == "global" else humanize_label(value),
    )

    st.sidebar.divider()
    if st.session_state["_show_manual_page"]:
        if st.sidebar.button(tr("Back to Dashboard"), use_container_width=True):
            st.session_state["_show_manual_page"] = False
            st.rerun()
    else:
        if st.sidebar.button(tr("Open User Guide"), use_container_width=True):
            st.session_state["_show_manual_page"] = True
            st.rerun()

    try:
        if st.session_state["_show_manual_page"]:
            render_user_manual_view()
        elif selection == "global":
            render_global_view()
        else:
            render_country_view(selection)
    except FileNotFoundError as exc:
        st.warning(str(exc))


if __name__ == "__main__":
    main()
