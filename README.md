# global-macro-monitor

## Project Goal

`global-macro-monitor` is a lightweight Python project for investment research teams that want to track macroeconomic data, score macro factors, identify macro regimes, and inspect results through a dashboard.

The initial version focuses on a clean foundation instead of a fully automated production system. It provides a modular structure that can later be extended with more data vendors, richer factor definitions, and more advanced regime logic.

## Architecture

The project is organized into a small set of focused modules:

- `app/data`: data ingestion and normalization utilities
- `app/factors`: factor scoring logic for macro indicators
- `app/regime`: macro regime classification
- `app/valuation`: valuation features and valuation scoring
- `app/dashboard`: Streamlit dashboard for exploration
- `app/utils`: shared helpers such as logging
- `configs`: country, indicator, and weight definitions
- `data/raw`: raw downloaded or imported data
- `data/processed`: cleaned datasets and DuckDB files
- `tests`: starter test coverage

## Setup Instructions

### 1. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e .[dev]
```

### 3. Run the CLI starter

```bash
python main.py
```

### 4. Fetch US macro data from FRED

1. Create a free API key from the [FRED API Keys page](https://fredaccount.stlouisfed.org/apikeys).
2. Export it in your shell:

```bash
export FRED_API_KEY=your_fred_api_key_here
```

3. Run the fetch command:

```bash
python main.py fetch-us
```

The CSV files will be saved under `data/raw/fred/`.

## Multi-Country Manual Data

The project now supports:

- `us`
- `china`
- `eurozone`

For non-US markets, or for any series that is not yet automated, place manual CSV files under:

- `data/raw/manual/us/`
- `data/raw/manual/china/`
- `data/raw/manual/eurozone/`

Expected CSV format:

- `date`
- `value`
- `series_id`

### 5. Build feature-engineered monthly data

After raw FRED CSV files are available, build the standardized feature panel:

```bash
python main.py build-us-features
```

Generalized multi-country commands:

```bash
python main.py build-country-features --country us
python main.py build-country-features --country china
python main.py build-country-features --country eurozone
```

This step reads `data/raw/fred/*.csv`, aligns them by month, computes features such as YoY inflation, 3-month unemployment averages, and 3-month rate changes, and writes:

- `data/processed/us_macro_features.csv`

### 6. Classify macro regimes

```bash
python main.py classify-us-regime
```

Generalized multi-country command:

```bash
python main.py classify-country-regime --country us
```

This step loads or builds the feature panel, computes Growth, Inflation, and Liquidity scores, classifies each month into a macro regime, and writes:

- `data/processed/us_macro_regimes.csv`

The current classifier is a monitoring-oriented regime engine:

- factor scores use history-aware standardization instead of full-sample normalization
- regime labels include a small neutral band so signals near zero do not flip too aggressively
- the processed regime file now includes `regime_raw`, `regime_confidence`, and `regime_note` for auditability

### 7. Run the dashboard

```bash
streamlit run app/dashboard/app.py
```

The dashboard now supports country views and a global view. Country pages show the latest regime, liquidity overlay, valuation regime, asset preferences, and score charts. The global page shows the latest weighted regime, investment clock quadrant, latest country regime table, and global score history.

### 8. Run tests

```bash
pytest
```

## Streamlit Deployment And Automatic Refresh

This project can be deployed to Streamlit Community Cloud and refreshed automatically through GitHub Actions.

### Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from your repository.
4. Use this entrypoint:

```text
app/dashboard/app.py
```

### Streamlit secrets

In your Streamlit app settings, add these root-level secrets:

```toml
FRED_API_KEY = "your_fred_api_key"
ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data"
EUROSTAT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
OECD_API_BASE = "https://sdmx.oecd.org/public/rest/data"
APP_ENV = "production"
REQUEST_TIMEOUT = "15"
DUCKDB_PATH = "data/processed/global_macro.duckdb"
```

`TUSHARE_TOKEN` is optional in the current setup because the China pipeline is AkShare-first.

### GitHub Actions automatic refresh

The repository includes a workflow at `.github/workflows/refresh-monitor.yml`.

It runs:

```bash
python main.py refresh-monitor
pytest
```

on a daily schedule and can also be run manually from the GitHub Actions page.

To enable it, add these GitHub repository secrets:

- `FRED_API_KEY`
- `TUSHARE_TOKEN` (optional)

The workflow:

- refreshes US, China, and Eurozone data
- rebuilds macro, valuation, allocation, and consensus outputs
- commits updated data files back to `main`

Once GitHub pushes new data files, Streamlit Community Cloud will redeploy and show the latest results.

## Valuation And Asset Mapping

Macro regime alone is often not enough for investment interpretation. The same `goldilocks` or `slowdown` regime can imply very different asset decisions when equities are already expensive or when real yields are unusually restrictive.

The valuation layer adds a second lens on top of macro:

- Buffett indicator
- real yield
- term spread
- equity risk proxy
- credit spread proxy

For V1, the system uses existing macro features where possible and allows optional manual CSV inputs under `data/raw/manual/` for anything not already available.

Build the valuation layer with:

```bash
python main.py build-us-valuation
```

This writes:

- `data/processed/us_valuation_features.csv`

Then map macro plus valuation into simple asset preferences with:

```bash
python main.py map-us-assets
```

This writes:

- `data/processed/us_asset_preferences.csv`

The asset mapping layer combines macro regime, liquidity regime, and valuation score to produce transparent preferences for equities, duration, gold, and the dollar.

Use the generalized commands when needed:

```bash
python main.py build-country-valuation --country us
python main.py map-country-assets --country us
python main.py build-global-summary
python main.py build-global-allocation
python main.py run-global-monitor
```

Generated outputs include:

- `data/processed/{country}_macro_features.csv`
- `data/processed/{country}_macro_regimes.csv`
- `data/processed/{country}_valuation_features.csv`
- `data/processed/{country}_asset_preferences.csv`
- `data/processed/global_macro_summary.csv`
- `data/processed/global_allocation_map.csv`

## Global Allocation Map

The global macro monitor now adds a simple cross-asset interpretation layer on top of the global summary.

This layer translates macro state into investable preferences for:

- global equities
- United States equities
- China equities
- Eurozone equities
- duration
- gold
- dollar
- commodities

Build it with:

```bash
python main.py build-global-allocation
```

The output is saved to:

- `data/processed/global_allocation_map.csv`

Each row contains:

- `date`
- `asset`
- `preference`
- `score`
- `confidence`
- `reason`

The logic is intentionally simple and explicit:

- `Goldilocks` plus fair or cheap valuation leans bullish equities
- `Slowdown` leans bullish duration
- `Stagflation` leans bullish gold and cautious duration
- `Reflation` leans bullish commodities and less constructive on long duration

Confidence is separate from preference direction. Confidence falls when:

- global country coverage is incomplete
- country inputs are stale
- valuation inputs are missing

This is useful because a bullish signal built from fresh multi-country data should be treated differently from the same signal built on partial or stale inputs.

The confidence rules are conservative by design:

- global assets cannot be `High` if any contributing region is `Stale` or `Very stale`
- partial coverage caps confidence at `Medium`, and deep partial coverage can push it to `Low`
- missing valuation inputs downgrade confidence by at least one level
- country assets with `Very stale` data, or that are not usable in the selected mode, are capped at `Low`

Reasons are also mode-aware:

- in `Latest available`, reasons state that countries are contributing from different latest dates
- in `Last common date`, reasons state the shared evaluation date
- reasons only mention the countries actually used in that mode
- reasons say explicitly whether valuation was used or missing

Building the global allocation map also writes:

- `data/processed/global_change_log.csv`
- `data/processed/regime_evaluation_summary.csv`
- `data/processed/global_summary_history.csv`
- `data/processed/global_allocation_history.csv`
- `data/processed/regime_frequency_summary.csv`
- `data/processed/regime_transition_matrix.csv`
- `data/processed/regime_forward_return_summary.csv`
- `data/processed/confidence_bucket_summary.csv`

## Consensus Deviation

The monitor also supports a lightweight `Consensus Deviation` layer. This does not try to trade the news. Instead, it compares the model's current macro view with mainstream public narratives across:

- growth
- inflation
- policy bias

Supported regions:

- `us`
- `eurozone`
- `china`

### Consensus note workflow

Save raw consensus notes under:

- `data/raw/consensus/us/`
- `data/raw/consensus/eurozone/`
- `data/raw/consensus/china/`

Supported file types:

- `.md`
- `.txt`
- `.json`

Simple text-note format:

```text
source_name: Reuters
source_type: media
date: 2026-03-20
title: US growth still resilient

Analysts describe resilient growth, easing inflation, and a dovish policy turn.
```

Normalize and ingest one region's notes with:

```bash
python main.py ingest-consensus-notes --region us --path data/raw/consensus/us
```

The monitor also supports automatic fetching of public official/institution notes:

```bash
python main.py fetch-consensus-sources --region us
python main.py fetch-consensus-sources --region eurozone
python main.py fetch-consensus-sources --region china
```

This writes auto-fetched raw notes under:

- `data/raw/consensus/us/auto/`
- `data/raw/consensus/eurozone/auto/`
- `data/raw/consensus/china/auto/`

The current automatic source mix is intentionally conservative:

- `us`: Federal Reserve RSS plus BIS central-bank speeches RSS
- `eurozone`: ECB RSS plus BIS central-bank speeches RSS
- `china`: PBOC English press releases plus BIS central-bank speeches RSS

This keeps the pipeline transparent and official-source heavy. Media and sell-side notes can still be added manually.

If you want to refresh the full stack in one go, use:

```bash
python main.py refresh-monitor
```

This best-effort command will:

- fetch country macro data where available
- rebuild macro, valuation, asset, and global outputs
- refresh consensus notes, snapshots, and deviations
- refresh descriptive evaluation tables

Then build snapshots and deviations:

```bash
python main.py build-consensus-snapshots
python main.py build-consensus-deviation
```

Generated outputs:

- `data/processed/consensus_notes.csv`
- `data/processed/consensus_snapshots.csv`
- `data/processed/consensus_deviation.csv`
- `data/processed/consensus_diagnostics.csv`

### Scoring logic

The parser is intentionally simple and rules-based. It maps recent public notes into:

- `growth_view`
- `inflation_view`
- `policy_bias_view`
- `confidence`

Recent notes carry more weight than old notes:

- within 14 days: full weight
- 15 to 30 days: reduced weight
- older than 30 days: ignored by default

Official central-bank communication is weighted slightly higher than generic commentary, and higher-confidence note classifications also receive more weight.

Deviation scores are centered and directional:

- `growth_deviation_score`
  - `+` means the model is more growth-positive than consensus
  - `-` means the model is more growth-negative than consensus
- `inflation_deviation_score`
  - `+` means the model sees less inflation risk than consensus
  - `-` means the model sees more inflation risk than consensus
- `policy_deviation_score`
  - `+` means the model is more dovish/easy than consensus
  - `-` means the model is more hawkish/tight than consensus

This layer is descriptive, not predictive. Its main use is to flag when the model's macro state is materially out of line with mainstream public narratives, while keeping the logic transparent and auditable through `consensus_diagnostics.csv`.

`global_change_log.csv` highlights:

- whether the global regime changed versus the prior observation
- whether the investment clock changed
- which assets changed preference
- which assets changed confidence
- which countries changed local regime

The dashboard renders these into three mutually exclusive `What Changed` sections:

- `Regime Changes`
- `Preference Changes`
- `Confidence Changes`

This avoids duplicating the same event across multiple sections. The page also shows summary counts and a short `Why It Changed` list with the main drivers, such as stale country inputs, missing valuation inputs, regime transitions, or investment clock transitions.

Snapshot history is now persistent. Each run appends the latest global summary and allocation snapshot to the history files instead of overwriting prior runs. Change detection compares the current snapshot only against the most recent prior comparable snapshot for the same `as_of_mode`.

If no prior comparable snapshot exists, the dashboard will say:

- `No prior snapshot is available yet for this mode.`

If older history exists but the schema changed, the dashboard will say:

- `Change history for this mode starts from the latest schema version.`

`regime_evaluation_summary.csv` is intentionally lightweight. It is not a trading backtest. It only provides:

- regime counts
- regime transition counts
- average next 1m / 3m / 6m forward return for optional proxy series when available

If no proxy return series are available, the evaluation file is still written, but return fields stay empty and the note column explains why.

## Regime Evaluation

The project now includes a lightweight `Regime Evaluation` layer for descriptive historical validation. It is designed for research use, not as a full portfolio simulator or execution backtest.

Run:

```bash
python main.py evaluate-regimes
python main.py evaluate-confidence
```

This produces:

- `data/processed/regime_frequency_summary.csv`
- `data/processed/regime_transition_matrix.csv`
- `data/processed/regime_forward_return_summary.csv`
- `data/processed/confidence_bucket_summary.csv`

The evaluation uses:

- global summary history
- global allocation history
- any available proxy return series from `data/processed/returns/` or `data/raw/manual/returns/`

Forward return summaries are descriptive. For each regime or investment clock state, the system reports:

- count
- average forward return
- median forward return
- hit ratio

for:

- 1 month
- 3 months
- 6 months

Interpret these tables carefully:

- sparse history can make averages unstable
- proxy coverage may be incomplete
- results describe historical tendencies, not robust tradable edge
- partial proxy availability is allowed, so some assets may appear while others do not

## Partial Coverage And Manual Onboarding

Global output is only as reliable as the country coverage behind it. The global summary now tracks:

- available countries on each date
- missing countries on each date
- coverage ratio
- configured weights and effective renormalized weights

If coverage falls below `0.70`, the system does not pretend the global signal is fully reliable. Instead it sets:

- `global_regime = partial_view`
- `investment_clock = partial_view`
- `coverage_warning = "Global summary is based on incomplete country coverage."`

This is meant to prevent overconfidence when only one market, or a small subset of markets, has usable data.

To add China or Eurozone manual data, place one CSV per series under:

- `data/raw/manual/china/`
- `data/raw/manual/eurozone/`

Example template files already exist:

- `data/raw/manual/china/cpi.csv`
- `data/raw/manual/china/pmi.csv`
- `data/raw/manual/eurozone/cpi.csv`
- `data/raw/manual/eurozone/pmi.csv`

Each file should use:

```csv
date,value,series_id
2024-01-01,0.2,cpi
2024-02-01,0.4,cpi
2024-03-01,0.1,cpi
```

When too many countries are missing, the dashboard will show a warning banner and clearly label the global view as partial.

The global page now supports two time-alignment modes:

- `Latest available`: each country contributes using its own latest valid observation
- `Last common date`: only uses the most recent date shared across contributing countries

This matters because stale data can materially change interpretation. A country can still be locally `Ready`, but if its most recent valid observation is too old or does not line up with the selected global mode, it may not be usable in the latest global aggregate.

The system also tracks staleness for each country:

- `Fresh`: 0 to 90 days stale
- `Stale`: 91 to 180 days stale
- `Very stale`: more than 180 days stale

Use the selected mode carefully:

- `Latest available` gives the broadest inclusion, but countries may contribute older observations
- `Last common date` is stricter, but can move the global view further back in time

For manual fallback countries, the minimum required V1 series are:

- `cpi`
- `pmi`
- `policy_rate`
- `yield_10y`

Validate a manual country before trying to classify it:

```bash
python main.py validate-manual-data --country china
python main.py validate-manual-data --country eurozone
```

The workflow is:

1. Add the minimum CSV files under `data/raw/manual/{country}/`
2. Run `validate-manual-data`
3. Once the country is marked ready, run:

```bash
python main.py build-country-features --country china
python main.py classify-country-regime --country china
```

## Eurozone And China API Ingestion

The project now supports API-based raw ingestion for non-US countries while keeping the existing downstream feature, regime, valuation, and allocation pipeline unchanged.

Supported adapters:

- `ECB` for Eurozone policy-rate, yield, and liquidity proxies
- `Eurostat` for Eurozone inflation and labor proxies
- `OECD` as an optional Eurozone fallback for growth or confidence proxies
- `AkShare` as the primary China programmatic source
- `China NBS / National Data style sources` as a secondary China fallback
- `ChinaMoney / ChinaBond style public rates sources` as a secondary China fallback
- `IMF` as a simple fallback for selected China macro gaps

Set environment variables in your shell or `.env`:

```bash
export ECB_API_BASE=
export EUROSTAT_API_BASE=
export OECD_API_BASE=
```

Fetch API-backed raw data with:

```bash
python main.py fetch-country-api-data --country eurozone
python main.py fetch-country-api-data --country china
```

Fetched files are saved under:

- `data/raw/api/eurozone/`
- `data/raw/api/china/akshare/`
- `data/raw/api/china/nbs/`
- `data/raw/api/china/rates/`
- `data/raw/api/china/imf/`
- `data/raw/api/china/normalized/`

Each normalized raw file uses:

- `date`
- `value`
- `series_id`
- `country`
- `source`
- `frequency`
- `release_date`
- `ingested_at`

Indicator configuration is source-driven in `configs/indicators.yaml`. Each indicator can specify:

- `source`
- `source_series_id`
- `source_hint`
- `frequency`
- `transform`
- `fallback_source`
- `required_for_minimum_regime`

Downstream feature generation now reads from:

- `data/raw/fred/` for US FRED data
- `data/raw/api/eurozone/` for Eurozone API-ingested data
- `data/raw/api/china/normalized/` for normalized China API data
- `data/raw/manual/{country}/` as a fallback when API data is unavailable

For China, the source priority is:

1. AkShare
2. existing public-source adapters
3. manual CSV fallback

China minimum regime inputs for V1 are:

- `cpi`
- `pmi`
- `policy_rate`
- `yield_10y`

Fetch and validate China data with:

```bash
python main.py fetch-country-api-data --country china
python main.py validate-country-data --country china
python main.py rebuild-country-normalized-data --country china
```

`validate-country-data` reports whether China is ready for:

- feature building
- regime classification
- valuation mapping

It also prints:

- normalized files found
- series ids found
- missing required ids
- whether API-first mode is active

China public-source ingestion is intentionally pragmatic rather than exhaustive:

- some public endpoints are inconsistent or undocumented
- release dates are not always provided
- IMF fallback is used only where practical
- manual fallback remains available when API coverage is incomplete

Current limitations:

- live API coverage depends on external permissions, credentials, and endpoint stability
- some Eurozone and China series still rely on manual fallback if the configured API source is unavailable
- tests mock all API calls; they validate parsing and normalization, not vendor uptime
- the China adapter currently targets a practical V1 subset rather than a full macro coverage map

## Roadmap For V1

- Persist cleaned time series into DuckDB
- Expand factor scoring to use rolling z-scores and percentile ranks
- Add cross-country comparison views
- Improve regime classification with more robust rule sets
- Add valuation overlays for equity and rates assets
- Add scheduled refresh jobs and better data quality checks
