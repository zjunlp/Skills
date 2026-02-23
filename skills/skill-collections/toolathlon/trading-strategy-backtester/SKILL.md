---
name: trading-strategy-backtester
description: When the user wants to backtest a quantitative trading strategy based on specific signals, this skill implements strategy logic, executes simulated trades, calculates performance metrics, and generates comprehensive reports. It handles position sizing, transaction costs, holding periods, and computes key metrics like total return, annualized return, Sharpe ratio, win rate, and maximum drawdown. Triggers include backtesting requests, strategy evaluation, or trading performance analysis.
---
# Instructions

## 1. Understand the Request
- Identify the assets, signal logic, holding period, position sizing, and transaction costs from the user request or referenced documentation.
- If a `detail.md` or similar spec file is mentioned, read it first.
- Clarify any ambiguous parameters (e.g., "last 12 months" means 12 complete calendar months).

## 2. Fetch Required Data
- Use `yahoo-finance-get_historical_stock_prices` or equivalent data source to fetch historical price data for all specified assets/tickers.
- Ensure data covers the required backtest period plus any lookback needed for indicator calculation (e.g., 6-month rolling Z-score).
- Align data frequencies (e.g., monthly closes) and handle missing months if necessary.

## 3. Calculate Indicators & Generate Signals
- Implement the specified signal logic (e.g., Z-score thresholding).
- Calculate all required indicators (e.g., spreads, MoM%, Z-scores) as defined.
- Generate a clear signal for each period (e.g., Long Spread, Short Spread, Flat).

## 4. Execute Strategy Backtest
- Simulate trades based on signals:
    - Entry: At period end when signal is generated (non-Flat).
    - Exit: After the specified holding period (e.g., 1 month later).
    - Position Sizing: Apply specified weights (e.g., equal weight on legs).
    - Transaction Costs: Deduct specified round-trip cost from each trade's net PnL.
- Only one position at a time; close existing position before opening new one if strategy rules require.
- For spread trades (e.g., Long Spread = Long Asset A + Short Asset B), calculate leg returns and combine them according to position sizing.

## 5. Calculate Performance Metrics
Compute and report:
- **Total Return %** (compounded across all trades)
- **Annualized Return %** (based on actual backtest period length)
- **Sharpe Ratio (annualized)** (using monthly returns, annualized with √12)
- **Win Rate %** (percentage of profitable trades)
- **Maximum Drawdown %** (peak-to-trough decline in cumulative returns)
- **Number of Trades**
- **Backtest Period** (Start and End dates)

## 6. Prepare Output for Notion (or specified destination)
- If the target is Notion databases:
    1. Identify the correct database IDs using `notion-API-post-search` or similar.
    2. Check existing entries; delete placeholder/empty rows if necessary.
    3. Create new pages for each record (e.g., monthly summary rows, trade rows, metric row).
    4. **Important**: The available `notion-API-post-page` and `notion-API-patch-page` functions have limited schemas and may only allow updating the `title` property. If database properties (number, select, rich_text) cannot be updated via these functions, append the data as content blocks using `notion-API-patch-block-children` as a workaround. Each block should present the data clearly in a human-readable format.
- Structure output according to the database schemas described in the request (e.g., "Oil Market Summary" and "Spread Strategy Backtest").

## 7. Generate Summary Report
- Create a comprehensive markdown report summarizing:
    - Executive Summary
    - Data Overview & Key Observations
    - Methodology (indicator calculations, strategy rules)
    - Trade Details & Analysis
    - Performance Metrics
    - Conclusions & Recommendations
- Save the report to a file in the workspace (e.g., `summary_report.md`).
- Optionally, add a high-level summary to a main Notion page using content blocks.

## 8. Use Bundled Scripts for Complex Calculations
- For repetitive, complex, or error-prone calculations (e.g., Z-score rolling window, trade simulation, metric computation), use the bundled Python script `backtest_engine.py`.
- Adapt the script's parameters (tickers, signal thresholds, costs, etc.) to match the current request.

## Key Principles
- **Data Validation**: Check for missing periods or data inconsistencies; inform the user if gaps exist.
- **Transparency**: Document any assumptions or workarounds (e.g., Notion API limitations).
- **Reproducibility**: Ensure the backtest logic is clear and deterministic.
