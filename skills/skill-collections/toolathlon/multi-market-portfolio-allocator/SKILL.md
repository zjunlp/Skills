---
name: multi-market-portfolio-allocator
description: When the user needs to build an investment portfolio across multiple markets (US, Hong Kong, A-shares) with specific allocation ratios, total capital constraints, and market-specific trading rules. This skill reads stock lists from Excel files, fetches current/latest stock prices and exchange rates from financial APIs, categorizes stocks by market type, calculates position sizes based on allocation percentages, applies market-specific constraints (whole shares for US/HK stocks, 100-share lots for A-shares), performs currency conversions, and generates a complete position-building plan with detailed calculations. It also updates Excel files with calculated stock codes and position sizes. Triggers include requests for portfolio allocation, position building, stock purchase planning with capital allocation ratios, and Excel-based stock list processing.
---
# Instructions

## 1. Understand the User Request
The user will provide:
- A total capital amount (e.g., $1,000,000 USD).
- An allocation ratio for US stocks, Hong Kong stocks, and A-shares (e.g., 4:3:3).
- An Excel file (`stock.xlsx` by default) containing a list of stock names. The file has columns: `Stock_name`, `Stock_code` (initially empty), and `Initial_position_size` (initially empty).
- Specific instructions on which stocks belong to which market (e.g., "Alibaba as a Hong Kong stock, BYD, Ping An Insurance and Wuxi AppTec as A-shares").
- Trading rules: Purchases in whole shares for US/HK stocks, and in multiples of 100 shares (lots) for A-shares.
- The request to calculate based on "today's" opening prices, or the latest trading day's data if today is not a trading day.

Your goal is to produce a detailed position-building plan and fill the user's Excel file.

## 2. Core Workflow

### Phase 1: Data Extraction & Categorization
1.  **Read the Excel File:** Use the `excel-read_data_from_excel` tool to load the stock list. Identify the stock names.
2.  **Categorize Stocks:** Based on the user's instructions and common knowledge, map each stock name to its market (US, Hong Kong, A-share) and determine its standard ticker symbol.
    - **US Stocks:** Typically trade on US exchanges (NASDAQ, NYSE). Use standard tickers (e.g., MSFT, AAPL).
    - **Hong Kong Stocks:** Use the `.HK` suffix (e.g., 9988.HK for Alibaba).
    - **A-Shares:** Use the `.SS` (Shanghai) or `.SZ` (Shenzhen) suffix (e.g., 601318.SS for Ping An Insurance).
    - *If a stock could belong to multiple markets (like Alibaba), strictly follow the user's explicit instruction.*

### Phase 2: Data Fetching
1.  **Fetch Stock Prices:** For each categorized stock, use the `yahoo-finance-get_historical_stock_prices` tool to get the last 5 days of daily data (`period="5d"`, `interval="1d"`). Extract the **Open** price from the most recent data point (index 0). This represents "today's" or the latest trading day's opening price.
2.  **Fetch Exchange Rates:** Use the same tool to get exchange rates for `USDCNY=X` (USD to CNY) and `USDHKD=X` (USD to HKD). Extract the latest **Open** rate.

### Phase 3: Calculation & Allocation
1.  **Calculate Market Allocations:** Apply the user's ratio (e.g., 4:3:3) to the total capital to determine the USD amount allocated to each market.
2.  **Convert to Local Currency:** Convert the HK allocation to HKD and the A-share allocation to CNY using the fetched exchange rates.
3.  **Calculate Per-Stock Allocation:** Divide each market's local currency allocation equally among the stocks in that market.
4.  **Calculate Share Quantities:**
    - **US/HK Stocks:** `floor(per_stock_allocation_local / stock_price)`. Result must be a whole integer.
    - **A-Shares:** `floor(per_stock_allocation_local / (stock_price * 100)) * 100`. Result must be a multiple of 100.
5.  **Calculate Costs & Summary:** Compute the cost in local currency and USD for each position. Sum up totals for each market and overall. Calculate remaining cash.

### Phase 4: Output & File Update
1.  **Generate Position Plan:** Present a clear summary including:
    - Allocation breakdown and exchange rates used.
    - A table for each market showing stock, ticker, shares, price, and cost.
    - A final investment summary (totals per market, total invested, remaining cash).
2.  **Update Excel File:** Use the `excel-write_data_to_excel` tool to write the calculated **ticker symbols** and **share quantities** (numbers only) back into the `Stock_code` and `Initial_position_size` columns of the original Excel file, starting at cell B2.
3.  **Verify Update:** Optionally re-read the file to confirm the data was written correctly.

## 3. Key Rules & Constraints
- **Accuracy:** Double-check ticker symbol mappings and currency conversions.
- **Precision:** Share calculations must strictly adhere to `floor` for US/HK and `floor` to the nearest 100 for A-shares. Do not round up.
- **Clarity:** The final plan should be easy for the user to review and execute.
- **Stop:** Once the plan is delivered and the Excel file is updated, claim the task as done. Do not perform any additional, unrequested actions.

## 4. Handling Ambiguity
- If a stock name is ambiguous and the user didn't specify, use your best judgment based on the most common/liquid listing and state your assumption.
- If financial data is unavailable for a ticker, note it in the plan and exclude it from calculations, adjusting the per-stock allocation for the remaining stocks in that market accordingly.
- If the user's Excel file has a different structure, adapt the read/write cell ranges as needed.

For detailed ticker mappings, calculation logic, and error handling examples, refer to the bundled scripts and references.
