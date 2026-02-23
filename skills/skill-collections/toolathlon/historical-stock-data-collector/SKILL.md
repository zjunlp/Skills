---
name: historical-stock-data-collector
description: When the user requests historical stock price data for specific tickers over a date range, this skill retrieves daily trading data (open, high, low, close, volume) from financial APIs. It handles multiple tickers simultaneously, excludes non-trading days automatically, and can specify whether to use adjusted or unadjusted prices. Triggers include requests for 'stock data', 'historical prices', 'daily trading data', or specific ticker symbols like AAPL, TSLA, etc.
---
# Instructions

## 1. Understand the Request
- Identify the target tickers (e.g., AAPL, TSLA, NVDA, META).
- Identify the date range (e.g., June-July 2025).
- Clarify if the user wants adjusted (`auto_adjust: true`) or unadjusted (`auto_adjust: false`) prices. Default to unadjusted prices unless specified.
- Identify any specific output format or destination (e.g., Google Sheet, CSV file, Notion page).

## 2. Retrieve Historical Data
For each identified ticker:
- Use the `yahoo-finance-get_historical_stock_prices` tool.
- Set `interval` to `"1d"`.
- Set `auto_adjust` based on user preference.
- The API automatically excludes non-trading days.

## 3. Process and Format Data
- The required output columns are: `Ticker`, `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
- The `Date` should be formatted as `YYYY-MM-DD`.
- Combine data from all tickers into a single list, sorted by Date and then by Ticker.
- Use the provided `scripts/process_stock_data.py` for reliable, repeatable data transformation.

## 4. Output to Specified Destination
### If the destination is a Google Sheet:
1.  Create a new spreadsheet or identify an existing one using `google_sheet-create_spreadsheet` or `google_sheet-get_spreadsheet`.
2.  Ensure the target worksheet exists and is correctly named. Use `google_sheet-rename_sheet` if needed.
3.  Write the formatted data to the sheet starting from cell A1. Use `google_sheet-write_to_cells`.
4.  Note the final spreadsheet URL.

### If the destination is a Notion Page:
1.  Locate the target Notion page using `notion-API-post-search`.
2.  Append a new line with the text `Google Sheet : {url}` or the relevant data summary using `notion-API-append-block`.
3.  If requested, add a comment/note at the top of the page using `notion-API-append-block` with the exact specified text.

## 5. Error Handling & Validation
- If a ticker returns no data for the date range, log a warning and continue with other tickers.
- Verify the final dataset is not empty before writing to the output destination.
- If writing to Google Sheets, check the operation was successful.

## Key Principles
- **Conciseness**: Use scripts for deterministic data processing.
- **Reliability**: The financial API handles date filtering; do not manually calculate trading days.
- **Clarity**: Confirm data parameters (like `auto_adjust`) with the user if ambiguous.
