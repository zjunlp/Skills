---
name: multi-currency-expense-aggregator
description: When the user needs to calculate total expenses across multiple individuals with costs in different currencies and convert them to a target currency using historical exchange rates. This skill handles reading multiple CSV/JSON files containing expense data, fetching exchange rates from financial APIs for specific dates, performing currency conversions, aggregating totals per person, and generating structured JSON reports with calculated results. Key triggers include requests for 'total trip cost', 'expense calculation with exchange rates', 'convert multiple currencies to [target currency]', or tasks involving reading multiple expense files with USD, EUR, TRY, CNY amounts.
---
# Instructions

## Objective
Calculate the total expenses for a group of individuals from multiple files, convert all amounts to a target currency (e.g., CNY) using historical exchange rates from a specific date, and output a structured JSON report.

## Core Workflow
1.  **Identify Inputs:** Determine the target currency, the historical date for exchange rates, the list of individuals, and the location of their expense files (CSV/JSON). The user may provide this explicitly or it may need to be inferred from the workspace.
2.  **Read Data:** List and read all relevant expense files from the workspace. Files are typically CSVs with columns for `Date`, `Activity/Description`, `Cost/Amount`, and `Currency`, or similar variations.
3.  **Fetch Exchange Rates:** Use the Yahoo Finance tool (`yahoo-finance-get_stock_price_by_date`) to get the closing exchange rates for the required currency pairs (e.g., `USDCNY=X`, `EURCNY=X`) on the specified historical date. For indirect pairs (e.g., TRY/CNY), you may need to fetch a cross-rate (e.g., `TRYUSD=X`) and calculate the target rate.
4.  **Parse & Aggregate:** For each individual's file:
    *   Parse the data, summing amounts by currency (USD, EUR, TRY, CNY, etc.).
    *   Convert each currency sum to the target currency using the fetched rates.
    *   Calculate the individual's total in the target currency.
5.  **Calculate Grand Total:** Sum all individual totals.
6.  **Generate Output:** Create or update a specified JSON file (e.g., `calculation.json`) with the exchange rates used and each individual's total in the target currency, along with the grand total. Format numbers appropriately (e.g., rounded to 2 decimal places).

## Key Considerations & Edge Cases
*   **File Format Variability:** Expense files may have different column names, structures, or delimiters. Be prepared to handle variations like `Cost` vs `Amount`, or files where costs are spread across multiple currency columns.
*   **Missing Direct Rates:** If a direct exchange rate pair (e.g., `TRYCNY=X`) is not available, you must calculate it using a common intermediary (e.g., `TRYUSD=X` and `USDCNY=X`).
*   **Date Handling:** The exchange rate date is critical. Ensure the correct date is used for the `yahoo-finance-get_stock_price_by_date` call. Use the `close` price for conversion.
*   **Error Handling:** If a file cannot be read or a rate cannot be fetched, note the issue and proceed if possible, or ask for clarification.
*   **Calculation Verification:** For complex sums, consider writing a temporary Python script to perform the arithmetic and verify totals, then clean it up.

## Typical Tool Sequence
1.  `filesystem-list_directory` - Explore the workspace.
2.  `filesystem-read_multiple_files` / `filesystem-read_file` - Load expense data.
3.  `yahoo-finance-get_stock_price_by_date` - Fetch historical FX rates.
4.  `terminal-run_command` - Optionally run a Python script for accurate bulk calculations.
5.  `filesystem-write_file` - Write the final JSON report.

## Output
The primary output is a JSON file (e.g., `calculation.json`) populated with keys for exchange rates and individual/group totals in the target currency. Provide a concise summary to the user.
