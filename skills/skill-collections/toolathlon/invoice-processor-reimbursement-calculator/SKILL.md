---
name: invoice-processor-reimbursement-calculator
description: Scans a workspace for invoice PDFs, extracts vendor, date, amount, and currency, fetches historical exchange rates, converts all amounts to a target currency (default CNY), calculates totals including tax, and outputs a CSV summary and a JSON grand total.
---
# Instructions

## 1. Scan Workspace & Identify Invoices
- Use `filesystem-directory_tree` to list all files in the user's personal workspace (typically `/workspace` or user-specified).
- Filter the list to identify potential invoice files. Prioritize files with extensions `.pdf` and names containing keywords like `invoice`, `receipt`, `flight`, or `itinerary`. Ignore temporary directories (e.g., `.pdf_tools_tempfiles`).

## 2. Extract Data from Invoice PDFs
- For each identified PDF invoice file:
    - Use `pdf-tools-read_pdf_pages` to extract text from the first few pages (typically pages 1-5).
    - Analyze the extracted text to find the following key fields:
        - **Vendor**: Look for company names, "Bill To", "Seller", or "Issuing Airline".
        - **Date**: Look for "Date", "Date of issue", "Billed On", "Transaction date". Parse into YYYY-MM-DD format.
        - **Original Amount & Currency**:
            - Identify the total amount. Look for patterns like `$XX.XX USD`, `CNY XXX.XX`, `SGD XXX.XX`, "Total:", "Amount due:", or "FARE:" + "TAX:".
            - **Crucially, include any tax amounts.** If separate fare and tax lines are found (common in flight itineraries), sum them to get the total original amount.
            - Record the currency code (USD, CNY, SGD, etc.).
        - **Filename**: Record the source filename.

## 3. Handle Currency Conversion
- For each invoice **not** already in the target currency (Chinese Yuan, CNY):
    - Use `yahoo-finance-get_stock_price_by_date` to fetch the historical exchange rate.
    - **Ticker Format**: Use `USDCNY=X` for USD to CNY, `SGDCNY=X` for SGD to CNY, etc.
    - **Date**: Use the invoice date from Step 2. Set `find_nearest: true` to handle non-trading days.
    - Use the `close` price from the returned data as the exchange rate.
- For invoices already in CNY, set the exchange rate to `1.0000`.
- Calculate `amount_in_cny = original_amount * exchange_rate`.

## 4. Generate the Summary CSV File
- Check if a file named `invoice_summary.csv` already exists in the workspace to understand its format.
- Create/overwrite `invoice_summary.csv` in the user's workspace.
- Use `filesystem-write_file` to write the data with the following header and one row per invoice:
    