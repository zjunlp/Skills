---
name: google-sheets-market-data-organizer
description: Creates and populates Google Sheets with structured financial or market data. It creates spreadsheets with custom names and worksheets, formats data according to specified templates (like CSV examples), and writes organized datasets. It handles spreadsheet creation, sheet renaming, and bulk data insertion.
---
# Instructions

## 1. Understand the Request
The user will request to create a Google Sheet with a specific name and a worksheet with a specific title. They will specify a list of stock tickers and a date range. They will reference an example CSV file for the required data format. The final step is to post a notification to a specific Notion page.

**Key Requirements:**
*   Use **original prices** (set `auto_adjust: false` when fetching data).
*   **Exclude non-trading days** automatically (the data source handles this).
*   Follow the column format from the example: `Ticker,Date,Open,High,Low,Close,Volume`.
*   Post a specific notification to a Notion page upon completion.

## 2. Execute the Workflow

### Phase 1: Preparation & Data Fetching
1.  **Locate the Example Template:** Read the provided example CSV file (e.g., `example.csv`) to confirm the required data structure.
2.  **Find the Target Notion Page:** Search for the specified Notion page (e.g., "Quant Research") and note its Page ID.
3.  **Create the Google Sheet:** Create a new spreadsheet with the requested title (e.g., "2025_Market_Data").
4.  **Rename the Worksheet:** Rename the default sheet in the new spreadsheet to the requested worksheet title (e.g., "Jun-Jul_2025").
5.  **Fetch Stock Data:** For each specified ticker (e.g., AAPL, TSLA, NVDA, META), retrieve historical daily price data for the given date range. **Crucially, ensure `auto_adjust` is set to `false`.**

### Phase 2: Data Processing & Population
6.  **Process the Data:** Use the bundled `process_data.py` script to transform the fetched JSON data into a list of rows matching the example CSV format (`Ticker, Date, Open, High, Low, Close, Volume`). The script handles date formatting and data structuring.
7.  **Write Data to Sheet:** Use the `google_sheet-append_to_sheet` tool to write the processed data rows to the target worksheet. Write the header row first, followed by all data rows.

### Phase 3: Notification
8.  **Generate Google Sheet URL:** Construct the Google Sheet URL using the spreadsheet ID (format: `https://docs.google.com/spreadsheets/d/{spreadsheetId}`).
9.  **Update Notion Page:**
    *   Append a new line to the target Notion page with the exact text: `Google Sheet : {url}`.
    *   Add a comment at the top of the same page with the exact text: `"Monthly market data is ready. The reporting team can view it directly"`.

## 3. Error Handling & Validation
*   If the example CSV cannot be read, inform the user and ask for clarification on the format.
*   If the target Notion page is not found, search again with a more specific filter (`"filter": {"property":"object","value":"page"}`) and confirm the correct page ID before proceeding.
*   Verify the stock data for each ticker was fetched successfully before processing.
*   After writing to the sheet, consider adding a final step to read back a sample cell to confirm the data was written correctly.
