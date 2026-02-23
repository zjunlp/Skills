---
name: temporal-transaction-recorder
description: When the user requests to add transaction records from specific time periods (like 'last week', 'week before last', 'this month', etc.) to an accounting system or spreadsheet. This skill retrieves transaction data from memory or knowledge graph, analyzes date ranges relative to a reference date, checks existing account book files (Excel/CSV), filters transactions based on temporal criteria, and appends the filtered records to the appropriate file while excluding transactions from specified time periods. It handles purchase/sales transactions, date calculations, and maintains data integrity in financial ledgers.
---
# Skill: Temporal Transaction Recorder

## Purpose
Add transaction records from specified past time periods (e.g., "last week", "week before last") to an existing account book (Excel file), while explicitly excluding transactions from other periods (e.g., "this week"). The skill coordinates reading from a memory/knowledge graph, interpreting date-based constraints, reading the target file, and performing filtered appends.

## Core Workflow
1.  **Interpret Request & Calculate Dates**: Parse the user's request to identify the target time periods (to include) and the reference date (often "today"). Calculate the exact date ranges.
2.  **Retrieve Source Data**: Fetch transaction records from the memory/knowledge graph.
3.  **Locate & Inspect Target File**: Find the account book file (typically an Excel workbook) in the workspace and read its structure to determine where to append new data.
4.  **Filter & Transform Data**: Filter the retrieved transactions based on the calculated date ranges. Map the source data fields to the target file's column schema.
5.  **Append Data**: Write the filtered, transformed transaction records to the next available row in the target file.
6.  **Verify & Summarize**: Read back the newly added rows to confirm success and provide a clear summary to the user.

## Detailed Instructions

### 1. Parse the Request & Calculate Date Ranges
-   Extract the **reference date** (e.g., "Today is January 18, 2024"). If not provided, use the current date.
-   Identify the **inclusion periods** (e.g., "last week and the week before last").
-   Identify the **exclusion periods** (e.g., "do not include any transactions from this week").
-   Calculate concrete date ranges (start and end dates) for each period. Use ISO 8601 format (`YYYY-MM-DD`).
-   **Example Logic (for weeks, Monday-Sunday)**:
    -   This Week: Monday of the reference date's week to the reference date.
    -   Last Week: The 7-day period immediately preceding "This Week".
    -   Week Before Last: The 7-day period immediately preceding "Last Week".

### 2. Retrieve Source Transaction Data
-   Use the `memory-read_graph` tool (or equivalent) to fetch stored transaction entities.
-   Look for entities of type `Purchase Transaction` or `Sales Transaction`.
-   Extract key fields from each transaction's `observations`: `Date`, `Product`, `Quantity`, `Supplier`/`Customer`, `Amount`. Note the entity type to determine the `Type` (Purchase/Sales).

### 3. Locate and Inspect the Target Account Book
-   Use `filesystem-list_directory` to find files like `Account_Book.xlsx` in the workspace.
-   Use `excel-get_workbook_metadata` to confirm the sheet name (e.g., `Ledger`).
-   Use `excel-read_data_from_excel` to:
    -   Read the header row to understand the column schema (e.g., Date, Type, Product, Quantity, Unit Price, Total, Customer/Supplier, Notes).
    -   Determine the last used row (`used_ranges` from metadata or find the first empty row in column A).

### 4. Filter and Prepare Data for Append
-   For each transaction from memory, check if its `Date` falls within any of the **inclusion** date ranges.
-   **Exclude** any transaction whose `Date` falls within an **exclusion** period.
-   For each included transaction:
    -   Map fields to the target sheet's columns.
    -   Calculate `Unit Price` if needed (`Amount / Quantity`).
    -   Ensure `Customer/Supplier` field is populated correctly based on transaction type.
    -   The `Notes` column can be left empty or populated from relevant observations.

### 5. Append the Data to the Excel File
-   The `start_cell` for writing should be the first cell of the row immediately following the last used row (e.g., `A{last_row+1}`).
-   Use `excel-write_data_to_excel` with the prepared list of row data.
-   Write data in the exact column order of the target sheet.

### 6. Verification and User Communication
-   Read back the appended section using `excel-read_data_from_excel` to confirm data was written correctly.
-   Provide a concise summary to the user:
    -   Number of transactions added.
    -   List them briefly (Date, Type, Product, Amount).
    -   Explicitly state which transactions (if any) were excluded and why (e.g., "Excluded transaction from 2024-01-16 as it falls within 'this week'").

## Key Considerations
-   **Date Handling**: Be precise with date logic. Assume weeks start on Monday unless specified otherwise.
-   **Schema Matching**: The skill must adapt to the existing file's column order and naming. Do not assume a fixed schema.
-   **Data Integrity**: Never overwrite existing data. Always append.
-   **Error Handling**: If the target file doesn't exist, inform the user. If the memory contains no relevant transactions, state this clearly.
