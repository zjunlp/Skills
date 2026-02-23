---
name: expense-claim-validator
description: When the user needs to validate employee expense claims against company policies and supporting documentation. This skill extracts and analyzes expense claim PDFs and corresponding invoices, verifies that all claimed items have matching receipts with correct amounts, and checks compliance with destination-specific expense caps based on employee levels. Triggers include expense claim processing, reimbursement validation, policy compliance checking, invoice verification, or when dealing with travel expense files in PDF/tar.gz formats.
---
# Instructions

## Overview
Process a batch of employee expense claims by:
1. **Extracting** claim forms and invoices from compressed archives.
2. **Validating** that every claimed item has a corresponding invoice with matching amounts.
3. **Checking** all expenses against company policy caps (destination + employee level).
4. **Taking Action** based on validation results:
   - **Document Issues** (missing receipts/amount mismatches): Send "Review Required" email to employee + CC manager.
   - **Over-Cap Issues** (exceeds policy limits): Insert claim into database with `FLAG = 1` and send "Over-Cap Notice" email to employee + CC manager.
   - **Valid Claims** (no issues): Insert into database with `FLAG = 0`.

## Prerequisites
- **Workspace**: Expense claim archives (`.tar.gz` files) in `/workspace/dumps/workspace/files/`
- **Policy Document**: `policy_en.pdf` in the same directory
- **Snowflake Database**: `TRAVEL_EXPENSE_REIMBURSEMENT` with tables:
  - `2024Q4REIMBURSEMENT` (target table)
  - `ENTERPRISE_CONTACTS` (employee/manager email mapping)

## Step-by-Step Workflow

### 1. Initial Setup & Exploration
- List workspace files to locate claim archives and policy PDF.
- Connect to Snowflake: verify database, schema, and table structures.
- Read the policy PDF to extract destination-specific caps per employee level.

### 2. Extract & Organize Claims
- Extract all `.tar.gz` archives to a temporary directory (`/workspace/dumps/workspace/extracted/`).
- Each archive contains:
  - One `*_main.pdf`: The expense claim form with header info and itemized expenses.
  - Multiple `*_invoice_*.pdf`: Supporting invoices, numbered sequentially.

### 3. Parse & Validate Each Claim
For each claim directory:
- Read the main PDF to extract:
  - Claim metadata (ID, employee, destination, dates, total)
  - Itemized expenses (date, category, amount)
- Read all corresponding invoice PDFs to extract:
  - Invoice amount and category
  - Verify receipt exists (no "No receipt available" messages)
- **Validation Checks**:
  - **Document Completeness**: Every claim item must have a corresponding invoice.
  - **Amount Matching**: Claimed amount must match invoice amount (within 0.01 CNY tolerance).
  - **Policy Compliance**: Each expense category must not exceed daily/trip caps for the employee's level at that destination.

### 4. Categorize & Act on Results
Based on validation:
- **Document Issues**: Missing receipts or amount mismatches → Send "Expense Claim Review Required: {claim_id}" email.
- **Over-Cap Issues**: All documents valid but expenses exceed caps → Insert into database with `FLAG = 1` and send "Expense Over-Cap Notice: {claim_id}" email.
- **Valid Claims**: No issues → Insert into database with `FLAG = 0`.

### 5. Database Operations
- Insert claims into `TRAVEL_EXPENSE_REIMBURSEMENT.PUBLIC."2024Q4REIMBURSEMENT"` with appropriate flag.
- Required columns: `CLAIM_ID`, `EMPLOYEE_ID`, `EMPLOYEE_NAME`, `DEPARTMENT`, `DEST_CITY`, `DEST_COUNTRY`, `TRIP_START`, `TRIP_END`, `NIGHTS`, `TOTAL_CLAIMED`, `FLAG`.

### 6. Email Notifications
- Use `emails-send_email` tool.
- **To**: Employee email (from `ENTERPRISE_CONTACTS`)
- **CC**: Manager email (from `ENTERPRISE_CONTACTS`)
- **Subject**: As specified in requirements
- **Body**: Clear description of issues found (list specific items and amounts).

## Key Decisions & Edge Cases
- **Missing Invoices**: If an invoice PDF says "No receipt available", treat as missing receipt.
- **Amount Discrepancies**: Flag any difference > 0.01 CNY between claim and invoice.
- **Policy Caps**: Use the most specific cap available (city + employee level). If city not in policy, cannot validate caps.
- **Multiple Claims per Employee**: Process each claim independently.
- **Empty Database**: Handle case where `2024Q4REIMBURSEMENT` table is empty (start ID from 1).

## Tools Required
- `filesystem-list_directory`, `filesystem-directory_tree`
- `pdf-tools-read_pdf_pages`, `pdf-tools-get_pdf_info`
- `snowflake-list_databases`, `snowflake-list_tables`, `snowflake-describe_table`, `snowflake-read_query`, `snowflake-write_query`
- `local-python-execute` (for complex parsing/analysis)
- `emails-send_email`

## Output
- Database records inserted into `2024Q4REIMBURSEMENT`
- Email notifications sent to employees and managers
- Console summary of processed claims and issues found
