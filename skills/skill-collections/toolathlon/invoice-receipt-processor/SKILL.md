---
name: invoice-receipt-processor
description: When the user needs to process invoice receipts from PDF files in a workspace and update a purchase invoice database with payment tracking. This skill automatically scans workspace directories for PDF receipts, extracts invoice data (invoice ID, supplier, amount, purchaser email, date, payment status), inserts records into database tables, sets up payment tracking with outstanding flags, identifies unpaid or partially paid invoices, and sends email notifications to relevant purchasing managers with specific filenames requiring attention. Key triggers include processing receipts, updating invoice databases, tracking outstanding payments, sending payment reminder emails, and handling PDF invoice files with payment status indicators.
---
# Instructions

## 1. Initialize and Explore
- **Explore Workspace:** List the contents of the `/workspace/dumps/workspace` directory to locate the `files` folder containing PDF receipts.
- **Explore Database:** List schemas in the `PURCHASE_INVOICE` database, then list tables in the `PUBLIC` schema.
- **Examine Tables:** Describe the structure of the `INVOICES` and `INVOICE_PAYMENTS` tables. Read sample data to understand existing records and column formats.

## 2. Extract Data from PDF Receipts
- **Scan Files:** List all PDF files in the workspace's `files` directory.
- **Read PDFs:** For each PDF file, extract text from the first page. Parse the following key fields:
  - **Invoice ID:** Look for patterns like "Invoice:", "Invoice ID:", "Document ID:", or text near the filename.
  - **Supplier Name:** Typically under "Supplier:", "Vendor:", or "VENDOR INFORMATION:".
  - **Invoice Amount:** Look for "Amount:", "Total Amount:", or "Total:" followed by a dollar amount.
  - **Purchaser Email:** Look for "Contact:", "Bill To:", or "DEPARTMENT INFORMATION:" followed by an email address.
  - **Invoice Date:** Look for "Date:", "Processing Date:", or "Invoice Date:".
  - **Payment Status:** Identify status keywords:
    - `PAID` or checkmark (✓): Fully paid.
    - `PARTIAL` or square (■): Partially paid. Extract the paid amount if available.
    - `UNPAID`, `Awaiting payment`, `Pending`, `Verification in process`: Outstanding (unpaid).
    - Default to outstanding if status is unclear.

## 3. Update Database Tables
- **Insert Invoices:** For each extracted receipt, insert a record into `PURCHASE_INVOICE.PUBLIC.INVOICES` with columns: `INVOICE_ID`, `SUPPLIER_NAME`, `INVOICE_AMOUNT`, `PURCHASER_EMAIL`, `INVOICE_DATE`.
- **Insert Payments:** For each invoice, insert a corresponding record into `PURCHASE_INVOICE.PUBLIC.INVOICE_PAYMENTS`:
  - `INVOICE_ID`: The extracted invoice ID.
  - `PAYMENT_AMOUNT`:
    - If status is `PAID`: Set to the full invoice amount.
    - If status is `PARTIAL`: Set to the extracted paid amount.
    - Otherwise: Set to `0`.
  - `OUTSTANDING_FLAG`:
    - `0` if status is `PAID`.
    - `1` for all other statuses (`PARTIAL`, `UNPAID`, `Awaiting`, `Pending`, `Verification`).
- **Set Column Description:** Execute an `ALTER TABLE` statement to set the comment/description for the `OUTSTANDING_FLAG` column in `INVOICE_PAYMENTS` to exactly: `0=Paid, 1=Outstanding`. Verify the description was applied.

## 4. Identify and Notify on Outstanding Invoices
- **Query Outstanding Invoices:** Perform a `JOIN` between `INVOICES` and `INVOICE_PAYMENTS` where `OUTSTANDING_FLAG = 1`. Group results by `PURCHASER_EMAIL`.
- **Map to Filenames:** For each outstanding invoice, map the `INVOICE_ID` back to the original PDF filename (e.g., `INV-2024-013` → `INV-2024-013.pdf`).
- **Send Emails:** For each unique purchasing manager email (`PURCHASER_EMAIL`):
  - **Subject:** `Process Outstanding Invoices`
  - **Body:** List the filenames of all outstanding invoices for that manager. Use a clear, professional format.
  - Send the email.

## 5. Final Verification and Summary
- Provide a concise summary report including:
  - Count of receipts processed and inserted.
  - Count of outstanding vs. paid invoices.
  - List of managers notified and the number of files each needs to process.
  - Confirmation that the column description was set.
