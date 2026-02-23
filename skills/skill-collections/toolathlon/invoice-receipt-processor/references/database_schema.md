# PURCHASE_INVOICE Database Schema

## Table: INVOICES
Stores basic invoice information extracted from receipts.

| Column Name | Data Type | Nullable | Description |
|-------------|-----------|----------|-------------|
| INVOICE_ID | TEXT | NO | Unique invoice identifier (from PDF) |
| SUPPLIER_NAME | TEXT | NO | Name of the supplier/vendor |
| INVOICE_AMOUNT | NUMBER | NO | Total invoice amount in dollars |
| PURCHASER_EMAIL | TEXT | NO | Email of the purchasing manager |
| INVOICE_DATE | DATE | YES | Date of the invoice |

## Table: INVOICE_PAYMENTS
Tracks payment status and amounts.

| Column Name | Data Type | Nullable | Default | Description |
|-------------|-----------|----------|---------|-------------|
| INVOICE_ID | TEXT | NO | - | Foreign key to INVOICES table |
| PAYMENT_AMOUNT | NUMBER | YES | 0 | Amount paid to date |
| OUTSTANDING_FLAG | NUMBER | YES | 1 | **0=Paid, 1=Outstanding** |

## Key Relationships
- `INVOICE_PAYMENTS.INVOICE_ID` → `INVOICES.INVOICE_ID`
- Each invoice should have exactly one payment record
