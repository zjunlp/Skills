# Common PDF Patterns for Invoice Extraction

## Invoice ID Patterns
- `Invoice: 2024-1771`
- `Invoice ID: BL-2024-338`
- `Document ID: 2024-4853`
- `Invoice ID: INV-2024-007`
- `Invoice: PO69638-24`

## Supplier/Vendor Patterns
- `Supplier:\n[Suzhou Manufacturing Solutions Inc.]`
- `Vendor:\n[Beijing Office Supplies Vendor]`
- `VENDOR INFORMATION:\n[Shanghai Technology Equipment Co., Ltd.]`

## Amount Patterns
- `Amount: $58678.11`
- `Total Amount: $13857.73`
- `Total: $106183.48`

## Purchaser Email Patterns
- `Contact: dcooper@mcp.com`
- `Bill To:\nPurchasing Department\nContact: ashley_anderson@mcp.com`
- `DEPARTMENT INFORMATION:\nPurchasing Department\nContact: turnerj@mcp.com`
- `Bill To:\nPurchasing Dept\nanthony_murphy24@mcp.com`

## Date Patterns
- `Date: 2024-02-25`
- `Processing Date: 2024-10-03`
- `Invoice Date: 2024-07-02`

## Payment Status Indicators

### Paid (OUTSTANDING_FLAG = 0)
- `PAYMENT STATUS: ✓ PAID ($13857.73)`
- `PAYMENT STATUS: ✓ PAID`
- `PAID ($1482003.12)`
- Checkmark symbol (✓) followed by "PAID"

### Partial (OUTSTANDING_FLAG = 1)
- `PAYMENT STATUS: ■ PARTIAL ($41064.28 of $108909.36)`
- `Remaining Balance: $67845.08`
- Square symbol (■) followed by "PARTIAL"

### Unpaid/Outstanding (OUTSTANDING_FLAG = 1)
- `PAYMENT STATUS: ✗ UNPAID`
- `Awaiting payment authorization`
- `Pending payment approval`
- `Payment verification in process`
- `Verification in process`
- Cross symbol (✗) followed by "UNPAID"
