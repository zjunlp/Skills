# Google Sheets Purchase Requisition Schema

## Required Columns
The Google Sheet should have the following column structure:

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| A | Product ID | String/Number | "9" |
| B | Product Name | String | "Laptop Dell XPS 13" |
| C | SKU | String | "DELL-XPS-13" |
| D | Current Stock | Number | 5 |
| E | Safety Threshold | String/Number | "10" |
| F | Supplier Name | String | "Dell China" |
| G | Supplier ID | String | "SUP001" |
| H | Supplier Contact | String | "supplier@dell.cn" |
| I | Alert Time | Date/Time | "2025-11-26" |
| J | Suggested Order Quantity | String/Number | "15" |

## Notes
- Row 1 should contain header labels
- Date format: YYYY-MM-DD
- Suggested Order Quantity calculation: `(threshold - current_stock) × 1.5` (rounded up)
- The sheet name should match the spreadsheet title (e.g., "stock_sheet")
