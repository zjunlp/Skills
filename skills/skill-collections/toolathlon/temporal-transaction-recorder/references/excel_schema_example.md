# Account Book Schema Example

Based on the trajectory, the target Excel file (`Account_Book.xlsx`) has the following structure:

## Sheet Name: `Ledger`

| Column | Header | Data Type | Description |
|--------|--------|-----------|-------------|
| A | Date | Date (YYYY-MM-DD) | Transaction date |
| B | Type | String | "Purchase" or "Sales" |
| C | Product | String | Product name (e.g., "iPhone 15 Pro") |
| D | Quantity | Integer | Number of units |
| E | Unit Price | Number | Price per unit (in currency) |
| F | Total | Number | Total amount (Quantity × Unit Price) |
| G | Customer/Supplier | String | Customer name (for Sales) or Supplier name (for Purchase) |
| H | Notes | String | Optional notes (e.g., "Discount X%") |

## Data Mapping from Memory Graph

Transaction entities from the memory graph typically have `observations` containing:
- `Date YYYY-MM-DD`
- `Product <Product Name>`
- `Quantity <Number>`
- `Supplier <Name>` (for Purchase) / `Customer <Name>` (for Sales)
- `Amount <Number>` (Total amount)

**Mapping Logic:**
1.  Determine `Type` from the entity's `entityType` ("Purchase Transaction" -> "Purchase", "Sales Transaction" -> "Sales").
2.  Extract `Date`, `Product`, `Quantity` directly.
3.  Use `Amount` as the `Total` column value.
4.  Calculate `Unit Price` = `Amount` / `Quantity`. Round if necessary.
5.  Populate `Customer/Supplier` with the appropriate name.
6.  `Notes` can be left empty or populated from other observations if needed.

## Appending Data
- New rows should be appended **after the last used row**.
- The `used_ranges` metadata from `excel-get_workbook_metadata` indicates the current data boundary (e.g., `A1:H121`).
- The next available row is `last_row + 1`. Write starting at cell `A{last_row+1}`.
