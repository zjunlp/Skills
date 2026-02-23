# CSV Export Patterns Reference

## Common Export Scenarios

### 1. Academic Performance Analysis (from trajectory)
**Context**: Export students with significant score drops
**Columns**: student_id, name, avg_score, latest_score, drop_percentage
**Sorting**: drop_percentage DESC
**Precision**: 2 decimal places for scores/percentages
**Threshold**: Filter by drop_percentage > 25

### 2. Sales Report Export
**Columns**: product_id, product_name, category, units_sold, revenue, growth_rate
**Sorting**: revenue DESC
**Precision**: 2 decimal places for monetary values, 1 for percentages

### 3. System Log Analysis
**Columns**: timestamp, severity, component, message, user_id
**Sorting**: timestamp DESC (most recent first)
**Formatting**: ISO timestamp format, severity as uppercase

### 4. Inventory Management
**Columns**: item_id, item_name, category, current_stock, reorder_level, last_restock
**Sorting**: current_stock ASC (lowest stock first)
**Threshold**: Filter by current_stock < reorder_level

## Column Naming Conventions

### Identifier Columns
- Use consistent prefixes: `student_id`, `product_id`, `user_id`
- Keep IDs as strings even if numeric: `'S001'` not `1`

### Name/Description Columns
- Use simple names: `name`, `product_name`, `description`
- Avoid abbreviations: `customer_name` not `cust_nm`

### Metric Columns
- Be descriptive: `average_score` not `avg`
- Include units if ambiguous: `price_usd`, `weight_kg`
- For percentages: `drop_percentage`, `growth_rate_pct`

### Date/Time Columns
- Use ISO format: `2025-11-26T10:30:41`
- Be specific: `created_at`, `updated_at`, `expiration_date`

## Data Formatting Guidelines

### Numerical Values
- **Percentages**: Round to 2 decimal places (47.16%, 25.82%)
- **Currency**: Round to 2 decimal places ($123.45, €67.89)
- **Quantities**: Whole numbers for counts, decimals for measurements
- **Ratios**: 3-4 decimal places for precision (0.125, 1.3333)

### Text Values
- **Trim whitespace**: Remove leading/trailing spaces
- **Escape commas**: Use quotes if values contain commas
- **Handle line breaks**: Replace with spaces or remove

### Missing Values
- **Empty strings**: "" for missing text
- **Zero**: 0 for missing numerical values (if appropriate)
- **"N/A"**: For explicitly unavailable data

## Sorting Strategies

### Primary Sort Keys
1. **Severity metrics**: Highest values first (drop percentages, error counts)
2. **Timestamps**: Most recent first for logs, oldest first for history
3. **Alphabetical**: A-Z for names, categories
4. **Priority levels**: Critical > High > Medium > Low

### Secondary Sort Keys
- After primary metric, sort by ID or name
- For equal primary values, use natural ordering

## File Management

### Naming Conventions
- Descriptive: `student_performance_2025-11-26.csv`
- Include date: `export_YYYY-MM-DD.csv`
- Include scope: `department_sales_q4.csv`
- Version if needed: `report_v2.csv`

### Directory Structure
- `/workspace/exports/` for general exports
- `/workspace/reports/` for finalized reports
- `/workspace/dumps/` for intermediate/temporary files

## Integration Patterns

### With BigQuery
