# BigQuery Schema & Requirements Reference

## Target Table Structure
- **Full Table Path**: `bigquery_pricing_analysis.analysis`
- **Required Columns & Data Types**:
    1.  `Product Name` (STRING) - The exact product name from the internal CSV file.
    2.  `Our Price` (FLOAT) - Our product's price.
    3.  `Competitor Price` (FLOAT) - The matched competitor's price.
    4.  `Price Difference` (FLOAT) - Calculated as `Our Price - Competitor Price`.

## Data Processing Notes
- **Product Matching**: Use the `Product Name` from the internal file as the primary key. Match competitor products based on name similarity, keywords, or product categories.
- **Calculation**: `Price Difference` must be calculated during processing and stored as a column. A positive value means our product is more expensive; a negative value means it's cheaper.
- **Loading Mode**: The trajectory uses `WRITE_TRUNCATE` mode, which replaces all existing data in the table. Ensure this is the intended behavior.

## Example Valid Data
| Product Name        | Our Price | Competitor Price | Price Difference |
|---------------------|-----------|------------------|------------------|
| SmartWidget Pro     | 399.99    | 349.95           | 50.04            |
| CloudSync Enterprise| 89.99     | 129.99           | -40.00           |

## Common Agent Functions for BigQuery
- `google-cloud-bigquery_list_datasets()`: Check if the target dataset exists.
- `google-cloud-bigquery_load_csv_data(dataset_id, table_id, csv_file_path, skip_header=True, write_mode="WRITE_TRUNCATE")`: Loads the generated CSV file.
- `google-cloud-bigquery_run_query(query)`: Run a validation query (e.g., `SELECT * FROM dataset.table ORDER BY Product Name`).
