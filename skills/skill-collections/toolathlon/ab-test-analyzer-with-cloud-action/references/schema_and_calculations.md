# Data Schema & Calculation Reference

## BigQuery Table Schema
Each scenario table (e.g., `ab_Appliances`) in the `ab_testing` dataset has the following columns:
- `time_window`: STRING - Time period for the metrics (e.g., "7/29 08:00-08:59")
- `A_clicks`: INTEGER - Number of clicks on version A of the homepage.
- `A_store_views`: INTEGER - Number of subsequent store views from version A clicks.
- `B_clicks`: INTEGER - Number of clicks on version B of the homepage.
- `B_store_views`: INTEGER - Number of subsequent store views from version B clicks.

## Core Metrics & Formulas

### 1. Per-Scenario Conversion Rate
For a given scenario (e.g., Appliances):
- **Version A Conversion %** = `(SUM(A_store_views) / SUM(A_clicks)) * 100`
- **Version B Conversion %** = `(SUM(B_store_views) / SUM(B_clicks)) * 100`
- **Handling Zero Clicks**: If `SUM(A_clicks)` or `SUM(B_clicks)` is 0, the conversion rate for that version is 0%.

### 2. Overall Conversion Rate (Decision Metric)
The **winner** is determined by comparing the **arithmetic mean (average)** of all per-scenario conversion rates.
- Let `A_rates` = [A_conv_scenario1, A_conv_scenario2, ..., A_conv_scenario20]
- Let `B_rates` = [B_conv_scenario1, B_conv_scenario2, ..., B_conv_scenario20]
- `A_overall_mean` = `sum(A_rates) / 20`
- `B_overall_mean` = `sum(B_rates) / 20`
- **Winner**: The version with the higher `overall_mean`. If equal, it's a tie.

### 3. Aggregate Conversion Rate (For Reporting)
This is calculated for the final row in the CSV report:
- `A_aggregate_rate` = `(SUM(all_A_store_views) / SUM(all_A_clicks)) * 100`
- `B_aggregate_rate` = `(SUM(all_B_store_views) / SUM(all_B_clicks)) * 100`
- **Note**: This is **not** used to determine the winner, only for reporting.

## CSV Report Schema (`record.csv`)
The output CSV must have the following columns:
- `scenario`: The name of the product category/scenario.
- `A_conversion %`: The calculated conversion percentage for Version A (rounded to 2 decimals).
- `B_conversion %`: The calculated conversion percentage for Version B (rounded to 2 decimals).

**Row Order**:
1.  20 rows, one for each scenario, sorted alphabetically by scenario name.
2.  A final 21st row with:
    - `scenario`: "overall (total_store_views/total_clicks)"
    - `A_conversion %`: The `A_aggregate_rate`
    - `B_conversion %`: The `B_aggregate_rate`

## Conditional Action Logic
| Condition | Action | Details |
| :--- | :--- | :--- |
| `B_overall_mean` > `A_overall_mean` | **Create Cloud Storage Bucket** | - Name must start with `promo-assets-for-b`.<br>- Choose a location (e.g., `US`).<br>- **Do not** write any log. |
| `A_overall_mean` >= `B_overall_mean` | **Write Log Entry** | - Find bucket with name prefix `abtesting_logging`.<br>- Create a new object with content: `{'status': 'AB_Test_Concluded', 'winner': 'A', 'action': 'No_Change'}`.<br>- **Do not** create a new bucket. |
