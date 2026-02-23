---
name: ab-test-analyzer-with-cloud-action
description: When the user requests analysis of A/B test results from BigQuery data to determine a winner and execute cloud-based actions based on the outcome. This skill handles 1) Querying and analyzing clickstream data from BigQuery datasets, 2) Calculating conversion rates across multiple scenarios, 3) Determining the winner based on arithmetic mean of per-scenario rates, 4) Filling CSV reports with results, 5) Executing conditional cloud operations - creating Cloud Storage buckets when version B wins or writing log entries when version A wins or ties. Triggers include A/B test analysis, conversion rate comparison, BigQuery clickstream data, conditional cloud operations, CSV report generation.
---
# Instructions

## 1. Understand the Task
The user will provide a request to analyze concluded A/B test data. The raw clickstream data is stored in a BigQuery dataset named `ab_testing`. You must:
- Analyze the data to determine which version ('A' or 'B') has the highest overall conversion rate.
- The **overall conversion rate** is defined as the **arithmetic mean of the per-scenario conversion rates**.
- Fill a CSV report file (`record.csv`) with the results.
- Execute a conditional cloud action based on the winner:
    - **If version B wins**: Create a new Cloud Storage bucket whose name is prefixed with `promo-assets-for-b`. Do NOT write any log entry.
    - **If version A wins or it's a tie**: Write a log entry with the message `{'status': 'AB_Test_Concluded', 'winner': 'A', 'action': 'No_Change'}` to the existing log bucket whose name is prefixed with `abtesting_logging`. Do NOT create a bucket.
- Ignore any log entries that existed in the log bucket before the task started.

## 2. Core Workflow

### Phase 1: Data Discovery & Inspection
1.  **Locate the Dataset**: Use `google-cloud-bigquery_get_dataset_info` to confirm the `ab_testing` dataset exists and note its project ID (e.g., `toolathlon-eval-ds`).
2.  **Inspect the CSV Template**: Read the provided `record.csv` file using `filesystem-read_file` to understand the required output schema (list of scenarios).
3.  **List Available Tables**: Query the dataset's `INFORMATION_SCHEMA.TABLES` to get all table names. Expect tables named like `ab_<Scenario>` (e.g., `ab_Appliances`, `ab_Automotive`).
4.  **Examine Table Structure**: Sample one table (e.g., `ab_Appliances`) to confirm it contains columns: `time_window`, `A_clicks`, `A_store_views`, `B_clicks`, `B_store_views`.

### Phase 2: Data Aggregation & Calculation
1.  **Aggregate Data Per Scenario**: For each scenario table, run a query to sum `A_clicks`, `A_store_views`, `B_clicks`, and `B_store_views`.
    - **Efficient Method**: Use the bundled script `scripts/aggregate_data.py`. It dynamically constructs and executes a single, efficient query.
    - **Manual Method**: If the script fails, construct a UNION ALL query manually for all scenarios, similar to the trajectory.
2.  **Calculate Conversion Rates**: For each scenario row:
    - `A_conversion %` = (`A_store_views` / `A_clicks`) * 100 (handle division by zero).
    - `B_conversion %` = (`B_store_views` / `B_clicks`) * 100.
    - Round results to 2 decimal places.
3.  **Calculate Overall Rates**:
    - **Per-Scenario Mean (Primary Metric)**: Calculate the arithmetic mean (average) of all `A_conversion %` values and all `B_conversion %` values separately. This determines the winner.
    - **Aggregate Rate (For CSV)**: Also calculate `total_A_store_views / total_A_clicks * 100` and `total_B_store_views / total_B_clicks * 100` for the "overall (total_store_views/total_clicks)" row in the CSV.

### Phase 3: Reporting & Conditional Action
1.  **Populate the CSV**: Write the calculated `A_conversion %` and `B_conversion %` for each scenario, plus the aggregate rate row, to `record.csv`. Ensure the scenario order matches the original template.
2.  **Determine Winner**: Compare the arithmetic means from Step 2.3.
    - `B_mean > A_mean`: **Version B wins**.
    - `A_mean > B_mean` or `A_mean == B_mean`: **Version A wins or it's a tie**.
3.  **Execute Conditional Action**:
    - **If B Wins**:
        - Generate a unique bucket name starting with `promo-assets-for-b` (e.g., `promo-assets-for-b-full-promotion`).
        - Use `google-cloud-storage_create_bucket` to create it in the `US` location.
        - **Do not** write any log entry.
    - **If A Wins or Tie**:
        - Find the existing log bucket by listing buckets and identifying one with a name prefixed `abtesting_logging`.
        - Create a log object inside it (e.g., `log_<timestamp>.txt`) containing the exact string: `{'status': 'AB_Test_Concluded', 'winner': 'A', 'action': 'No_Change'}`.
        - **Do not** create a new storage bucket.

## 3. Key Rules & Edge Cases
- **Winner Criteria**: The decisive metric is the **arithmetic mean of per-scenario conversion rates**, not the aggregate `total_views/total_clicks` rate.
- **Log Handling**: When writing a log, you must **not** read, parse, or consider any existing objects in the log bucket. Simply write the new log entry.
- **Bucket Naming**: The new bucket name must **start with** `promo-assets-for-b`. Append a descriptive suffix (e.g., `-full-promotion`) to ensure uniqueness.
- **Data Integrity**: Verify all 20 scenario tables are processed. The final CSV must have 21 rows (20 scenarios + 1 overall row).
- **Error Handling**: If BigQuery queries fail, check table names and permissions. If the log bucket cannot be found when needed, inform the user.

## 4. Final Verification
Before claiming completion:
1.  Confirm `record.csv` is correctly formatted and saved.
2.  Verify the correct cloud action was taken (bucket created OR log entry written).
3.  Provide a brief summary to the user stating the winner (A/B/Tie), the calculated mean rates, and the action taken.
