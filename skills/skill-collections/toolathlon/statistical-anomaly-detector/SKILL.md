---
name: statistical-anomaly-detector
description: When the user requests anomaly detection in numerical datasets using statistical methods such as mean and standard deviation thresholds (e.g., 'amount > mean + 3*std'). This skill calculates per-group statistics, identifies outliers, and flags abnormal records. It is triggered by keywords like 'anomaly detection', 'outlier identification', 'statistical threshold', 'abnormal transactions', or any request involving standard deviation-based filtering in financial or transactional data.
---
# Instructions

## Core Objective
Detect anomalous records in a dataset by applying a statistical threshold rule (e.g., `value > mean + N * standard_deviation`) on a per-group basis. Output the flagged anomalies, typically to a structured file like an Excel spreadsheet.

## Trigger Keywords & Phrases
- "anomaly detection"
- "outlier identification"
- "statistical threshold"
- "abnormal transactions/records"
- "flag transactions where amount > mean + 3*std"
- "find outliers using standard deviation"
- "mark abnormal data points"

## Primary Workflow
Follow this sequence when the skill is triggered:

1.  **Understand the Request & Locate Inputs**
    *   Clarify the target dataset (e.g., BigQuery table, CSV file), the numerical column for analysis (e.g., `amount`), and the grouping column (e.g., `client_id`).
    *   Identify the threshold rule (e.g., `mean + 3 * std`). The default is `mean + 3 * std` unless specified otherwise.
    *   Locate the output destination (e.g., an Excel file path and target sheet).

2.  **Inspect the Output Template**
    *   Examine the structure of the target output file (e.g., using `excel-get_workbook_metadata`).
    *   Read a sample of the existing data to understand the required column headers and format (e.g., using `excel-read_data_from_excel`).
    *   **Remove any placeholder or sample data** from the output file before writing new results.

3.  **Extract Data & Calculate Statistics**
    *   Query or load the source data, filtering for the relevant records (e.g., by year, by group).
    *   Use a single, efficient query/calculation to:
        *   Compute the mean (`AVG`) and standard deviation (`STDDEV`) for the numerical column, grouped by the specified group column.
        *   Join these statistics back to the original records.
        *   Apply the threshold rule to flag anomalies (e.g., `WHERE numerical_column > (mean + N * std)`).
        *   **Sort the final results** as requested (commonly by a record ID).

4.  **Format and Write Results**
    *   Format the anomaly records (client_id, transaction_id, timestamp, etc.) to match the output template's column order and data types.
    *   Write the formatted records to the output file, starting from the first data row after the headers.
    *   Preserve all existing formatting and sheet structure.

5.  **Verify and Summarize**
    *   Read back a portion of the updated file to confirm successful writing.
    *   Provide a concise summary to the user: state the number of anomalies found, the groups analyzed, and the output file location.

## Key Considerations
*   **Data Volume:** For large datasets, ensure queries are efficient and use appropriate limits or batch processing.
*   **Output Handling:** Always check the output file's structure before writing. Do not overwrite unintended sheets or data.
*   **Clarity:** If the user's request is ambiguous (e.g., missing group column or threshold value), ask for clarification before proceeding.
*   **Tool Selection:** Prefer using direct database/query tools (e.g., `google-cloud-bigquery_run_query`) for statistical calculations over manual Python processing for efficiency and accuracy.

## Common Pitfalls to Avoid
*   Forgetting to clear sample data from the output template.
*   Incorrectly formatting timestamps or numerical values when writing to Excel.
*   Writing results in an unsorted order when sorting is explicitly requested.
*   Making multiple unnecessary queries; combine data extraction and statistical calculation into one operation where possible.
