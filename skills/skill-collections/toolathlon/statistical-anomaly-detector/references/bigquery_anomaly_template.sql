/*
BigQuery SQL Template for Anomaly Detection
Purpose: A reusable, parameterized SQL query to efficiently extract anomalies in one pass.
Modify the bracketed parameters [ ] based on the specific task.
*/

WITH group_stats AS (
  SELECT
    [GROUP_COLUMN], -- e.g., client_id, department_id
    AVG([VALUE_COLUMN]) AS mean_value,
    STDDEV([VALUE_COLUMN]) AS std_value
  FROM
    `[PROJECT.DATASET.TABLE_NAME]`
  WHERE
    [OPTIONAL_FILTER_CONDITIONS] -- e.g., EXTRACT(YEAR FROM txn_time) = 2025
  GROUP BY
    [GROUP_COLUMN]
),
all_records AS (
  SELECT
    [RECORD_ID_COLUMN], -- e.g., transaction_id
    [GROUP_COLUMN],
    [VALUE_COLUMN],
    [ADDITIONAL_COLUMNS], -- e.g., txn_time, currency, status
    [TIMESTAMP_COLUMN] -- for formatting if needed
  FROM
    `[PROJECT.DATASET.TABLE_NAME]`
  WHERE
    [OPTIONAL_FILTER_CONDITIONS] -- Must match the filter in group_stats
)
SELECT
  a.[GROUP_COLUMN],
  a.[RECORD_ID_COLUMN],
  a.[VALUE_COLUMN],
  -- Include additional columns for the report
  a.[ADDITIONAL_COLUMNS],
  -- Format timestamp if necessary for Excel compatibility
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S.%f UTC', a.[TIMESTAMP_COLUMN]) AS formatted_timestamp,
  -- Statistics for verification (can be omitted in final report)
  s.mean_value,
  s.std_value,
  (s.mean_value + [N] * s.std_value) AS threshold, -- [N] is the sigma multiplier (e.g., 3)
  'ANOMALY' AS status
FROM
  all_records a
JOIN
  group_stats s
ON
  a.[GROUP_COLUMN] = s.[GROUP_COLUMN]
WHERE
  -- Apply the anomaly rule: value > mean + N*std
  a.[VALUE_COLUMN] > (s.mean_value + [N] * s.std_value)
  -- For lower-bound anomalies: a.[VALUE_COLUMN] < (s.mean_value - [N] * s.std_value)
ORDER BY
  a.[RECORD_ID_COLUMN] -- Or any other specified sort order
-- LIMIT [MAX_RESULTS] -- Use if needed for very large anomaly sets
