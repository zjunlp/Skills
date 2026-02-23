# Expected Table Schema Example

For the skill to work correctly, all source tables in BigQuery should share a compatible schema. The following is an example based on the trajectory:

## Table: `scores_2501`, `scores_2502`, ... `scores_2507`
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `student_id` | STRING | Unique identifier for the student (e.g., 'S001'). |
| `name` | STRING | Full name of the student. |
| `score` | FLOAT | Numerical score for the test/quiz. |

## Key Points
*   **Consistency:** All tables must have the same column names and compatible data types.
*   **Historical Context:** Each table typically represents data from a different time period (e.g., test 1, test 2).
*   **Aggregation Key:** The `student_id` (and often `name`) is used to join and aggregate data across tables.
*   **Metric Column:** The `score` column is the primary numerical metric for aggregation (average, sum, etc.).

## Discovery Query
Use this query to find all relevant tables in a dataset:
