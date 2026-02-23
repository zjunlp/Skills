---
name: kubernetes-database-analyst
description: Analyzes data from databases in Kubernetes clusters by setting up port forwarding, executing SQL queries, and generating CSV reports.
---
# Instructions

## Phase 1: Initial Setup & Discovery
1.  **Read the Task:** Use `filesystem-read_file` to read the task file (e.g., `task.md`) to understand the analytical questions and required output format.
2.  **Inspect Kubernetes Resources:** Use `k8s-kubectl_get` to list all resources in the target namespace (e.g., `data`). Identify the database `Service` (e.g., `mysql-f1`) and its corresponding `Pod`.
3.  **Establish Database Connection:** Use `k8s-port_forward` to create a persistent port forward from a local port (e.g., `30124`) to the database service's target port (e.g., `3306`). **Note the `portForwardId`** to keep it alive.
4.  **List Workspace Files:** Use `filesystem-list_directory` to identify any template or input files in the workspace.

## Phase 2: Database Schema Exploration
5.  **Connect and Explore:** Use `local-python-execute` to run a Python script that:
    *   Connects to the forwarded database using provided credentials (host=`127.0.0.1`, port=`[LOCAL_PORT]`).
    *   Lists all tables to understand the data structure.
    *   Examines the schema of key tables relevant to the task (e.g., `drivers`, `constructors`, `results`, `races`, `standings` tables).

## Phase 3: Query Execution & Analysis
6.  **Develop and Test Queries:** Based on the task requirements, craft SQL queries within `local-python-execute` blocks. Test them iteratively to ensure correctness.
    *   For ranking/aggregation tasks (e.g., "highest points per year"), join `races`, `standings`, and dimension tables (`drivers`, `constructors`). Use window functions or subqueries with `GROUP BY` and `MAX()`.
    *   For complex conditional logic (e.g., "same constructor in first and last race"), use Common Table Expressions (CTEs) to break down the problem: filter by date/year, calculate first/last race per driver-season, count distinct rounds, and join on constructor ID.
7.  **Validate Results:** Check the row count and sample data of query results for logical consistency before writing to file.

## Phase 4: Report Generation & Cleanup
8.  **Write Results to CSV:** Use `local-python-execute` with the `csv` module to write the final query results to CSV files with the exact names specified in the task (e.g., `results_1.csv`, `results_2.csv`). Ensure headers match the template format.
9.  **Remove Template Files:** Use `local-python-execute` with `os.remove()` to delete any provided template files from the workspace as instructed.
10. **Verify Output:** Use `filesystem-read_multiple_files` to confirm the final CSV files exist and contain the expected data.
11. **Finalize:** The port forward started in Step 3 should remain active. Use `local-claim_done` to signal task completion.
