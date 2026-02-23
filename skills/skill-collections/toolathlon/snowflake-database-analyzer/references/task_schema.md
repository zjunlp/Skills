# Task Table Schema Reference

This document details the structure of the task-related tables observed in the trajectory.

## Common Schema Across All Task Tables
All task tables share a similar structure for consistency.

| Column Name | Data Type | Description | Constraints |
|-------------|-----------|-------------|-------------|
| `EMPLOYEE_ID` | NUMBER | Foreign key to the EMPLOYEE table. | NOT NULL |
| `TASK_NAME` | TEXT | The name/description of the training task. | NOT NULL |
| `CREATE_DATE` | DATE | The date the task record was created. | NOT NULL |
| `DDL` | DATE | The deadline for task completion. | NOT NULL |
| `FINISHED_FLAG` | BOOLEAN | Indicates if the task is complete (`TRUE`) or not (`FALSE`). | NOT NULL |

## Table Specifics

### PUBLIC_TASKS
Contains mandatory training tasks for **all employees**, regardless of department.
*   **Example Tasks:** Onboarding Training, Security Training, Confidentiality Training, Company Culture, Company Strategy.
*   **Standard Deadline:** D+7 (7 calendar days after the employee's `LANDING_DATE`).

### GROUP_TASKS_BACKEND
Contains role-specific training tasks for **Backend** department employees.
*   **Example Tasks:** Backend Development Process, Backend Development Standards, Backend Development Environment.
*   **Standard Deadline:** D+30.

### GROUP_TASKS_FRONTEND
Contains role-specific training tasks for **Frontend** department employees.
*   **Example Tasks:** Frontend Development Process, Frontend Development Standards, Frontend Development Environment.
*   **Standard Deadline:** D+45.

### GROUP_TASKS_TESTING
Contains role-specific training tasks for **Testing/QA** department employees.
*   **Example Tasks:** Testing Development Process, Testing Development Standards, Testing Development Environment.
*   **Standard Deadline:** D+60.

### GROUP_TASKS_DATA
Contains role-specific training tasks for **Data** department employees.
*   **Example Tasks:** Data Development Process, Data Development Standards, Data Development Environment.
*   **Standard Deadline:** D+75.

## Key Relationships
*   `EMPLOYEE_ID` in any task table references `EMPLOYEE.EMPLOYEE_ID`.
*   An employee's group is inferred by which `GROUP_TASKS_*` table they have records in, or by their manager's ID (`EMPLOYEE.REPORT_TO_ID`).
*   The `EMPLOYEE_LANDING.LANDING_TASK_ASSIGNED` flag should be set to `TRUE` after initial tasks are inserted for a new employee.
