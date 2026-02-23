---
name: snowflake-database-analyzer
description: When the user needs to explore, analyze, and manipulate Snowflake database structures and data. This skill provides 1) Listing databases, schemas, and tables within Snowflake environments, 2) Describing table structures including column names, data types, and constraints, 3) Executing read queries to extract employee data, task assignments, and status information, 4) Performing write operations to insert new records or update existing data, 5) Joining tables to establish relationships between employees, managers, and tasks, 6) Querying current dates and comparing with deadlines. Triggers include 'Snowflake database', 'query employee data', 'database schema exploration', 'data extraction from tables', 'database updates'.
---
# Instructions

## Core Workflow
This skill orchestrates a multi-step process to manage employee onboarding and task reminders via a Snowflake database. Follow these steps precisely.

### 1. Initial Exploration & Context Gathering
*   **Objective:** Understand the database landscape and locate the relevant onboarding document.
*   **Actions:**
    1.  List all available Snowflake databases. Identify the one likely containing employee data (e.g., `LANDING_TASK_REMINDER`).
    2.  List the contents of the `/workspace/dumps/workspace` directory to find the onboarding PDF (e.g., `landing_tips.pdf`).
    3.  Read the PDF to extract the standardized task list, deadlines (D+7, D+30, etc.), and group assignments (Backend, Frontend, Testing, Data).

### 2. Database Schema Investigation
*   **Objective:** Map the database structure to understand how data is organized.
*   **Actions:**
    1.  List all schemas within the target database.
    2.  List all tables within the target schema (e.g., `PUBLIC`).
    3.  Describe the key tables to understand their columns:
        *   `EMPLOYEE` (ID, Name, Email, Manager_ID)
        *   `EMPLOYEE_LANDING` (Employee_ID, Landing_Date, Task_Assigned_Flag)
        *   `PUBLIC_TASKS` (Employee_ID, Task_Name, Deadline, Status)
        *   Group-specific task tables (`GROUP_TASKS_BACKEND`, `GROUP_TASKS_FRONTEND`, etc.)

### 3. Data Analysis & Logic Building
*   **Objective:** Identify new employees needing onboarding and existing employees with overdue tasks.
*   **Actions:**
    1.  Query the `EMPLOYEE` and `EMPLOYEE_LANDING` tables. Cross-reference to find employees where `LANDING_TASK_ASSIGNED = FALSE`. These are the new hires.
    2.  For each new hire, determine their group/manager by joining the `EMPLOYEE` table with itself (manager lookup via `REPORT_TO_ID`). Infer group from manager ID based on historical data from group task tables.
    3.  Get the current date (`CURRENT_DATE()`).
    4.  For each task table (`PUBLIC_TASKS`, `GROUP_TASKS_*`), query for records where `FINISHED_FLAG = FALSE` AND `DDL < CURRENT_DATE()`. Join with employee/manager data for context.

### 4. Data Manipulation & Task Creation
*   **Objective:** Create task records for new employees and update their status flags.
*   **Actions:**
    1.  For each new employee, calculate deadlines based on their landing date and the rules from the PDF (e.g., Public Tasks: D+7, Backend Tasks: D+30).
    2.  Execute `INSERT` statements into `PUBLIC_TASKS` and the appropriate `GROUP_TASKS_*` table with the task names, calculated deadlines, and `FINISHED_FLAG = FALSE`.
    3.  Update the `EMPLOYEE_LANDING` table, setting `LANDING_TASK_ASSIGNED = TRUE` for the processed new hires.

### 5. Communication & Notification
*   **Objective:** Send formatted emails to employees and their managers.
*   **Actions:**
    1.  **For New Employees (Onboarding):** Generate an email for each new hire. Use the exact format:
        