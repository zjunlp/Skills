---
name: employee-onboarding-automator
description: When the user needs to automate new employee onboarding processes including training task assignment, email notifications, and database updates. This skill handles 1) Querying employee databases (Snowflake) to identify new hires and their group assignments, 2) Analyzing onboarding documents (PDFs) to extract training requirements and deadlines, 3) Generating personalized task lists based on employee groups (Backend/Frontend/Testing/Data), 4) Inserting task records into appropriate database tables with calculated deadlines, 5) Sending onboarding emails to new employees with their task lists, 6) Checking for overdue incomplete tasks and sending reminder emails, 7) Updating tracking flags in the database. Triggers include 'new employees joining', 'onboarding automation', 'training task assignment', 'employee onboarding emails', 'task deadline reminders'.
---
# Instructions

## 1. Initial Setup & Discovery
- **Discover Environment**: Use `filesystem-list_directory` to explore `/workspace/dumps/workspace` for onboarding documents (PDFs).
- **Database Discovery**: Use `snowflake-list_databases` and `snowflake-list_schemas` to locate the relevant database (e.g., `LANDING_TASK_REMINDER`).
- **Table Discovery**: Use `snowflake-list_tables` to find employee and task tables (e.g., `EMPLOYEE`, `EMPLOYEE_LANDING`, `PUBLIC_TASKS`, `GROUP_TASKS_*`).

## 2. Analyze Onboarding Document
- **Read PDF**: Use `pdf-tools-get_pdf_info` and `pdf-tools-read_pdf_pages` to extract the onboarding checklist from the PDF (e.g., `landing_tips.pdf`).
- **Extract Task Structure**: Parse the PDF to identify:
  - **Public Tasks** (for all departments): e.g., "Onboarding Training", "Security Training", "Confidentiality Training", "Company Culture", "Company Strategy". Note their relative deadlines (e.g., "complete by D+7").
  - **Group-Specific Tasks**: Identify tasks for Backend (D+30), Frontend (D+45), Testing (D+60), and Data (D+75) groups. Each group typically has "Development Process", "Development Standards", and "Development Environment" tasks.

## 3. Identify New Employees & Group Assignments
- **Query Employee Data**: Use `snowflake-read_query` on `EMPLOYEE` and `EMPLOYEE_LANDING` tables.
- **Find New Hires**: Identify employees where `LANDING_TASK_ASSIGNED = FALSE` and `LANDING_DATE` is recent (or as per user request).
- **Determine Group Assignment**: An employee's group is inferred from their `REPORT_TO_ID` (manager). Map managers to groups based on existing task table patterns (e.g., employees with tasks in `GROUP_TASKS_BACKEND` report to manager ID 8001).
- **Get Manager Emails**: Join with the `EMPLOYEE` table to get the manager's email for CC.

## 4. Check for Overdue Tasks
- **Query All Task Tables**: For `PUBLIC_TASKS` and each `GROUP_TASKS_*` table, find records where `FINISHED_FLAG = FALSE` AND `DDL < CURRENT_DATE()`.
- **Aggregate by Employee**: Collect all overdue tasks for each employee, including their manager's email.

## 5. Calculate Deadlines & Generate Task Lists
- **Base Date**: Use `CURRENT_DATE()` as the start date (Day 0).
- **Calculate Deadlines**:
  - Public Tasks: Day 0 + 7 days.
  - Backend Tasks: Day 0 + 30 days.
  - Frontend Tasks: Day 0 + 45 days.
  - Testing Tasks: Day 0 + 60 days.
  - Data Tasks: Day 0 + 75 days.
- **Format Dates**: Use 'YYYY-MM-DD' format for database insertion and email content.

## 6. Update Database
- **Insert New Tasks**: Use `snowflake-write_query` to insert records into the appropriate tables (`PUBLIC_TASKS`, `GROUP_TASKS_*`) for each new employee.
- **Update Tracking Flag**: Update `EMPLOYEE_LANDING` table, setting `LANDING_TASK_ASSIGNED = TRUE` for processed new hires.

## 7. Send Emails
- **Onboarding Emails**: For each new employee, use `emails-send_email`.
  - **To**: Employee's email.
  - **CC**: Their manager's email (from `REPORT_TO_ID` lookup).
  - **Subject**: "Onboarding Training Tasks".
  - **Body**: Follow the exact format:
    