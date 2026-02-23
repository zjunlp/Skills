# Database Schema Reference

## Database: LANDING_TASK_REMINDER
## Schema: PUBLIC

### Table: EMPLOYEE
Stores employee information.
| Column Name | Data Type | Nullable | Description |
|-------------|-----------|----------|-------------|
| NAME | TEXT | NO | Employee full name |
| EMAIL | TEXT | NO | Employee email address |
| EMPLOYEE_ID | NUMBER | NO | Unique employee identifier |
| REPORT_TO_ID | NUMBER | YES | Manager's EMPLOYEE_ID (foreign key to EMPLOYEE) |

### Table: EMPLOYEE_LANDING
Tracks employee onboarding status.
| Column Name | Data Type | Nullable | Description |
|-------------|-----------|----------|-------------|
| LANDING_DATE | DATE | NO | Employee start/joining date |
| EMPLOYEE_ID | NUMBER | NO | Foreign key to EMPLOYEE |
| LANDING_TASK_ASSIGNED | BOOLEAN | NO | Flag indicating if onboarding tasks have been assigned |

### Table: PUBLIC_TASKS
Stores public training tasks for all employees.
| Column Name | Data Type | Nullable | Description |
|-------------|-----------|----------|-------------|
| FINISHED_FLAG | BOOLEAN | NO | Task completion status (TRUE/FALSE) |
| TASK_NAME | TEXT | NO | Name of the task |
| CREATE_DATE | DATE | NO | Date task was assigned |
| EMPLOYEE_ID | NUMBER | NO | Foreign key to EMPLOYEE |
| DDL | DATE | NO | Task deadline date |

### Group Task Tables
Four tables with identical schema for different employee groups:
- `GROUP_TASKS_BACKEND` (Backend group, D+30 deadlines)
- `GROUP_TASKS_FRONTEND` (Frontend group, D+45 deadlines)
- `GROUP_TASKS_TESTING` (Testing group, D+60 deadlines)
- `GROUP_TASKS_DATA` (Data group, D+75 deadlines)

#### Schema for Group Task Tables:
| Column Name | Data Type | Nullable | Description |
|-------------|-----------|----------|-------------|
| FINISHED_FLAG | BOOLEAN | NO | Task completion status |
| TASK_NAME | TEXT | NO | Name of the task |
| CREATE_DATE | DATE | NO | Date task was assigned |
| EMPLOYEE_ID | NUMBER | NO | Foreign key to EMPLOYEE |
| DDL | DATE | NO | Task deadline date |

## Key Relationships
- `EMPLOYEE.EMPLOYEE_ID` → `EMPLOYEE_LANDING.EMPLOYEE_ID`
- `EMPLOYEE.EMPLOYEE_ID` → `PUBLIC_TASKS.EMPLOYEE_ID`
- `EMPLOYEE.EMPLOYEE_ID` → `GROUP_TASKS_*.EMPLOYEE_ID`
- `EMPLOYEE.REPORT_TO_ID` → `EMPLOYEE.EMPLOYEE_ID` (self-join for manager)

## Group Assignment Logic
Employee groups are determined by their manager (REPORT_TO_ID):
- REPORT_TO_ID = 8001 → Backend Group (uses GROUP_TASKS_BACKEND)
- REPORT_TO_ID = 8002 → Frontend Group (uses GROUP_TASKS_FRONTEND)
- REPORT_TO_ID = 8003 → Testing Group (uses GROUP_TASKS_TESTING)
- REPORT_TO_ID = 8004 → Data Group (uses GROUP_TASKS_DATA)
