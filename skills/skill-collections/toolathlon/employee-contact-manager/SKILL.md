---
name: employee-contact-manager
description: When the user needs to retrieve and manage employee contact information and organizational hierarchy for notification purposes. This skill queries employee databases to extract contact details, manager relationships, and department information, enabling automated notifications to employees and their managers. Triggers include employee lookup, manager identification, contact information retrieval, organizational hierarchy queries, or when preparing notifications for expense claim reviews.
---
# Employee Contact Manager

## Purpose
Retrieve and organize employee contact information, manager relationships, and department details from enterprise databases to facilitate automated notifications and workflow approvals.

## Primary Triggers
- Need to send notifications to employees regarding expense claims, approvals, or reviews
- Requirement to identify managers for CC or escalation purposes
- Employee lookup by ID, name, or department
- Organizational hierarchy queries for reporting or workflow routing

## Core Workflow

### 1. Database Discovery
First, identify available databases containing employee information:
- Use `snowflake-list_databases()` to list all databases
- Look for databases with names suggesting employee/contact data (e.g., "HR", "EMPLOYEES", "CONTACTS")

### 2. Schema and Table Exploration
For identified databases:
- Use `snowflake-list_schemas()` to explore available schemas (typically "PUBLIC" or "HR")
- Use `snowflake-list_tables()` to find relevant tables (e.g., "EMPLOYEE_CONTACTS", "ENTERPRISE_CONTACTS", "EMPLOYEES")

### 3. Table Structure Analysis
Examine the structure of identified contact tables:
- Use `snowflake-describe_table()` to understand column structure
- Key columns to look for:
  - Employee identifiers: `EMPLOYEE_ID`, `ID`
  - Contact info: `EMAIL`, `NAME`, `PHONE`
  - Organizational data: `DEPARTMENT`, `EMPLOYEE_LEVEL`, `TITLE`
  - Manager relationships: `MANAGER_EMAIL`, `MANAGER_ID`, `REPORTS_TO`

### 4. Data Retrieval
Query the contact table to extract needed information:
- Use `snowflake-read_query()` with appropriate SELECT statements
- Common queries:
  - All employees: `SELECT * FROM <database>.<schema>.<table>`
  - Specific employee: `SELECT * FROM ... WHERE EMPLOYEE_ID = '...'`
  - Employees by department: `SELECT * FROM ... WHERE DEPARTMENT = '...'`
  - Manager lookup: `SELECT MANAGER_EMAIL FROM ... WHERE EMPLOYEE_ID = '...'`

### 5. Data Organization
Structure retrieved data for notification purposes:
- Create mappings: employee_id → {name, email, department, manager_email}
- Group by department or manager for batch notifications
- Verify email formats and completeness

### 6. Notification Preparation
Use organized contact data to:
- Address emails to employee emails
- CC appropriate manager emails
- Include relevant organizational context in message bodies

## Common Use Cases

### Expense Claim Notifications
When processing expense claims:
1. Extract employee_id from claim documents
2. Look up employee contact info and manager email
3. Send notification to employee_email
4. CC manager_email for oversight

### Manager Escalation
For issues requiring managerial attention:
1. Identify employee's department
2. Retrieve department manager or direct manager
3. Route notifications appropriately

### Department-wide Communications
For group notifications:
1. Query all employees in specific department
2. Compile email distribution lists
3. Include department head as primary contact

## Best Practices

### Data Validation
- Verify email format validity before sending
- Check for NULL/empty manager emails (escalate to department head)
- Confirm employee status (e.g., "active" vs "inactive")

### Error Handling
- If primary contact table not found, explore alternative tables
- Provide fallback contacts when manager emails unavailable
- Log missing information for manual follow-up

### Performance
- Cache frequently accessed employee data when processing multiple claims
- Use efficient queries with WHERE clauses rather than full table scans
- Consider creating views for common lookup patterns

## Integration Points
This skill typically works alongside:
- Expense claim processing skills
- Document verification workflows
- Approval routing systems
- Reporting and analytics tools

## Output Format
Organized contact data should be structured as:
