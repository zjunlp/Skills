---
name: compliance-notification-dispatcher
description: When the user needs to send automated email notifications for expense claim issues to employees and their managers. This skill generates context-specific email templates for different violation types (missing documentation vs. policy overages), includes detailed issue descriptions, and sends notifications with proper CC to managers. Triggers include compliance notification sending, policy violation alerts, expense claim review requests, or when automated communication is needed for documentation or policy issues.
---
# Instructions

## Overview
This skill automates the notification process for expense claim compliance issues. It handles two main violation types:
1. **Documentation Issues**: Missing receipts or amount mismatches between claims and invoices.
2. **Policy Over-Cap Issues**: Expenses exceeding company policy limits for specific destinations and employee levels.

## Prerequisites
Before running this skill, ensure you have:
1. **Expense Claim Data**: Extracted and analyzed expense claims with identified issues
2. **Employee Contact Information**: Database or file containing employee emails and manager relationships
3. **Policy Caps**: Company travel expense policy caps by destination city and employee level
4. **Database Access**: Connection to the reimbursement database (Snowflake in the example)

## Workflow

### 1. Analyze Claims and Identify Issues
First, analyze all expense claims to identify:
- **Documentation Issues**: Check if every claimed item has a corresponding invoice and if amounts match
- **Policy Over-Cap Issues**: Compare expenses against policy caps for the destination and employee level

Use the analysis script (`scripts/analyze_claims.py`) to automate this process.

### 2. Categorize Claims
Categorize claims into three groups:
- **Document Issues**: Claims with missing receipts or amount mismatches
- **Over-Cap Issues**: Claims with complete documentation but exceeding policy caps
- **Valid Claims**: Claims meeting all requirements

### 3. Send Notifications
For each category, send appropriate email notifications:

#### A. Document Issues (Review Required)
- **Subject**: `Expense Claim Review Required: {claim_id}`
- **To**: Employee email
- **CC**: Manager email
- **Template**: Use `references/email_templates.md#document-issue-template`
- **Content**: List specific missing documents or amount discrepancies

#### B. Over-Cap Issues (Policy Violation)
- **Subject**: `Expense Over-Cap Notice: {claim_id}`
- **To**: Employee email
- **CC**: Manager email
- **Template**: Use `references/email_templates.md#over-cap-template`
- **Content**: List specific policy violations with amounts and caps

### 4. Database Operations
For over-cap claims that have complete documentation:
- Insert into the reimbursement database
- Set `FLAG = 1` to indicate abnormal/over-cap claims
- Do NOT insert claims with document issues

### 5. Verification
After processing:
- Verify emails were sent successfully
- Confirm database inserts are correct
- Provide a summary report of actions taken

## Key Decisions
- **Document Issues Take Precedence**: If a claim has both document issues and over-cap issues, treat it as a document issue and do NOT insert into database
- **Manager CC**: Always include the manager's email in CC for accountability
- **Detailed Descriptions**: Include specific amounts, dates, and policy caps in notifications
- **Batch Processing**: Process claims in logical batches to avoid overwhelming email systems

## Error Handling
- Log all email sending failures for retry
- Validate database connections before inserts
- Handle missing employee contact information gracefully
- Verify policy caps exist for all destinations mentioned in claims
