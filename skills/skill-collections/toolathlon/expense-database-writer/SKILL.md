---
name: expense-database-writer
description: When the user needs to write validated expense claim data into structured databases with proper flagging for policy violations. This skill formats expense claim information into database-compatible schemas, inserts records into reimbursement tables, and sets appropriate flags (e.g., flag=1 for abnormal claims) based on validation results. Triggers include database insertion, reimbursement record creation, expense claim archiving, or when moving validated claims to permanent storage in databases like Snowflake.
---
# Instructions

## Overview
This skill processes validated expense claims and writes them into a Snowflake database table (`2024Q4REIMBURSEMENT`). It handles data formatting, flagging for policy violations, and insertion into the structured database.

## Prerequisites
1. **Database Connection**: Ensure Snowflake connection is available with access to the `TRAVEL_EXPENSE_REIMBURSEMENT.PUBLIC` schema.
2. **Validated Claims Data**: Claims must have passed initial validation (document completeness and amount matching).
3. **Policy Caps Data**: Destination-specific expense caps must be available for validation.
4. **Employee Contacts**: Employee information including manager emails must be accessible.

## Step-by-Step Process

### 1. Initialize and Verify Environment
- List available databases and verify `TRAVEL_EXPENSE_REIMBURSEMENT` exists
- Check for required tables:
  - `2024Q4REIMBURSEMENT` (target table)
  - `ENTERPRISE_CONTACTS` (for employee/manager emails)
- Verify table schemas match expected structure

### 2. Load and Validate Claims Data
- Load expense claim data from validated sources (typically extracted PDFs or JSON files)
- Each claim must include:
  - Claim ID
  - Employee ID, Name, Level, Department
  - Destination (City, Country)
  - Trip dates (Start, End) and Nights
  - Total claimed amount
  - Itemized expenses with categories (Accommodation, Meals, Transportation, Communication, Miscellaneous)

### 3. Check Policy Compliance
For each claim, compare expenses against policy caps:
- **Daily Caps**: Accommodation, Meals, Transportation (per day)
- **Trip Caps**: Communication, Miscellaneous (per trip)
- **Employee Level**: Caps vary by employee level (L1-L4)

**Flag Determination**:
- If ANY expense exceeds policy caps → Set `FLAG = 1` (abnormal)
- If ALL expenses within caps → Set `FLAG = 0` (normal)

### 4. Handle Different Claim Statuses

#### A. Claims with Document Issues (Missing receipts/Amount mismatches)
- **DO NOT** insert into database
- Send email notification to employee and CC manager
- **Email Subject**: `Expense Claim Review Required: {claim_id}`
- Include specific issues in email body

#### B. Claims with Policy Violations (Over-cap)
- Insert into database with `FLAG = 1`
- Send email notification to employee and CC manager
- **Email Subject**: `Expense Over-Cap Notice: {claim_id}`
- List specific over-cap items in email body

#### C. Fully Valid Claims (No issues)
- Insert into database with `FLAG = 0`
- No email notification required

### 5. Database Insertion
- Generate sequential ID for each claim
- Map claim data to database schema:
  - ID (sequential integer)
  - CLAIM_ID (text)
  - EMPLOYEE_ID (text)
  - EMPLOYEE_NAME (text)
  - DEPARTMENT (text)
  - DEST_CITY (text)
  - DEST_COUNTRY (text)
  - TRIP_START (date)
  - TRIP_END (date)
  - NIGHTS (number)
  - TOTAL_CLAIMED (number)
  - FLAG (number: 0 or 1)

**Insertion Pattern**:
