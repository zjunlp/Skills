---
name: notion-job-tracker-query
description: When the user needs to query and retrieve job application data from a Notion database with specific filtering criteria. This skill searches for Notion pages/databases by name, queries job tracker databases with status filters, and extracts structured job application information including company details, positions, salaries, locations, and contact information. Triggers include 'Notion database', 'job tracker', 'query jobs', 'filter by status', or when working with structured job application data.
---
# Instructions

## Primary Objective
Query a Notion database named "Job Tracker" to find job application entries that match specific user-defined criteria, extract structured information, and optionally perform follow-up actions (like sending emails or updating statuses).

## Core Workflow

### 1. Locate the Target Database
- Use the `notion-API-post-search` tool to search for pages/databases containing "Job Finder" or "Job Tracker".
- **Important**: Do not filter by object type (`"filter": {"property":"object","value":"page"}`) in the initial search, as this may exclude databases. Search broadly first.
- Identify the database with title "Job Tracker". Note its `database_id`.

### 2. Query the Database with Filters
- Use the `notion-API-post-database-query` tool with the identified `database_id`.
- Apply filters based on user requirements. Common filters include:
    - `Status` (e.g., `{"property":"Status","select":{"equals":"Checking"}}`)
    - `Flexibility` (e.g., `{"property":"Flexibility","select":{"equals":"On-site"}}`)
    - `Position` (text/URL field containing "Software Engineer" or "Software Manager" - filtering logic may need to be applied post-query).
- Retrieve all pages (results) from the query.

### 3. Extract and Process Structured Data
For each page (job entry) in the results, extract the following properties. Refer to the example trajectory for exact property names and types:
- `Company` (title property)
- `Position` (url property, treat as text)
- `Status` (select property)
- `Salary Range` (rich_text property) - **Parse the minimum salary value** (e.g., extract the first number from "$4200 - $4800/mo").
- `Location` (rich_text property)
- `Email` (rich_text property)
- `In-touch Person` (rich_text property)
- `Flexibility` (select property)

**Data Parsing Script**: For consistent and error-proof parsing of complex fields like `Salary Range`, use the bundled script `scripts/parse_job_data.py`.

### 4. Apply Advanced Criteria (Geographic, Salary)
If criteria involve geographic distance:
1.  Obtain reference coordinates (e.g., for "UCLA") using `google_map-maps_geocode`.
2.  For each job entry's `Location`, geocode it using the same tool.
3.  Calculate the distance. Use the bundled script `scripts/calculate_distance.py` for accurate haversine distance calculation.
4.  Filter entries where distance <= specified radius (e.g., 500 km).

If criteria involve numeric thresholds (e.g., minimum salary > $3000):
- Use the parsed minimum salary from Step 3 for comparison.

### 5. Execute Follow-up Actions (If Required)
Based on the user's goal (e.g., "send application emails"):
1.  For each qualifying entry, construct a personalized email using a template.
    - Template: `"Hi {Company Name}, I am Janet Mendoza, I want to apply for the {Position} position..."`
    - Subject: `"Job Application for {Position}"`
2.  Send the email using `emails-send_email`.
3.  Update the entry's `Status` in Notion to "Applied" using `notion-API-patch-page`.

## Key Considerations
- **Schema Variability**: Notion database property names and types may differ. Use the initial query results to inspect the schema before writing extraction logic.
- **Error Handling**: When parsing salaries or geocoding, some entries may have malformed data. Log warnings and skip those entries.
- **Batching**: For large result sets, consider processing in batches to avoid timeouts.
- **Confirmation**: Before sending emails or updating many records, summarize the qualifying entries for user confirmation.
