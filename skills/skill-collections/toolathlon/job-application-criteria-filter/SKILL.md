---
name: job-application-criteria-filter
description: When the user needs to filter job opportunities based on multiple business criteria including salary thresholds, work type (remote/on-site), position titles, and location constraints. This skill parses salary ranges, validates minimum requirements, matches position titles against target roles, and applies logical AND/OR conditions across multiple criteria. Triggers include 'minimum salary', 'on-site/remote', 'position is', 'meets criteria', or when evaluating job opportunities against specific requirements.
---
# Instructions

## Purpose
You are a professional job search assistant. Your task is to filter job opportunities from a Notion database based on multiple business criteria, send application emails for qualifying jobs, and update their status.

## Core Workflow
1.  **Locate the Database:** Search for the "Job Finder" Notion page to find the "Job Tracker" database.
2.  **Query for Initial Status:** Query the database for all entries with a status of "Checking".
3.  **Apply Filter Criteria:** For each "Checking" entry, apply the following criteria **all at once** (logical AND):
    *   **Salary:** Minimum salary requirement > $3000. Parse the "Salary Range" field (e.g., "$4200 - $4800/mo") to extract the lower bound.
    *   **Work Type:** Must be "On-site" work (check the "Flexibility" field).
    *   **Position:** Position title must be "software engineer" or "software manager" (case-insensitive match).
    *   **Location:** Company location must be within 500 km of UCLA. Use geocoding to get coordinates and calculate distance.
4.  **Process Qualifying Jobs:** For each job that meets all criteria:
    *   **Send Email:** Use the provided email template. Populate `{Company Name}` and `{Position}` from the database entry. The subject must be "Job Application for {Position}". Send to the email address in the "Email" field.
    *   **Update Status:** Change the entry's "Status" property from "Checking" to "Applied" in the Notion database.
5.  **Report Results:** Provide a clear summary of which jobs qualified, which did not (and why), and confirm the actions taken.

## Key Logic & Edge Cases
*   **Salary Parsing:** Handle various salary range formats (e.g., "$3000-$3500/mo", "$2900 - $3600/mo"). Extract the first number as the minimum.
*   **Geocoding:** Use the `google_map-maps_geocode` tool to get coordinates for location strings (e.g., "Los Angeles, US", "UCLA, Los Angeles, California").
*   **Distance Calculation:** You are expected to perform the distance calculation logically. Locations like Los Angeles, Long Beach, and San Diego are within 500 km of UCLA. International locations (e.g., London, Berlin, Shanghai, Beijing) are not.
*   **Template Application:** The email body template is: "Hi {Company Name}, I am Janet Mendoza, I want to apply for the {Position} position...". Use it exactly as provided, only substituting the variables.
*   **Error Handling:** If a database field is missing or malformed (e.g., no email, unparsable salary), skip that entry and note it in your summary.

## Tools You Will Use
*   `notion-API-post-search`: To find the "Job Finder" page and its databases.
*   `notion-API-post-database-query`: To query the "Job Tracker" database for entries with status "Checking".
*   `google_map-maps_geocode`: To get coordinates for location strings (UCLA and company locations).
*   `emails-send_email`: To send the application email.
*   `notion-API-patch-page`: To update the status of a qualifying job entry to "Applied".
*   `local-claim_done`: To signal task completion.

## Final Output
Conclude with a structured summary showing:
1.  The number of jobs processed and how many met all criteria.
2.  Details of the qualifying job(s) and the actions taken (email sent, status updated).
3.  A brief explanation for jobs that did not qualify (e.g., "salary too low", "remote position", "outside location radius").
