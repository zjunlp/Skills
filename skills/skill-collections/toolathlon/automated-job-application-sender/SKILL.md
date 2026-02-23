---
name: automated-job-application-sender
description: When the user needs to automatically send job application emails using templates with personalized placeholders and update application status in tracking systems. This skill generates personalized email content from templates with {Company Name}, {Position} placeholders, sends emails to specified contacts, and updates status fields in databases. Triggers include 'send job application email', 'apply for position', 'update status to Applied', 'email template', or when automating job application workflows.
---
# Automated Job Application Sender

## Purpose
Automatically identify qualifying job opportunities from a tracking database, send personalized application emails using templates, and update the application status.

## Core Workflow
1. **Locate Database**: Find the specified Notion database (e.g., "Job Tracker").
2. **Filter Entries**: Query for entries with status "Checking".
3. **Apply Criteria**: Filter candidates based on:
   - Minimum salary > $3000
   - Work type: On-site
   - Position: "software engineer" OR "software manager"
   - Location within 500 km of a reference point (e.g., UCLA)
4. **Send Email**: For each qualifying entry:
   - Generate email from template using `{Company Name}` and `{Position}` placeholders.
   - Send to the contact email from the database.
   - Use subject: "Job Application for {Position}"
5. **Update Status**: Change the entry's status from "Checking" to "Applied".

## Key Instructions
- **Salary Parsing**: Extract the minimum salary from "Salary Range" fields (e.g., "$4200 - $4800/mo" → 4200). Handle various formats.
- **Location Filtering**: Use geocoding to get coordinates for company locations and the reference point (UCLA). Calculate distances using the Haversine formula (see `scripts/calculate_distance.py`).
- **Template**: The default email body is: "Hi {Company Name}, I am Janet Mendoza, I want to apply for the {Position} position..."
- **Error Handling**: If geocoding fails for a location, log it and skip that entry. Ensure email sending and status updates are atomic per entry (send email, then update status).

## Required Tools
- `notion-API-post-search`: To find the database/page.
- `notion-API-post-database-query`: To fetch entries with status "Checking".
- `notion-API-patch-page`: To update the status to "Applied".
- `google_map-maps_geocode`: To get coordinates for distance calculations.
- `emails-send_email`: To send the application email.
- `local-claim_done`: To signal task completion.

## Configuration Notes
- The reference location (UCLA) and distance threshold (500 km) are currently hardcoded in the logic. For flexibility, consider making them parameters.
- The email template and sender name ("Janet Mendoza") are fixed. For customization, see `assets/email_template.txt`.
