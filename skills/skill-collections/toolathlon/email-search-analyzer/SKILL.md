---
name: email-search-analyzer
description: When the user needs to search and analyze emails for specific content patterns, particularly for tracking submissions, communications, or specific file types. This skill performs intelligent email searches with multiple keyword variations, extracts structured information from email subjects and metadata, and identifies patterns in submission data. Triggers include searching for assignment submissions, finding emails with specific patterns, extracting submission timestamps, analyzing email metadata, and identifying submission confirmation messages.
---
# Instructions

## Core Workflow
1.  **Clarify & Parse Request:** Understand the user's goal. Identify the key submission or communication pattern to search for (e.g., "final presentation", "homework 3", "report submission"). Note any specific folders, senders, or date ranges.
2.  **Perform Intelligent Email Search:**
    *   Start with the user's specified keywords.
    *   If initial search yields no results, automatically try related keyword variations (e.g., "presentation" if "final presentation" fails, or "NLP" for a course code).
    *   Use the `emails-search_emails` tool. Search in the specified folder (default: INBOX). Use a sufficient `page_size` (e.g., 50) to capture all relevant emails.
    *   Extract and log the total number of results.
3.  **Analyze Submission Patterns & Extract Data:**
    *   Examine email subjects and metadata (From, Date).
    *   Look for structured patterns in subjects (e.g., `nlp-presentation-<StudentID>-<Name>`). Extract identifiers (IDs, names) into a list.
    *   Note the submission timeframe from email dates.
4.  **Cross-Reference with External Data (if applicable):**
    *   If the user provides a reference file (e.g., student roster Excel file), load and parse it.
    *   Use tools like `excel-get_workbook_metadata` and `excel-read_data_from_excel` to read the data.
    *   Identify the relevant columns (e.g., Student ID, Name, Email, Status/Notes).
5.  **Identify Discrepancies & Target List:**
    *   Compare the list of submitters (from emails) against the master list (from reference data).
    *   Identify individuals who are on the master list but **not** in the submission list.
    *   **Apply filters:** Exclude individuals based on status notes (e.g., "withdrew", "auditing") from the target list for reminders.
    *   Compile the final list of non-submitters with their associated contact info (email, ID, name).
6.  **Execute Follow-up Actions (if requested):**
    *   If the task requires sending notifications (e.g., reminder emails), draft and send personalized messages.
    *   Use a subject and body format specified by the user. **Always include the recipient's name and ID in the body** to personalize the message and avoid spam filters.
    *   Use the `emails-send_email` tool for each recipient.
7.  **Summarize Findings:**
    *   Provide a clear summary including:
        *   Total emails found matching the pattern.
        *   Number of unique submitters identified.
        *   Number of individuals missing submissions (after applying filters).
        *   List of non-submitters (IDs, Names).
        *   Confirmation of any actions taken (e.g., "Reminder emails sent to X and Y").
    *   Mention any excluded individuals and the reason (e.g., withdrawn, auditing).

## Key Principles
*   **Keyword Flexibility:** Don't rely on a single search term. Iterate through logical variations.
*   **Pattern Recognition:** Actively look for and decode structured data within email subjects and metadata.
*   **Data Hygiene:** Always cross-check and filter lists using available status/note fields before taking action.
*   **Personalization & Anti-Spam:** Include identifiable user details (name, ID) in any communication body.
*   **Clear Reporting:** Structure the final summary to answer the user's original query explicitly.
