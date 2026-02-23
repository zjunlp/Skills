---
name: academic-candidate-filter
description: Filters candidates based on specific academic publication criteria (e.g., first-author, sole-author, or independent publications) from resumes or application materials. Searches through candidate data, extracts publication history, and applies filtering logic to identify qualified individuals.
---
# Instructions

## 1. Objective
Filter candidates from provided materials (e.g., emails, documents) based on user-specified authorship criteria (e.g., "first-author", "sole-author", "independent publications"). For qualified candidates, schedule interviews according to user constraints and sync to Google Calendar.

## 2. Core Workflow

### Phase 1: Data Collection & Parsing
1.  **Identify Source:** Determine where candidate materials are stored based on user request (e.g., "emails", "documents in a folder").
2.  **Search & Retrieve:** Use appropriate tools (e.g., `emails-search_emails`, `files-search_files`) to find candidate materials using relevant keywords ("resume", "CV", "application").
3.  **Extract Content:** Read the full content of each retrieved item.
4.  **Parse Publications:** For each candidate, scan the content to identify a "Publications", "Research Results", or similar section. Extract the list of publications, paying close attention to authorship indicators (e.g., "**First Author**", "*Sole Author*", "First Author:", "Author:").

### Phase 2: Candidate Filtering
1.  **Apply Filter Logic:** Evaluate each candidate's publication list against the user's stated criteria.
    -   **Default Logic (from trajectory):** A candidate qualifies if they have **at least one publication explicitly marked as "First Author" or "Sole Author"**. Co-authored publications (Second, Third, etc.) do **not** count.
    -   **Adaptation:** If the user specifies a different criterion (e.g., "at least two first-author papers"), adjust the logic accordingly.
2.  **Compile Results:** Create a clear list of qualified and non-qualified candidates, citing the specific publications that caused their inclusion or exclusion.

### Phase 3: Interview Scheduling
1.  **Interpret Time Constraints:** Parse the user's scheduling instructions (dates, working hours, interview duration).
    -   *Example Logic:* If "tomorrow and the day after tomorrow" is specified, calculate these dates relative to the **most recent date found in the source materials** (as seen in the trajectory).
2.  **Generate Time Slots:** Within the specified date range and working hours, create candidate interview slots that respect the minimum duration.
    -   Consider buffer time between interviews if scheduling multiple candidates.
3.  **Create Calendar Events:** For each qualified candidate, create a Google Calendar event (`google_calendar-create_event`).
    -   **Summary:** Format as `"Interview with [Candidate Name]"`
    -   **Description:** Include key details: University/Background, Research Areas, **Qualifying Publication(s)**, and contact information (email/phone) extracted from their materials.
    -   **Time Zone:** Explicitly set the time zone based on user instruction (e.g., `"Asia/Hong_Kong"`).

### Phase 4: Reporting
1.  **Present Summary:** Deliver a final summary to the user in a structured format (e.g., a table).
2.  **Include:**
    -   List of qualified candidates and their qualifying publications.
    -   Final interview schedule (Candidate, Date, Time Slot, Duration).
    -   List of non-qualified candidates and the reason (e.g., "No first-author publications").
    -   Confirmation that events were added to Google Calendar.

## 3. Key Logic & Rules (For Scripting)
-   **Authorship Parsing Rule:** A qualifying publication must be explicitly associated with the candidate via phrases like "First Author", "Sole Author", or "Author" in a context that implies primary/independent authorship. Implied authorship from being listed first in a citation is not sufficient unless explicitly stated in the text.
-   **Date Resolution Rule:** When relative dates like "tomorrow" are used, the anchor date is the most recent date found among the retrieved candidate source materials (e.g., the latest email date).
-   **Scheduling Rule:** Interviews must be scheduled within the user-defined working hour window and must last *at least* the specified minimum duration.

## 4. Error Handling & Edge Cases
-   If no candidates meet the criteria, inform the user and do not schedule any events.
-   If the source materials contain no date information, ask the user to clarify the interview dates.
-   If a candidate's material has no clear publication section, treat them as having no qualifying publications.
-   If scheduling conflicts occur (e.g., not enough slots), report the conflict to the user and schedule as many as possible.

## 5. Required Tools
-   Email search & read tools (e.g., `emails-search_emails`, `emails-read_email`).
-   Google Calendar event creation tool (`google_calendar-create_event`).
-   (Optional) File search & read tools for non-email sources.
