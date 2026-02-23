# Common Email Patterns for Submission Tracking

## Subject Line Patterns
When searching for submissions, look for these common patterns in email subjects:

### Academic Assignments
*   `[Course]-[AssignmentType]-[ID]-[Name]` - `nlp-presentation-2000016605-Carol Carter`
*   `Submission: [Assignment Name] - [Student Name]`
*   `[Student ID] - [Assignment]`
*   `[Course Code] Assignment [Number] Submission`

### Project/File Submissions
*   `[Project Name] - Final Deliverables - [Team/Name]`
*   `Attached: [Filename] for [Purpose]`
*   `[Document Type] Submission: [Title]`

### General Keywords to Try
If a specific search fails, iterate through these related terms:
1.  Primary term: `"final presentation"`
2.  Broader term: `"presentation"`
3.  Course code: `"NLP"`
4.  Assignment type: `"submission"`, `"assignment"`, `"homework"`
5.  File indicators: `"attached"`, `"attachment"`, `"file"`

## Metadata Extraction
From each relevant email, extract:
*   **Subject:** Parse for IDs, names, assignment types.
*   **From:** Sender's email address (may be student or system).
*   **Date:** Submission timestamp.
*   **Body/Attachments:** May contain additional identifiers or confirmation details.

## Cross-Referencing Strategy
1.  **Identifier Matching:** Use the most unique identifier available (Student ID > Name).
2.  **Normalization:** Remove spaces, convert to consistent case when comparing.
3.  **Status Filtering:** Always check for status notes like "withdrew", "dropped", "auditing", "incomplete" before flagging as non-submitter.
