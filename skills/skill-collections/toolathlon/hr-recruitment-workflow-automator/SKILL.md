---
name: hr-recruitment-workflow-automator
description: Processes job applications by extracting candidate information from resumes (PDF files), updating HR databases (Notion), and managing candidate communications. Automates the end-to-end recruitment workflow including resume parsing, database updates, and automated rejection emails.
---
# HR Recruitment Workflow Automator

## Purpose
Automate the processing of job applications by:
1. Reading and parsing multiple PDF resumes to extract candidate details
2. Updating Notion HR databases with candidate information
3. Comparing applications against open positions based on headcount data
4. Sending automated rejection emails for closed positions
5. Managing database entries (create, update, delete)

## Trigger Phrases
- "update candidate information"
- "process resumes"
- "HR record management"
- "send rejection emails"
- "PDF resume processing"
- "Notion database updates"
- "recruitment workflow automation"

## Core Workflow

### Phase 1: Discovery & Setup
1. **Locate HR Record Page**
   - Search Notion for "HR Record" page
   - Identify the page containing recruitment databases

2. **Find Resume Files**
   - Scan workspace directories for PDF resumes
   - Typically found in `/workspace/dumps/workspace/cvs/` or similar paths

3. **Understand Database Structure**
   - Retrieve schema for "Job Positions" database (contains Position and Head Counts)
   - Retrieve schema for "Candidates" database (contains Name, Email, Applied Position, Highest Degree, School)

### Phase 2: Data Processing
1. **Extract Resume Information**
   - For each PDF resume, extract:
     - Candidate Name
     - Email Address
     - Applied Position
     - Highest Degree
     - School/University
   - Use PDF parsing tools to read text content

2. **Analyze Job Positions**
   - Query "Job Positions" database to identify:
     - Positions with Head Counts > 0 (open for recruitment)
     - Positions with Head Counts = 0 (closed/not hiring)

### Phase 3: Database Operations
1. **Clean Existing Data**
   - Delete sample/test entries from Candidates database
   - Remove any placeholder or incomplete records

2. **Create Candidate Entries**
   - Create new database entries for all candidates
   - Add detailed candidate information as page content (due to API limitations)

### Phase 4: Communication Management
1. **Identify Candidates for Rejection**
   - Compare each candidate's applied position against open positions
   - Flag candidates applying for positions with Head Counts = 0

2. **Send Rejection Emails**
   - Use standardized rejection email template
   - Send personalized emails to candidates with closed positions
   - Include proper line breaks and formatting as specified

## Key Considerations

### API Limitations
- Notion API wrapper may restrict setting database properties beyond "title"
- Workaround: Add candidate details as page content using block children API
- Database properties (School, Highest Degree, Applied Position, Email) may remain empty in table view but are accessible in page content

### Data Accuracy
- Extract information strictly from resume content
- Do not modify, add, or remove any words from original resumes
- Ensure email addresses and positions match exactly what's in the PDFs

### Error Prevention
- Verify position names match exactly between resumes and Job Positions database
- Double-check email addresses before sending
- Confirm headcount values before determining position availability

## Required Tools
- Notion API (search, database queries, page creation, block management)
- PDF parsing tools
- Filesystem navigation
- Email sending capabilities

## Output
- Updated Notion Candidates database with all applicant information
- Rejection emails sent to applicable candidates
- Clean database without sample entries
- All information filled according to resume content
