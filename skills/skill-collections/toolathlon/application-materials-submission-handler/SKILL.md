---
name: application-materials-submission-handler
description: When the user needs to submit application materials (for academic programs, jobs, grants, etc.) based on email instructions with specific formatting requirements. This skill handles the complete workflow reading email instructions with submission guidelines, retrieving personal information from memory, organizing files into specified folder structures, processing PDFs (reading content, merging multiple files), creating directory hierarchies, compressing folders into ZIP files, and sending formatted emails with attachments to specified recipients.
---
# Instructions

## 1. Understand the Request
- The user will ask you to submit application materials based on email instructions.
- The request typically references an email from a specific sender (e.g., "kaiming") containing submission guidelines.
- The subject line for the outgoing email is often specified (e.g., `PhD Application Materials Submission (Student ID: {studentid})`).
- The user's personal information is stored in memory.

## 2. Retrieve Essential Information
- **Search for the instruction email**: Use `emails-search_emails` with a query (e.g., sender name or relevant keywords) to find the email containing submission guidelines.
- **Read the instruction email**: Use `emails-read_email` to get the full content. Extract:
    - Recipient email address for submission (e.g., `myersj@mcp.com`).
    - Required folder structure and naming conventions.
    - Any specific file processing instructions (e.g., "merge award certificates into a single PDF sorted by date").
- **Read user's personal information from memory**: Use `memory-read_graph` to get:
    - Full name.
    - Student ID (or equivalent identifier).
    - Other relevant details (email, phone) for the email body.

## 3. Locate and Inspect Source Materials
- **List the workspace directory**: Use `filesystem-list_directory` to see available files.
- **Examine key files if needed**:
    - For recommendation letters: Use `pdf-tools-read_pdf_pages` to extract professor names from the first page for correct renaming.
    - For award certificates: Use `pdf-tools-read_pdf_pages` to confirm dates for sorting.

## 4. Create the Required Folder Structure
- Construct the root folder name using the user's name and ID without spaces (e.g., `Application_Materials_MaryCastillo_2201210606`).
- Create the full directory hierarchy as specified in the instructions using `terminal-run_command` with `mkdir -p`.
- Typical structure includes:
    - `01_Personal_Information/` (ID_Card.pdf, Photo.jpg, Resume.pdf)
    - `02_Academic_Materials/` with `Awards_Certificates/` subfolder
    - `03_Recommendation_Letters/`
    - `04_Supplementary_Materials/`

## 5. Organize and Process Files
- **Copy files** to their target locations using `terminal-run_command` with `cp`.
- **Rename files** according to conventions (e.g., `Recommendation_Letter_[ProfessorName]-1.pdf`). Professor names should have no spaces.
- **Merge PDFs when required**: Use `pdf-tools-merge_pdfs` to combine multiple award certificates into `All_Awards_Certificates.pdf`. Sort by date (ascending).
- Ensure all files use English names without special characters or spaces.

## 6. Verify and Compress
- **Verify the structure**: Use `filesystem-directory_tree` to confirm the folder hierarchy and file placement.
- **Create a ZIP archive**: Use the provided Python script (`scripts/create_zip.py`) if `zip` command is unavailable. The ZIP file should be named after the root folder (e.g., `Application_Materials_MaryCastillo_2201210606.zip`).

## 7. Send the Submission Email
- Use `emails-send_email` with:
    - **To**: The recipient email from the instructions.
    - **Subject**: Exactly as specified, inserting the student ID (e.g., `PhD Application Materials Submission (Student ID: 2201210606)`).
    - **Body**: A professional email including the user's name, ID, contact information, and a brief description of the attached materials.
    - **Attachments**: The path to the created ZIP file.

## 8. Finalize
- Provide a summary to the user confirming the email was sent, the structure created, and any special processing performed.
- Use `local-claim_done` to indicate task completion.
