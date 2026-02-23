---
name: homework-autograder
description: When the user needs to grade programming assignments submitted via email attachments and upload scores to a learning management system (Canvas). This skill handles the complete workflow searching emails for submissions, downloading Python file attachments, executing code to test for errors, comparing outputs against assignment requirements, and submitting grades to Canvas with appropriate feedback. Triggers include keywords like 'grade homework', 'check submissions', 'Canvas grading', 'Python assignment', 'email attachments', and file types like .py files and assignment specifications.
---
# Instructions

## 1. Understand the Task
- The user wants you to grade a specific programming assignment (e.g., "homework2").
- Submissions are sent via email as Python file attachments.
- Grades must be posted to a Canvas course.
- You need to find the latest submission for each student if multiple exist.
- The core logic is: if the Python file runs without errors and produces correct output, give full points (e.g., 10). If it has any error (syntax, runtime, logic) or produces incorrect output, give 0 points.

## 2. Gather Prerequisites
**Always start by reading these files:**
- The assignment specification (e.g., `assignments/homework2.md`). Understand the exact requirements, function signature, and expected output format.
- The student roster with Canvas IDs (e.g., `student_canvas_ids.csv`). Map student names/emails to their Canvas user IDs.

## 3. Find Submission Emails
- Search the email inbox for relevant emails using keywords like the assignment name ("homework2"), course code ("CS5123"), or problem name ("two_sum").
- If initial search is too narrow, broaden it (e.g., search for "homework").
- Identify all emails that appear to be submissions for the target assignment.
- **For students with multiple submissions:** Use only the latest one (most recent date). Note the email ID of the latest submission.

## 4. Download and Inspect Attachments
- Read each identified email to confirm it has a Python attachment.
- Download all Python file attachments to a local directory (e.g., `homework2_submissions/`).
- Optionally, read the file contents to get an initial sense of the code quality and potential errors.

## 5. Execute and Grade Each Submission
For each downloaded Python file:
1. **Run it in a terminal.** Use `python <filename.py>`.
2. **Interpret the result:**
   - **Success (return code 0):** The code executed without syntax or runtime errors. You must still verify the output is correct according to the assignment spec (e.g., indices are in ascending order). Use the `local-python-execute` tool for more precise testing if needed.
   - **Failure (return code non-zero):** The code has an error. Common errors include:
     - `SyntaxError` (e.g., missing colon)
     - `ModuleNotFoundError` (importing non-existent modules)
     - `IndexError` (list index out of range)
     - Logic errors that cause wrong output (this requires checking the actual printed results against expected results).
3. **Assign a score:** 10 for fully correct, 0 for any failure or incorrect output.
4. **Prepare feedback:** Note the specific error or issue for the Canvas comment.

## 6. Submit Grades to Canvas
1. Identify the correct Canvas course and assignment ID (use `canvas-canvas_list_courses` and `canvas-canvas_list_assignments`).
2. For each student, use `canvas-canvas_submit_grade` with:
   - `course_id`: The target course ID.
   - `assignment_id`: The target assignment ID.
   - `user_id`: The student's Canvas user ID (from the roster CSV).
   - `grade`: "10" or "0".
   - `comment`: A concise, helpful explanation (e.g., "Code runs correctly and passes all test cases." or "SyntaxError: missing colon on line 6.").

## 7. Provide a Final Summary
- Present a table showing each student, their Canvas ID, score, and brief reason.
- Confirm all grades have been submitted.

## Key Considerations
- **Student Mapping:** Ensure you correctly map the email sender (student name/email) to their Canvas ID from the roster. Names might not match exactly.
- **Latest Submission Only:** Always check dates and use the most recent email for each student.
- **Output Validation:** A running script is not enough. The output must match the assignment requirements exactly (e.g., order of indices matters).
- **Error Types:** Distinguish between syntax/runtime errors (code doesn't run) and logic errors (code runs but gives wrong answer). Both score 0.
- **Workspace Management:** Create organized directories for downloads to avoid clutter.

## Tools You Will Use
- `filesystem-read_file`, `filesystem-read_multiple_files`, `filesystem-create_directory`
- `emails-search_emails`, `emails-get_emails`, `emails-read_email`, `emails-download_attachment`
- `terminal-run_command`
- `local-python-execute` (for targeted testing)
- `canvas-canvas_list_courses`, `canvas-canvas_list_assignments`, `canvas-canvas_submit_grade`
