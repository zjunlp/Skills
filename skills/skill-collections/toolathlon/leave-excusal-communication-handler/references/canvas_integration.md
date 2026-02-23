# Canvas LMS Integration Guide

## API Endpoints Used

### 1. List Courses
**Endpoint:** `canvas-canvas_list_courses`
**Purpose:** Retrieve all courses the student is enrolled in.
**Key Fields:**
- `id`: Course ID (used for subsequent calls)
- `name`: Course name
- `course_code`: Course code (e.g., FILM101)
- `teachers`: Array of instructor information

### 2. List Assignments with Submissions
**Endpoint:** `canvas-canvas_list_assignments`
**Parameters:**
- `course_id`: Required, from course listing
- `include_submissions`: Set to `true` to get submission status
**Key Submission Fields:**
- `submission.workflow_state`: "submitted", "unsubmitted", "graded"
- `submission.submitted_at`: Date of submission (null if not submitted)
- `submission.missing`: Boolean indicating if assignment is missing
- `submission.excused`: Boolean indicating if assignment is excused

### 3. Submit Assignment with File
**Endpoint:** `canvas-canvas_submit_assignment_with_file`
**Parameters:**
- `course_id`: Course ID
- `assignment_id`: Assignment ID from assignment listing
- `file_path`: Local path to the file to submit
**Important Notes:**
- File naming conventions are often specified in assignment descriptions
- Check `allowed_extensions` in assignment details
- Some assignments may require specific submission types (`online_text_entry`, `online_upload`, etc.)

## Common Patterns

### Identifying Missing Assignments
