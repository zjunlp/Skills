# Canvas LMS API Guide for Assignment Submission

## Key Endpoints Used

### 1. List Courses
**Tool**: `canvas-canvas_list_courses`
**Purpose**: Identify the target course by name or code.
**Response Fields**:
- `id`: Course ID (required for subsequent calls)
- `name`: Course name (e.g., "Cinema Culture Appreciation")
- `course_code`: Short code (e.g., "FILM101")

### 2. List Assignments with Submissions
**Tool**: `canvas-canvas_list_assignments`
**Parameters**:
- `course_id`: (Required) ID from course listing.
- `include_submissions`: Set to `true` to get the user's submission status.

**Critical Response Fields**:
- `id`: Assignment ID (required for submission).
- `name`: Assignment name (e.g., "Assignment 4: Cinematography and Visual Storytelling").
- `description`: Often contains file naming instructions.
- `submission_types`: Array like `["online_upload"]` or `["online_text_entry", "online_upload"]`.
- `allowed_extensions`: Array like `["md"]` for markdown files.
- `submission`: An object containing the user's submission status.
  - `workflow_state`: "unsubmitted", "submitted", "graded", etc.
  - `score`: The grade if graded.
  - `submitted_at`: Timestamp of submission.

### 3. Submit Assignment with File
**Tool**: `canvas-canvas_submit_assignment_with_file`
**Parameters**:
- `course_id`: Course ID.
- `assignment_id`: Assignment ID.
- `file_path`: Absolute path to the file to upload.

**Success Response**: Contains `workflow_state: "submitted"` and `submitted_at` timestamp.

## Workflow State Interpretation
- `"unsubmitted"`: No submission exists. Action: Submit if missing per grade sheet.
- `"submitted"`: Submission exists but not graded. Action: Do not re-submit.
- `"graded"`: Submission has been graded. Action: Do not re-submit.
- `"pending_review"`: Submission is under review. Action: Do not re-submit.

## File Naming & Validation
- Always check the `description` field for instructions like "The upload file should be named as StudentID_FILM101_Assignment4.md".
- Validate the file extension against `allowed_extensions`.
- The tool `canvas-canvas_submit_assignment_with_file` handles the actual upload; you only need to provide the correct local file path.
