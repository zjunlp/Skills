---
name: canvas-student-enrollment-manager
description: Manages student enrollments in Canvas LMS courses for late-joining students. Handles reading student lists from CSV files, searching for courses, identifying enrolled vs. unenrolled students, enrolling missing students, and sending personalized private messages about assignment policies.
---
# Instructions

## Overview
Use this skill when you need to manage student enrollments in Canvas, particularly for students who have joined a course late (e.g., after changing majors) and have missed initial assignments. The skill automates the process of enrolling students from a provided list and notifying them of relevant grade policies.

## Core Workflow

### 1. Locate and Parse the Student List
- **Action:** Search the workspace for a CSV file containing student information. The expected format is `Name,email` (case-sensitive headers).
- **Tool:** Use `filesystem-list_directory` to explore `/workspace/dumps/workspace` or similar directories.
- **Next:** Once found, read the file with `filesystem-read_file` to parse the list of students.

### 2. Find the Target Course in Canvas
- **Action:** Identify the correct Canvas course by name (e.g., "Introduction to AI-8").
- **Tool:** Use `canvas-canvas_list_account_courses` with a `search_term`. You may need to first get the root account ID using `canvas-canvas_get_account`.
- **Verification:** Confirm the course details (ID, name) using `canvas-canvas_get_course`.

### 3. Identify Existing and Missing Students
- **Action:** Determine which students from the CSV are already enrolled and which are not.
- **Sub-step A - Find Student User IDs:** For each student email in the CSV, search the Canvas account users using `canvas-canvas_list_account_users` with the email as the `search_term`. Record the `id` for each found user.
- **Sub-step B - Get Current Enrollments:** Retrieve the list of currently enrolled students for the course using `canvas-canvas_get_course_grades`. This returns an array of enrollment objects containing `user_id`.
- **Analysis:** Compare the list of found user IDs (from Sub-step A) against the list of enrolled `user_id`s (from Sub-step B). Students whose IDs are not in the enrollment list need to be enrolled.

### 4. Enroll Missing Students
- **Action:** Enroll each identified missing student into the target course.
- **Tool:** For each student user ID, call `canvas-canvas_enroll_user`.
- **Parameters:**
    - `course_id`: The ID of the target course.
    - `user_id`: The Canvas ID of the student.
    - `role`: `"StudentEnrollment"`
    - `enrollment_state`: `"active"`
- **Note:** Enroll students one by one. The tool will return a confirmation for each successful enrollment.

### 5. Notify Newly Enrolled Students
- **Action:** Send a private, personalized message to each student you just enrolled.
- **Tool:** Use `canvas-canvas_create_conversation` for each student.
- **Message Content:** The message should:
    1. Welcome the student to the course.
    2. Acknowledge they joined late/missed the first assignment.
    3. Clearly state the policy: **Their first assignment grade will be the same as their second assignment grade.**
    4. Encourage them to complete the second assignment diligently, as it will determine both grades.
    5. Offer a point of contact for questions.
- **Parameters:**
    - `recipients`: An array containing the single student's user ID (e.g., `["75"]`).
    - `subject`: A clear subject (e.g., "Important: First Assignment Grade Policy for New Students").
    - `body`: The personalized message. Use the student's first name from the CSV data.

### 6. Final Summary
- **Action:** Provide a concise summary to the user.
- **Content:** List the number of students enrolled, their names, and confirm that notification messages were sent. Mention any students who were already enrolled.

## Error Handling & Considerations
- **Missing Students:** If a student email from the CSV is not found in the Canvas system, note this and skip them. Inform the user.
- **Course Not Found:** If the course search yields no results, double-check the course name and account. You may need to list all courses to find it.
- **Large Batches:** The trajectory shows individual API calls for each student search and enrollment. This is acceptable for moderate-sized lists. For very large lists, consider the efficiency but prioritize reliability.
- **Message Personalization:** Extract the student's first name from the `Name` field in the CSV (typically the first word) to personalize the message.

## Key Tools Used
- `filesystem-list_directory`
- `filesystem-read_file`
- `canvas-canvas_get_account`
- `canvas-canvas_list_account_courses`
- `canvas-canvas_get_course`
- `canvas-canvas_list_account_users`
- `canvas-canvas_get_course_grades`
- `canvas-canvas_enroll_user`
- `canvas-canvas_create_conversation`
