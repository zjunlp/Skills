---
name: canvas-exam-schedule-compiler
description: When the user needs to compile a schedule of upcoming final exams from Canvas LMS, this skill retrieves enrolled courses, scans course announcements for exam information, filters based on date and exam requirements, and populates an Excel spreadsheet with the remaining exams. It handles 1) Fetching user's enrolled courses from Canvas, 2) Retrieving announcements for each course to find exam details, 3) Filtering out courses with no exams, already-ended courses, or exams that have already passed, 4) Processing exam information (dates, times, locations, proctors, exam types), 5) Formatting data into a structured Excel table with proper headers. Key triggers include requests for 'exam schedule', 'final exams', 'Canvas announcements', 'remaining exams', or tasks involving 'exam_schedule.xlsx' files.
---
# Instructions

## Primary Objective
Compile a schedule of remaining final exams by extracting information from Canvas course announcements and populating an Excel file (`exam_schedule.xlsx`).

## Core Workflow

### 1. Initial Setup & File Check
- **First Action:** Always check if the target Excel file (`exam_schedule.xlsx`) exists in `/workspace/dumps/workspace/`. Use `terminal-run_command` with `ls -la`.
- **Preserve Headers:** If the file exists, read it to confirm the column structure. The expected headers are:
    - Course Code
    - Course Name
    - Proctor Name
    - Open-book/Closed-book
    - Final Date (MM/DD/YYYY)
    - Start Time (HH:MM)
    - Duration (minutes)
    - Location
    - Information Source(Announcement/Email/Message)
    - Course Credit
- **Do not change these headers or the file name.**

### 2. Gather Course Data
- **Fetch Enrolled Courses:** Use `canvas-canvas_list_courses` to get all courses where the user has a student enrollment.
- **For Each Course:** Use `canvas-canvas_list_announcements` with the course `id` to retrieve all announcements. Scan the `message` field of each announcement for exam details.

### 3. Extract & Process Exam Information
For each course, analyze announcements to find:
- **Final Exam Announcement:** Look for titles containing "Final Exam", "Exam Announcement", or similar. The message typically contains details like Date, Time, Duration, Location, and Exam Type (Open/Closed book).
- **Proctor Information:** By default, the proctor is the course instructor (first teacher listed in course data). Override this **only** if an announcement explicitly states a different proctor (e.g., "proctor will be replaced by...").
- **Missing Information:** If any detail (Date, Time, Location) is stated as "to be confirmed", "TBD", or is completely absent, mark the corresponding cell as `TBD`.
- **No Exam:** If an announcement explicitly states "does not have a final exam" or "no final exam", exclude the entire course from the schedule.
- **Course Code/Name Formatting:** Remove the "-x" suffix (e.g., "CS101-3" → "CS101", "Introduction to Computer Science-3" → "Introduction to Computer Science").

### 4. Apply Filters
Exclude a course if **ANY** of the following is true:
- The course has an `end_at` date that is in the past (relative to the user-provided "today" date).
- The final exam date has already passed (relative to the user-provided "today" date). If the exam is scheduled for "today", consider the start time; if the time has passed, exclude it.
- An announcement explicitly states there is no final exam for the course.
- The user is explicitly exempt (e.g., "students with exemption score... may be exempt"). If exemption is conditional ("may be exempt") and not confirmed for the user, **include** the course.

### 5. Populate the Excel File
- Use the provided Python script (`scripts/compile_schedule.py`). It handles:
    - Reading the existing file template.
    - Applying all filtering logic based on the current date.
    - Formatting the data correctly (dates as MM/DD/YYYY, times as HH:MM, duration as a number).
    - Writing the cleaned data back to the same file path.
- The script is the **single source of truth** for the data processing logic. Refer to it for exact implementation details.

### 6. Verification & Completion
- After running the script, briefly display the first few rows of the generated file to confirm successful execution.
- Provide a concise summary to the user: number of courses included, key exclusions, and the location of the output file.

## Important Notes
- **Email Fallback:** If an announcement mentions exam details were sent via email, check using `emails-search_emails`. If no emails are found, mark the information as `TBD`.
- **Date Context:** The user will provide the current date (e.g., "today is January 16, 2025"). Use this as the reference for filtering past exams.
- **Error Handling:** If the Canvas API calls fail, inform the user and suggest checking connectivity. If the Excel file is missing critical headers, report the error before proceeding.
