---
name: course-schedule-analyzer
description: When the user needs to find and select courses from a university schedule (typically in PDF format) based on multiple criteria including course names, course types, class times, exam formats, and campus locations. This skill extracts course information from PDF schedules, searches for specific courses by name or category, filters results based on constraints like day of week, exam type (open-book/closed-book), and time slots, checks for scheduling conflicts, and outputs organized course selection options in structured formats like Excel files.
---
# Instructions

## 1. Understand the Request
- Identify the user's required courses, categories, and constraints.
- Key constraints often include:
  - Specific course names (e.g., "自然语言处理", "软件体系结构").
  - Course categories (e.g., "思政B", "通识教育核心课程").
  - Day of week (e.g., "Monday").
  - Exam type (e.g., "开卷" for open-book).
  - Campus location.
  - Output format (e.g., Excel files with specific headers).
- Clarify if multiple valid options should be saved as separate files.

## 2. Locate and Inspect Reference Files
- List the workspace directory to find the schedule PDF and any reference files (e.g., `reference_format.xlsx`).
- Read the reference format file to understand the required output structure (headers, data format).
- Inspect the PDF file to understand its structure (total pages, layout).

## 3. Extract and Search Course Data
- Use PDF search functions to locate specific courses by name or category.
- For each required course/category, perform targeted searches.
- Extract detailed information by reading relevant page ranges.
- Parse the extracted text to identify:
  - Course ID
  - Course name
  - Instructor
  - Campus
  - Class time (day and periods)
  - Enrollment limit
  - Credits
  - Assessment method (e.g., "闭卷", "开卷", "论文")
  - Exam time
  - Course selection restrictions

## 4. Apply Filters and Check Constraints
- Filter courses based on:
  - Day of week (e.g., Monday courses only).
  - Exam type (e.g., only "开卷" exams).
  - Other user-specified constraints.
- For courses with multiple sections, identify all valid options.
- Check for time conflicts between fixed required courses.
- Ensure selected courses do not overlap in schedule.

## 5. Generate Course Selection Schemes
- Combine fixed required courses with optional courses that meet all constraints.
- For each valid combination, create a distinct selection scheme.
- Structure the data according to the reference format.
- Ensure values are in Chinese while headers remain in English.

## 6. Create Output Files
- For each selection scheme, create a separate Excel file.
- Use a programmatic approach (e.g., Python with openpyxl) to generate files efficiently.
- Save files with clear naming (e.g., `course_selection_scheme_1.xlsx`).
- Clean up the workspace by removing temporary or reference files if requested.

## 7. Verify and Summarize
- Verify the content of generated files matches the expected format.
- Provide a clear summary to the user listing:
  - The fixed courses included in all schemes.
  - The variable options for each scheme.
  - File names and locations.
