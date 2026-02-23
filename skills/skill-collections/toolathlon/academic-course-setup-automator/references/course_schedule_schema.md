# Course Schedule Schema

## Expected Format
The course schedule is typically provided as a markdown table with the following columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| Course Name | Full course name with section | "English Literature-7" |
| Instructor | Instructor's full name | "Melissa Sanchez" |
| Class Time | Day and period information | "Monday 7,8,9" |
| Academic Administrator | Admin responsible for setup | "admin3" |

## Parsing Rules
1. **Course Name Format**: Usually follows "Course Title-Section" pattern
2. **Instructor Names**: Match exactly with CSV instructor list
3. **Admin Filtering**: Only process rows where "Academic Administrator" matches current user
4. **Time Format**: Preserve exact text for announcements

## Sample Data
