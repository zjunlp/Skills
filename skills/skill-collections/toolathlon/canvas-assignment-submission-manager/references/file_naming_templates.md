# File Naming Convention Templates

## Standard Patterns
When an assignment description specifies a naming convention, it typically follows one of these patterns:

1. **`{StudentID}_{CourseCode}_Assignment{Number}.{ext}`**
   - Example: `2201210606_FILM101_Assignment4.md`
   - Components:
     - `{StudentID}`: Retrieved from user memory.
     - `{CourseCode}`: From Canvas course `course_code` (e.g., FILM101).
     - `{AssignmentNumber}`: Usually found in the assignment `name` (e.g., "Assignment 4").
     - `{ext}`: Must be one of the `allowed_extensions` (e.g., `md`).

2. **`{StudentID}_{CourseCode}_{AssignmentName}.{ext}`**
   - Example: `2201210606_FILM101_Cinematography.md`
   - AssignmentName is a simplified version of the assignment name.

3. **`{LastName}_{FirstName}_Assignment{Number}.{ext}`**
   - Example: `Edwards_Zachary_Assignment4.md`

## Extraction Logic
1. Parse the assignment `description` text for phrases like:
   - "should be named as"
   - "file should be named"
   - "upload file should be named"
   - "name your file as"

2. Use regex to extract the template pattern. Example regex: `(\w+)_(\w+)_Assignment(\d+)\.(\w+)`

3. Replace placeholders with actual data:
   - Student ID from memory.
   - Course code from Canvas course data.
   - Assignment number from assignment name (extract digits after "Assignment ").

## Fallback Strategy
If no specific naming convention is found in the description:
1. Use a safe default: `{StudentID}_{AssignmentNameSanitized}.{ext}`
2. Sanitize the assignment name: remove special characters, replace spaces with underscores.
3. Example: `2201210606_Cinematography_and_Visual_Storytelling.md`
