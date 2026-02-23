# Output Format Reference

## Excel File Structure
All course selection scheme files must follow this format:

### Headers (English)
1. Course name
2. Course ID
3. Instructor
4. Campus
5. Class Time
6. Enrollment
7. Credits
8. Assessment Method
9. Exam Time
10. Course Selection Restrictions

### Data Values (Chinese)
- All content values should be in Chinese.
- Numeric values (Enrollment, Credits) should be numbers.
- Class Time format: "周一 第2-3节 [1-16]" (Day Periods [Weeks])
- Exam Time format: "YYYY-MM-DD HH:MM-HH:MM"
- Course Selection Restrictions: Use "无" for no restrictions, or list restrictions separated by commas.

### Example Row
| Course name | Course ID | Instructor | Campus | Class Time | Enrollment | Credits | Assessment Method | Exam Time | Course Selection Restrictions |
|-------------|-----------|------------|---------|------------|------------|---------|-------------------|-----------|-------------------------------|
| 自然语言处理 | COMP130141.01 | 黄萱菁 | 邯郸校区 | 周一 第2-3节 [1-16] | 60 | 2 | 闭卷 | 2025-12-22 08:55-10:40 | 23 22 计算机科学与技术, 大数据与人工智能(2024), 大数据与人工智能(2023), 大数据与人工智能(2022) |

## Naming Convention
- Course selection scheme files: `course_selection_scheme_{n}.xlsx`
- Where `{n}` starts from 1 and increments for each valid combination.

## File Cleanup
- Remove the reference format file after use if requested.
- Keep only the final course selection scheme files in the workspace.
