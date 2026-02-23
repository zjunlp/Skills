# Exemption Logic Reference

## Common Exemption Patterns

### 1. English Proficiency Exemption
- **Trigger**: Course announcement indicates exemption for high entrance exam scores
- **Data Source**: User memory contains `"Score for the entrance English exam: 95"`
- **Logic**: If score ≥ threshold (e.g., 90), exclude ENG101 quizzes
- **Example Announcement**: "Students with exemption score of 90 or above may be exempt from all quizzes."

### 2. Practice vs Graded Quizzes
- **Field**: `quiz_type`
- **Graded**: `"assignment"` - Include in report
- **Practice**: `"practice_quiz"` - Exclude from report
- **Identification**: Check if `assignment_id` is present (graded quizzes have this)

### 3. Already Submitted Work
- **Field**: `submission.workflow_state`
- **Submitted**: `"submitted"` or has `submitted_at` timestamp - Exclude
- **Unsubmitted**: `"unsubmitted"` - Include

## Decision Flowchart

