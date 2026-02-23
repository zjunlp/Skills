# Canvas Quiz API Quick Reference

## Key Endpoints Used

### 1. List Courses
**Endpoint:** `GET /api/v1/courses`
**Tool:** `canvas-canvas_list_courses`
**Purpose:** Find the target course ID.
**Response Fields to Note:**
- `id`: Course ID (required for all subsequent calls)
- `name`: Course name
- `course_code`: Course code (e.g., "AH101")

### 2. Create Quiz
**Endpoint:** `POST /api/v1/courses/:course_id/quizzes`
**Tool:** `canvas-canvas_create_quiz`
**Required Parameters:**
- `course_id`: The course ID
- `title`: Quiz title (string)
- `quiz_type`: Use "assignment" for graded quizzes. Other valid types: "practice_quiz", "graded_survey", "survey"
- `published`: Set to `false` to create as draft

**Important Notes:**
- The "graded" quiz type may cause "invalid_quiz_type" error in some Canvas instances. "assignment" is more reliable.
- The quiz will be associated with an assignment automatically (see `assignment_id` in response).

### 3. Create Quiz Question
**Endpoint:** `POST /api/v1/courses/:course_id/quizzes/:quiz_id/questions`
**Tool:** `canvas-canvas_create_quiz_question`
**Required Parameters for Multiple Choice:**
- `course_id`: Course ID
- `quiz_id`: Quiz ID from creation step
- `question_name`: Short identifier for the question
- `question_text`: The actual question text
- `question_type`: "multiple_choice_question"
- `points_possible`: Point value (integer)
- `position`: Order in quiz (1-indexed)
- `answers`: Array of answer objects

**Answer Object Structure:**
