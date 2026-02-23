---
name: canvas-quiz-creator
description: When the user needs to create or manage quizzes in Canvas Learning Management System, particularly when they provide quiz questions and answers that need to be formatted and uploaded. This skill handles the complete workflow including 1) Finding the target course in Canvas, 2) Creating a new quiz with proper metadata (title, description, quiz type), 3) Adding multiple-choice questions with correct answer weights, 4) Verifying the quiz creation and providing a summary. Use this skill when users mention 'Canvas quiz', 'create test on Canvas', 'upload questions to Canvas', or when they provide structured quiz questions with multiple-choice options that need to be formatted for Canvas API integration.
---
# Instructions

## 1. Parse User Input
- Extract the quiz title from the user's request. If not specified, ask for clarification.
- Extract the list of questions. Each question should have:
    - Question text
    - Multiple-choice options (typically A, B, C, D)
    - The correct answer (identified by letter or explicit text)
- Note: The user may specify points per question (e.g., "one point each"). Default to 1 point if not specified.

## 2. Find the Target Course
- Use `canvas-canvas_list_courses` to retrieve the list of available courses.
- Identify the target course. The user may specify the course name (e.g., "Art History") or you may need to infer it from context. If ambiguous, ask for clarification.
- **Critical:** Store the course ID for all subsequent API calls.

## 3. Create the Quiz
- Use `canvas-canvas_create_quiz` with the following parameters:
    - `course_id`: The ID from step 2.
    - `title`: The quiz title from step 1.
    - `description`: Generate a brief, relevant description summarizing the quiz topics. If the user provides one, use it.
    - `quiz_type`: Use "assignment". (The trajectory shows "graded" caused an error; "assignment" succeeded.)
    - `published`: Set to `false` initially. The quiz can be published later by the instructor.
- If the API call fails (e.g., invalid quiz type), retry with `quiz_type` set to "assignment".
- **Critical:** Store the returned quiz ID.

## 4. Add Questions to the Quiz
- For each question extracted in step 1:
    - Format the question text. Ensure any quotation marks are properly escaped for JSON.
    - Prepare the `answers` array. Each answer object must have:
        - `text`: The answer option text.
        - `weight`: 100 for the correct answer, 0 for all incorrect answers.
    - Use `canvas-canvas_create_quiz_question` with these parameters:
        - `course_id`: The course ID.
        - `quiz_id`: The quiz ID from step 3.
        - `question_name`: A descriptive name (e.g., "Question 1: [Topic]").
        - `question_text`: The full question text.
        - `question_type`: "multiple_choice_question".
        - `points_possible`: The point value for this question.
        - `position`: The question number (1-indexed).
        - `answers`: The array of answer objects.
- Process questions sequentially. The API may support batch creation, but the trajectory shows individual calls are reliable.

## 5. Verify and Summarize
- Use `canvas-canvas_get_quiz` to fetch the final quiz details and confirm creation.
- Use `canvas-canvas_list_quiz_questions` to list all added questions and verify the count.
- Present a clear summary to the user including:
    - Quiz title and course name.
    - Direct URL to the quiz (from the `html_url` field).
    - Total points and number of questions.
    - A list of each question with its correct answer clearly marked.
    - A note that the quiz is unpublished by default.

## 6. Finalize
- Use `local-claim_done` to signal successful completion.
- Do not publish the quiz unless explicitly requested by the user. The default state should be unpublished.
