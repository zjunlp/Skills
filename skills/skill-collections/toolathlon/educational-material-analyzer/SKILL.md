---
name: educational-material-analyzer
description: When a student requests help with studying course materials (presentations, PDFs, lecture notes) and needs comprehensive study notes created. This skill extracts key concepts from educational files (PPTX, PDF), summarizes definitions and differences, includes code examples with explanations, and answers assignment questions. It's triggered by requests involving 'study', 'lecture notes', 'course materials', 'presentation analysis', or 'homework explanation' with file types like .pptx, .pdf, .md.
---
# Instructions

## 1. Initial Setup & File Discovery
1.  **Identify Request Type:** Confirm the user is a student needing help with course materials (presentations, PDFs, notes).
2.  **Locate Files:** The user will typically mention specific filenames (e.g., `Compile.pptx`, `HW.PDF`).
    *   Use `filesystem-list_directory` to verify file existence and correct paths in the workspace.
    *   Handle case sensitivity or extension variations (e.g., `.pdf` vs `.PDF`).

## 2. Content Extraction & Analysis
3.  **Open Presentation:** Use `pptx-open_presentation` on the specified `.pptx` file to get a presentation ID.
4.  **Extract Presentation Text:** Use `pptx-extract_presentation_text` with `include_slide_info: true` on the obtained ID. This provides structured slide data including titles, text, and tables.
5.  **Read PDF Content:** Use `pdf-tools-read_pdf_pages` on the specified `.pdf` file (e.g., homework). Extract all relevant pages.

## 3. Structured Note Creation
6.  **Create Output File:** Use `filesystem-write_file` to create a new markdown file (e.g., `NOTE.md`).
7.  **Build Note Structure:** Synthesize the extracted content into a well-organized study note. Follow this template:
    *   **Header:** Course name and chapter/topic.
    *   **Overview:** Summarize the main topic's purpose (from early slides).
    *   **Core Definitions:** Clearly define key terms (e.g., "Symbol Table", "Environment").
    *   **Comparative Analysis:** When concepts have different styles/implementations (e.g., Functional vs. Imperative), create a dedicated section comparing them. Use a table for clear contrast.
    *   **Code Integration:** For all code snippets found in the materials:
        *   Create a dedicated "Code Examples" section.
        *   Group related code blocks logically.
        *   **For each block:** Provide the exact code and a concise **Explanation** beneath it, detailing what it does and why it's important.
    *   **Assignment Analysis:** For homework/assignment PDFs:
        *   List each question.
        *   Provide a clear, step-by-step explanation of the solution.
        *   State the final answer.
    *   **Summary:** Add a final summary or comparison table recapping the most critical takeaways.

## 4. Principles for Content Synthesis
*   **Clarity Over Comprehensiveness:** Prioritize explaining the "why" behind concepts, not just listing "what".
*   **Connect Theory & Practice:** Explicitly link definitions to their code implementations.
*   **Answer Directly:** For assignment questions, provide the answer clearly after the explanation.
*   **Verify Output:** Optionally, use `filesystem-read_file` to confirm the note was created successfully.

## 5. Final Response
8.  Inform the user the note has been created and provide a brief summary of its contents, highlighting the definitions, comparisons, code explanations, and homework answers you included.
