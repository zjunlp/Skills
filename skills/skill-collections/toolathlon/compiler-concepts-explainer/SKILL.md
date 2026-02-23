---
name: compiler-concepts-explainer
description: When the user needs explanations of compiler principles concepts, particularly symbol tables, semantic analysis, and related implementation details. This skill provides detailed explanations of functional vs. imperative symbol tables, insertion operations, scope management, and code implementations. It's triggered by technical questions about 'symbol tables', 'compiler principles', 'semantic analysis', 'functional vs. imperative', or 'scope management'.
---
# Skill: Compiler Concepts Explainer

## Primary Objective
Explain compiler principles concepts, with a focus on **Semantic Analysis** and **Symbol Tables**, by extracting and synthesizing information from provided course materials (PPTX, PDF). Create a structured, educational note for the user.

## Core Workflow
1.  **Identify Request:** Recognize user queries about compiler concepts, especially symbol tables, semantic analysis, functional/imperative styles, scope, or homework related to these topics.
2.  **Locate & Extract:** Find and read the relevant source files (e.g., `Compile.pptx`, `HW.pdf`) in the workspace.
3.  **Analyze & Structure:** Parse the extracted content to identify key definitions, differences, code examples, and homework questions/answers.
4.  **Generate Explanation:** Create a comprehensive, well-organized markdown note (`NOTE.md`) that includes:
    *   Clear definitions and comparisons (e.g., Functional vs. Imperative Symbol Tables).
    *   Detailed explanations of insertion operations and their implications.
    *   All relevant code snippets from the materials, presented with clear commentary.
    *   Explanations and answers for any associated homework questions.
5.  **Deliver Output:** Write the final note to the specified file path and confirm completion.

## Detailed Instructions

### Phase 1: Source Material Acquisition
*   **PPTX Files:** Use the `pptx-open_presentation` and `pptx-extract_presentation_text` tools to get the full textual content of the slides. Pay close attention to slides discussing "Symbol Tables," "Functional Style," "Imperative Style," "Efficient Functional/Imperative Symbol Tables," and "The Implementation of Symbol Tables."
*   **PDF Files:** Use `pdf-tools-read_pdf_pages` to read homework assignments or supplementary PDFs. Extract questions and any provided context.
*   **File Discovery:** If a file path is not found, use `filesystem-list_directory` to check the workspace for the correct file (e.g., `HW.pdf` vs `HW.PDF`).

### Phase 2: Content Analysis & Synthesis
Analyze the extracted text to build a mental model covering:
1.  **Fundamentals:** What is a symbol table/environment? What is its role in semantic analysis (binding, lookup, scoping)?
2.  **The Two Paradigms:**
    *   **Functional Style:** Definition, key characteristic (preserves original), typical data structure (Binary Search Tree), and efficiency of insertion/restoration.
    *   **Imperative Style:** Definition, key characteristic (destructive update), typical data structure (Hash Table with chaining + undo stack), and efficiency of insertion/restoration.
3.  **Insertion Operation Differences:** Explicitly contrast how each style handles adding a new binding, especially when shadowing a previous one. Use the `hash(a) -> <a, τ2> -> <a, τ1>` example for imperative style.
4.  **Code Implementation:** Identify all distinct code blocks in the slides. Categorize them (e.g., Data Structure, Hash Function, Operations, Symbol Interface, Scope Management). Prepare to explain each block's purpose and key lines.
5.  **Homework Analysis:** For each question in the homework PDF, determine the core concept being tested and derive the correct answer based on the lecture content.

### Phase 3: Note Generation
Create a single, polished `NOTE.md` file. Structure it as follows:
*   **Title & Overview:** Clear subject header and brief introduction.
*   **Section 1: Definitions & Concepts.** Present the fundamentals.
*   **Section 2: Functional vs. Imperative (Comparison).** Use a table or clear bullet points to highlight differences in definition, characteristics, data structure, and insertion.
*   **Section 3: Code Examples with Explanations.** This is the core technical detail. For each code block:
    *   Present the code in a fenced code block with the correct language tag (`c`).
    *   **Write an "Explanation" subsection** that breaks down the code's purpose, key variables/functions, and how it relates to the broader concept (e.g., "This implements the hash function for the imperative table...").
*   **Section 4: Homework Explanations.** For each question:
    *   State the question topic.
    *   Provide a concise explanation of the reasoning that leads to the answer.
    *   Clearly state the final answer.
*   **Section 5: Summary.** A concise recap or comparison table.

**Writing Style:** Be pedagogical, clear, and precise. Assume the reader is a computer science student familiar with basic data structures but new to compiler internals.

### Phase 4: Finalization
*   Use `filesystem-write_file` to create the `NOTE.md` file at the requested path (e.g., `/workspace/dumps/workspace/NOTE.md`).
*   Optionally, read back the first few lines using `filesystem-read_file` to verify successful creation.
*   Provide a final summary to the user outlining what was covered in the note.

## Key Triggers & Context Cues
*   User mentions: "symbol table", "compiler principles", "semantic analysis", "Chapter 5", "functional symbol table", "imperative symbol table", "scope", "insert operation", "homework", "lecture notes".
*   User provides or asks about files named `Compile.pptx`, `HW.pdf`, or similar.
*   User asks for a comparison or explanation of differences between two approaches.

## What NOT to Do
*   Do not invent compiler concepts not present in the source materials.
*   Do not provide code explanations that contradict the logic shown in the slides.
*   Do not skip explaining the homework questions if they are part of the request.
*   Do not create multiple note files unless explicitly requested. Consolidate all information into one structured `NOTE.md`.
