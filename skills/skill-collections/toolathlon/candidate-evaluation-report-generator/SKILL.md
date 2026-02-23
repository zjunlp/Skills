---
name: candidate-evaluation-report-generator
description: Analyzes candidate resumes and interviewer evaluations against specific job requirements, calculates match percentages, and generates standardized Word assessment reports for each candidate, identifying the best match.
---
# Skill: Candidate Evaluation Report Generator

## Purpose
You are an HR automation assistant. Your task is to process candidate application materials, evaluate them against strict job criteria, and produce a set of uniformly formatted assessment documents.

## Core Workflow
Follow these steps **in order**:

1.  **Gather Inputs:** Read the required files from the workspace.
2.  **Analyze Candidates:** Extract data and evaluate each candidate against the provided requirements.
3.  **Generate Reports:** Create a separate, formatted Word document for each candidate.
4.  **Output Recommendation:** Create a final file naming the single best candidate.

## Detailed Instructions

### Phase 1: Read and Parse Input Files
First, locate and read the following critical files. Their names may vary slightly; use common sense to identify them (e.g., `*resume*.docx`, `*eval*.docx`, `*format*.md`, `*requirement*.txt`).
-   **Formatting Guide (`Format.md`):** Contains the exact template, style specifications (fonts, sizes, colors), and naming convention for the output reports. **You must follow this precisely.**
-   **Candidate Resumes (`Candidates_resumes.docx` or similar):** A Word document containing resumes for all candidates. Parse it to extract structured data for each individual.
-   **Interviewer Evaluations (`Interviewer_evaluation.docx` or similar):** A Word document containing the interview feedback for each candidate. Parse it to extract structured evaluations.
-   **Job Requirements (`Requirement.txt` or similar):** A text file listing the **strict criteria** (e.g., "4+ years experience", "FAANG required"). You will use this to score candidates.

### Phase 2: Evaluate Each Candidate
For each candidate identified in the resume and evaluation documents:
1.  **Extract Key Data:** Combine information from both their resume and their evaluation.
2.  **Apply Requirements:** Systematically check the candidate's profile against every criterion listed in `Requirement.txt`.
3.  **Calculate Score:** Determine the percentage of requirements the candidate meets. (e.g., 7 out of 10 criteria = 70%).
4.  **Determine Recommendation:** Based on the analysis, identify if this candidate is the **single best match** according to the requirements. Only one candidate can be "Recommended" in the final output.

### Phase 3: Generate Formatted Word Reports
Create a new Word document for **every candidate**. Use the naming convention specified in `Format.md` (e.g., `Interview_Assessment_Report_<CandidateName>.docx`).

For each document, you **must**:
1.  **Apply Exact Formatting:** Follow all style rules in `Format.md` (headings, fonts, alignment, colors).
2.  **Populate Structure:** Include all specified sections:
    -   **Main Title:** "Candidate Interview Assessment Report" with correct style.
    -   **Candidate Name.**
    -   **"Candidate Information:"** section with data from their resume.
    -   **"Interviewer Evaluation:"** section with data from their evaluation.
    -   **"Evaluation Conclusion:"** section with a 2x2 table.
3.  **Format the Table:** The conclusion table must have:
    -   Gridline borders.
    -   A header row with background color `#D9D9D9`.
    -   A data row with background color `#DAEEF3`.
    -   Cell alignment: Horizontal centered, Vertical top.
    -   **Content:**
        -   Row 1, Col 1: "Overall Assessment"
        -   Row 1, Col 2: "Recommendation"
        -   Row 2, Col 1: The calculated percentage (e.g., "70%"). **Do not add reasoning.**
        -   Row 2, Col 2: Either "Recommended" (for the single best candidate) or "Not recommended". **Do not add reasoning.**

### Phase 4: Create Final Recommendation File
After all reports are generated, create a plain text file named `recommend.txt` in the workspace.
-   Its **only** content must be the full name of the best candidate (e.g., `John Smith`).
-   Do not include any other text, explanations, or formatting.

## Critical Rules
-   **Strict Formatting Adherence:** The visual format of the output documents is non-negotiable. Match `Format.md` exactly.
-   **One Best Candidate:** Only one candidate receives "Recommended" in their table and is named in `recommend.txt`. If multiple candidates have the same top score, use tie-breakers implicit in the requirements (e.g., more FAANG companies, lower salary).
-   **No Extra Commentary:** The table cells and `recommend.txt` file must contain only the specified data, no sentences or justifications.
-   **Complete Set:** You must generate a report for every candidate found in the source documents.
