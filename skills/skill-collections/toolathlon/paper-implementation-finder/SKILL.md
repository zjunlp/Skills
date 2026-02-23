---
name: paper-implementation-finder
description: Finds the most popular unofficial GitHub implementation for a given academic paper (PDF) and updates a JSON file with the repository URL.
---
# Instructions

## Objective
When a user provides a PDF of an academic paper (typically without an official code repository) and a `result.json` file, find the most popular (by star count) unofficial implementation on GitHub and update the JSON file with the repository URL.

## Core Workflow
1.  **Extract Paper Information:**
    *   Use `pdf-tools-get_pdf_info` to get basic metadata (e.g., page count).
    *   Use `pdf-tools-read_pdf_pages` to read the first few pages (e.g., 1-3) to identify the paper's **title**, **authors**, and **key technical terms** (e.g., "Mixture-of-Depths").

2.  **Formulate GitHub Search Query:**
    *   Construct a search query using the most distinctive terms from the paper's title and abstract. Prioritize the paper's title or a unique acronym (e.g., "Mixture-of-Depths transformer").
    *   **Avoid overly generic terms** that will return irrelevant results.

3.  **Search and Identify the Target Repository:**
    *   Use `github-search_repositories` with the formulated query.
    *   Analyze the search results (`items`).
    *   **Filtering Logic:** The target repository must be an **unofficial implementation** of the specific paper provided. Check repository `description` and `topics` for clear mentions of the paper.
    *   **Selection Criteria:** Among the filtered repositories, select the one with the highest `stargazers_count`. This is the "most popular" implementation.
    *   **Edge Case:** If the highest-starred repo is for a *different* paper (e.g., mentions the technique only as a feature), proceed to the next eligible repo.

4.  **Update the Result File:**
    *   Read the existing `result.json` file using `filesystem-read_file` to understand its structure.
    *   Write the selected repository's `html_url` into the `result.json` file using `filesystem-write_file`. Maintain the existing JSON structure (e.g., `{"URL": "https://github.com/..."}`).

5.  **Verification and Completion:**
    *   Optionally, read the updated `result.json` to confirm the write was successful.
    *   Provide a final summary to the user, stating the paper identified, the repository selected, and its star count.
    *   Claim completion with `local-claim_done`.

## Key Considerations
*   **Accuracy over Speed:** It is more important to find the *correct* unofficial implementation than the first one listed. Carefully read repository descriptions.
*   **Clear Communication:** Explain your reasoning to the user when selecting between multiple candidate repositories.
*   **File Paths:** The skill assumes the PDF and `result.json` are provided in a known workspace directory (e.g., `/workspace/dumps/workspace/`). Use the paths provided in the user's request or context.
