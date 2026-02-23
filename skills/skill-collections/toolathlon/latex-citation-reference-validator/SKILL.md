---
name: latex-citation-reference-validator
description: Checks and fixes LaTeX paper drafts for citation and reference issues across multiple .tex files with a .bib bibliography file. Verifies that all \cite{}, \label{}, \ref{}, and \autoref{} commands work properly and cross-reference correctly across all files. Identifies broken, misnamed, or incorrect citations and references, then fixes them to use the correct names from the .bib file and LaTeX labels.
---
# Instructions

## Overview
This skill validates and repairs LaTeX citation and reference integrity in a multi-file project. It systematically scans all `.tex` files and the `.bib` file to identify:
1.  **Broken Citations:** `\cite{}` commands referencing non-existent `.bib` entries.
2.  **Empty Citations/References:** `\cite{}`, `\ref{}`, or `\autoref{}` with empty braces.
3.  **Broken References:** `\ref{}` or `\autoref{}` commands pointing to non-existent `\label{}`s.
4.  **Incorrect References:** References to labels that exist but are likely typos (e.g., `fig:call-api-v0` vs. `fig:call-api`).

The core logic is implemented in the bundled Python script `validate_and_fix.py`. You should primarily execute this script, which provides a deterministic, repeatable analysis and repair process.

## Step-by-Step Procedure

1.  **Initial Exploration:** First, understand the workspace structure. List the allowed directories and examine the directory tree to locate all `.tex` and `.bib` files.

2.  **Run the Validation Script:** Execute the primary script `/workspace/latex-citation-reference-validator/scripts/validate_and_fix.py`. This script will:
    *   Recursively find all `.tex` files in the workspace.
    *   Read the `.bib` file to extract all citation keys.
    *   Parse all `.tex` files for `\cite{}`, `\label{}`, `\ref{}`, and `\autoref{}` commands.
    *   Perform a comprehensive analysis, identifying all four categories of issues listed above.
    *   **Automatically fix** the identified issues by editing the source `.tex` files in place.
    *   Generate a detailed summary report of all changes made.

3.  **Review and Confirm:** After the script runs, review its output summary. The summary will detail every file modified and the specific changes applied (e.g., "Fixed empty citation `\citep{}` -> `\citep{brown2020language}`"). You should verify that the fixes are correct and appropriate for the context of the paper.

4.  **Manual Verification (Optional):** For complex projects or if you suspect edge cases, you may manually spot-check a few fixed locations in the `.tex` files to ensure the corrections are contextually accurate (e.g., the added citation key `brown2020language` is indeed the correct paper for the phrase "learn in-context").

## Key Decisions & Heuristics
*   **Empty Citation Fixing:** The script uses a heuristic lookup table (`CITATION_FIX_MAP` in the script) to map common phrases to probable citation keys (e.g., "learn in-context" -> `brown2020language`, "SimCSE retriever" -> `gao2021simcse`). **You must review these automatic fixes for accuracy.** If the heuristic fails, you will need to manually determine and apply the correct citation key.
*   **Broken Reference Resolution:** The script identifies the closest matching existing label for a broken reference using string similarity. For example, `fig:call-api-v0` will be corrected to `fig:call-api`. **You must verify** that the suggested correction is semantically correct (i.e., references the intended figure/table/section).
*   **Scope:** The script operates on all `.tex` files within the `/workspace` directory. Ensure your target paper files are located there.

## Triggers
Use this skill when the user request involves:
*   "Check my LaTeX citations and references."
*   "Fix broken `\ref{}` commands in my paper."
*   "Validate that all `\cite{}` commands match my `.bib` file."
*   "Help me find missing labels or citations."
*   Working with file extensions `.tex` and `.bib`.

## Bundled Resources
*   `scripts/validate_and_fix.py`: The main analysis and repair script. Run this.
*   `references/common_citations.md`: A reference list of common NLP/ML paper citation keys and the contexts they typically appear in. Consult this if you need to manually determine a correct citation key not covered by the script's heuristics.
