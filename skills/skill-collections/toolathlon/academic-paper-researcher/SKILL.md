---
name: academic-paper-researcher
description: When the user needs to research academic papers and extract specific author information including full names, institutional affiliations (with department-level details when available), and Google Scholar profiles. This skill is triggered by requests involving academic paper analysis, conference research, author profiling, or when working with Excel sheets containing paper titles that need author metadata enrichment. It handles searching for papers, extracting information from PDFs, finding author profiles, and updating spreadsheet data.
---
# Instructions

## Goal
Fill an Excel spreadsheet with research data for a list of academic papers. For each paper, find:
1.  The full name of the first author.
2.  The complete institutional affiliation of the first author **exactly as it appears in the paper** (include all institutions if multiple are listed, specifying down to the department level when provided).
3.  The link to the first author's Google Scholar profile.

## Core Workflow

### 1. Locate and Inspect the Target Excel File
*   Use the `terminal-run_command` tool to find `.xlsx` or `.xls` files in the user's workspace (e.g., `/workspace/dumps/workspace`).
*   **Security Note:** Avoid using paths like `/dev/null` in commands as they may be outside allowed directories.
*   Once found, use `excel-get_workbook_metadata` and `excel-read_data_from_excel` to understand the file's structure: sheet names, headers (typically "Title", "First Author", "Affiliation", "Google Scholar Profile"), and the range of rows containing paper titles.

### 2. Research Each Paper
*   For each paper title in the Excel file, perform a parallel search using `local-web_search`. Craft queries like `"<Exact Paper Title>" paper first author`.
*   Analyze search results to identify:
    *   The first author's name.
    *   Potential sources for the full paper PDF (e.g., OpenReview, arXiv, conference proceedings).
    *   Initial clues for the author's Google Scholar profile.

### 3. Extract Precise Affiliation from Source PDFs
*   The most reliable source for the exact affiliation is the paper itself.
*   For each paper, locate and access the PDF using URLs found during the web search (e.g., `https://openreview.net/pdf?id=...`, `https://arxiv.org/pdf/...`).
*   Use `pdf-tools-read_pdf_pages` to read the first 1-2 pages of the PDF.
*   **Extraction Logic:** Scan the extracted text. The affiliation is typically listed on the first page, beneath the author list or in the footer/correspondence section. Look for patterns like:
    *   `*Equal contribution1Department of...`
    *   `1Gatsby Computational Neuroscience Unit, University College London`
    *   `Correspondence to: ...`
    *   Capture the **full, verbatim affiliation string** for the first author as it appears in the paper.

### 4. Find Google Scholar Profiles
*   For each identified first author, perform a targeted `local-web_search` (e.g., `"<Author Full Name>" Google Scholar profile` or `site:scholar.google.com "<Author Name>"`).
*   Validate the profile by checking if the author's listed publications include the target paper or are in a related field.
*   Extract the clean, direct URL to their Google Scholar citations page.

### 5. Compile and Write Data to Excel
*   Organize the collected data for all papers into a list of lists, where each sub-list contains `[first_author_name, affiliation, google_scholar_url]` for one paper.
*   Use `excel-write_data_to_excel` to write this data array into the appropriate columns ("First Author", "Affiliation", "Google Scholar Profile"), starting at the correct row (e.g., `B2`).
*   Verify the write operation by reading back the updated cell range with `excel-read_data_from_excel`.

### 6. Finalize and Report
*   Present a summary table to the user showing the completed entries.
*   Use `local-claim_done` to signal successful task completion.

## Key Principles & Error Handling
*   **Accuracy Over Speed:** Prioritize finding the original PDF to get the affiliation exactly right. Do not guess or paraphrase.
*   **Parallel Execution:** Use parallel tool calls for independent steps like searching for multiple papers or author profiles to improve efficiency.
*   **Fallback Strategies:** If a PDF cannot be accessed, use the most authoritative text snippet from search results (e.g., conference site, author's personal page) to infer the affiliation, and note any uncertainty.
*   **Validation:** Cross-check that the Google Scholar profile belongs to the correct person by verifying their institutional affiliation or paper titles.
