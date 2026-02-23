---
name: latex-table-generator
description: Creates formatted LaTeX tables from structured data. It parses data, applies sorting and formatting rules, generates table headers and structure following provided templates, and saves the output as .tex files.
---
# Instructions

## Primary Objective
Generate a LaTeX table from provided data sources (e.g., PDFs, text) and save it as a `.tex` file. The table must follow a user-specified format and sorting order.

## Core Workflow
1.  **Identify Inputs & Requirements:** The user will specify the data source, the desired table columns, and the sorting rule (e.g., "descending order of FID").
2.  **Locate & Parse Source Data:** Use appropriate tools (`filesystem-list_directory`, `pdf-tools-read_pdf_pages`, `pdf-tools-search_pdf_content`) to find and extract the required numerical and categorical data (e.g., Model Name, Method Category, Parameters, FID, Inception Score).
3.  **Extract & Structure Data:** Parse the extracted text to identify key-value pairs for the table. Focus on finding the *best* result per source (e.g., "the model with the lowest FID result in the paper").
4.  **Apply Formatting & Sorting:** Structure the data into rows. Apply the user-specified sorting rule to the final dataset.
5.  **Generate LaTeX Code:** Use the provided template (e.g., `format.tex`) to structure the final table. Insert the sorted data rows into the template's `tabular` environment.
6.  **Write Output File:** Use `filesystem-write_file` to save the generated LaTeX source code to the specified `.tex` file (e.g., `survey.tex`).
7.  **Verify & Conclude:** Optionally, read the created file to confirm its contents and present a summary to the user.

## Key Decisions & Rules
*   **Model Selection:** When a source (e.g., a paper) contains multiple models, select the one that best meets the user's criterion (e.g., lowest FID).
*   **Method Categorization:** Map the model's method to one of the standard categories: `VAE`, `GAN`, `Diffusion`, `Flow-based`, or `AR` (Autoregressive). Use context from the paper to make this determination.
*   **Data Validation:** Cross-reference extracted numbers (like Parameters, FID) within the paper (e.g., check text, tables, captions) to ensure accuracy before populating the final table.
*   **Template Adherence:** Strictly follow the column structure and LaTeX styling (e.g., `\\toprule`, `\\midrule`, `\\bottomrule` from `booktabs`) defined in the user's template file.

## Triggers
Initiate this skill when the user request contains keywords or phrases such as:
*   "create LaTeX table"
*   "save results in .tex"
*   "format table"
*   "tabular environment"
*   Mentions of `.tex` file extensions or LaTeX formatting.
*   Requests to summarize comparative data into a structured, sortable table.
