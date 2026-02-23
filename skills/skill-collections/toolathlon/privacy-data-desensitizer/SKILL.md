---
name: privacy-data-desensitizer
description: When the user needs to scan and desensitize sensitive information in documents across a workspace, particularly for privacy compliance or data protection purposes. This skill automatically identifies and replaces specific types of sensitive data (phone/fax numbers, social security numbers, email addresses, credit card numbers, IP addresses) with a uniform placeholder like '/hidden/' while preserving all other content. It handles multiple file formats (txt, csv, json, md, log) and creates desensitized copies with consistent naming conventions. Use this skill when users mention 'privacy desensitization', 'data anonymization', 'sensitive information removal', or when working with documents containing personal identifiable information (PII).
---
# Instructions

## Objective
Scan all documents in the user's workspace, identify specific types of sensitive information, and create desensitized copies where all sensitive data is replaced with `/hidden/`. The original files must remain unchanged.

## Sensitive Information Types
Only process and replace the following data types, even if they appear to be pseudo, mimic, or duplicated:
1.  **Phone/Fax numbers**: Includes formats like `(555) 123-4567`, `555-123-4567`, `555.123.4567`, and `1-800-XXX-XXXX` (both numeric and alphanumeric like `1-800-HOTEL-HELP`).
2.  **Social Security Numbers (SSN)**: Format `XXX-XX-XXXX`.
3.  **Email addresses**: Standard email format `user@domain.com`.
4.  **Credit card numbers**: 16-digit cards (with or without separators), Amex format (4-6-5 digits), and scientific notation (e.g., `4.11E+15`).
5.  **IP addresses**: IPv4 format `XXX.XXX.XXX.XXX`.

**Do not modify any information not on this list.**

## Core Workflow

### 1. Discover the Workspace
*   Use `filesystem-list_allowed_directories` to find the root workspace path.
*   Use `filesystem-directory_tree` or `filesystem-list_directory` on that path to get a list of all files.
*   **Exclude** the target output directory `desensitized_documents/` from the list of files to process.

### 2. Analyze File Contents
*   Read the content of all target files using `filesystem-read_multiple_files`.
*   Visually scan the content to confirm the presence of the specified sensitive data types.

### 3. Execute Desensitization
*   **Primary Method**: Run the bundled Python script `scripts/desensitize.py`. This is the preferred, reliable method.
    *   The script will process all files, create the `desensitized_documents/` directory, and save the modified copies.
*   **Alternative (Manual)**: If script execution fails, implement the desensitization logic directly using `local-python-execute` with the core logic defined in `references/patterns.md`.

### 4. Verify Results
*   Read a sample of the generated files from `desensitized_documents/` to ensure:
    *   All specified sensitive data is replaced with `/hidden/`.
    *   File structure, formatting, and non-sensitive content are preserved.
    *   Filenames follow the convention: `original_filename_desensitized.extension`.
*   Optionally, run a final count of replacements to provide a summary to the user.

## Output Requirements
*   Save all desensitized files in a subdirectory named `desensitized_documents/`.
*   Each output file must be named: `[original_basename]_desensitized.[original_extension]`.
*   Do not add any other files to the output directory.
*   Provide the user with a summary including the number of files processed and types of data replaced.

## Key Considerations
*   **Order of Operations**: Regex patterns must be applied in a specific order to avoid conflicts (e.g., processing `1-800` numbers before generic phone patterns).
*   **Precision**: Use word boundaries (`\b`) in regex patterns to avoid false positives (e.g., not matching dates as IP addresses).
*   **Preservation**: The replacement should occur exactly at the location of the sensitive data, without altering surrounding punctuation, spacing, or structure.
