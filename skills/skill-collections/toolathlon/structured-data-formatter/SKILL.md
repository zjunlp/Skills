---
name: structured-data-formatter
description: When the user needs to format extracted data according to specific templates or reference formats, particularly when working with markdown files, YAML structures, or predefined output formats. This skill reads reference format files, extracts raw data from various sources, and transforms it into consistently structured output following specified formatting rules. It's triggered by keywords like 'format should reference', 'write to file with format', 'following template', or when users provide reference files for output formatting.
---
# Instructions

## Primary Objective
Format extracted data into a specified output structure by reading a reference format file and applying its template to raw data.

## Core Workflow
1.  **Identify the Request:** The user will ask to write data to a file, explicitly mentioning a reference format file (e.g., "format should reference `format.md`") or a specific structure (e.g., "in YAML format").
2.  **Locate the Reference Format:**
    *   First, read the specified reference file (e.g., `format.md`) to understand the required output structure, headers, syntax (like YAML fences ` 