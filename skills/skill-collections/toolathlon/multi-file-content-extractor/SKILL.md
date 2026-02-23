---
name: multi-file-content-extractor
description: When the user references multiple files in their query or when the task requires reading content from several files in the workspace, this skill loads and extracts text content from specified file paths. It handles reading multiple files simultaneously, returning their contents for further processing. Triggers include phrases like 'located in the workspace', 'from the file', or when multiple file paths are mentioned.
---
# Instructions

## Primary Function
Use this skill when the user's request explicitly mentions or implies the need to read content from multiple files in the workspace. The core action is to call the `filesystem-read_multiple_files` tool with an array of file paths.

## Trigger Detection
Activate this skill when you detect any of the following in the user's query:
-   Explicit mention of multiple file paths (e.g., "/workspace/file1.md", "/workspace/file2.json").
-   References to files "in the workspace" or "located in the workspace".
-   Phrases like "from the file" or "based on the document" when multiple documents are involved.
-   The task logically requires synthesizing information from more than one source file.

## Execution Steps
1.  **Parse the Request:** Identify all file paths mentioned or implied by the user's query. Paths are often absolute (e.g., `/workspace/dumps/workspace/recommendation.md`).
2.  **Validate Paths:** Ensure the paths are plausible and within a workspace context (e.g., starting with `/workspace`). If a path is ambiguous, use your best judgment to construct it based on the query context.
3.  **Batch Read:** Call the `filesystem-read_multiple_files` tool once, passing a JSON array of all identified file paths to the `paths` argument.
    