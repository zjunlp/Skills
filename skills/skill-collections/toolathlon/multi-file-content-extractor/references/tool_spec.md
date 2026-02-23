# Tool Specification: filesystem-read_multiple_files

## Purpose
Reads and returns the text content of multiple files in a single operation.

## Input (`arguments`)
A JSON object with one key:
-   `paths` (array of strings): **Required**. The absolute paths to the files to be read.

## Output
A text string where the content of each file is concatenated. The format is:
