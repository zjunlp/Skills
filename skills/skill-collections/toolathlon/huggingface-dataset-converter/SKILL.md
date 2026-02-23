---
name: huggingface-dataset-converter
description: Downloads datasets from Hugging Face and converts them to specific formats (particularly for ML frameworks like Verl). Handles searching, analyzing structure, reading format specs, downloading, transforming schemas, and saving to target formats like Parquet.
---
# Instructions

## When to Use This Skill
Use this skill when the user requests to:
- Download a dataset from Hugging Face
- Convert a dataset to Parquet or another specific format
- Format a dataset for Verl, RLHF, or similar ML frameworks
- Find and use the "most downloaded" dataset for a given query
- Process Hugging Face datasets with specific format requirements

## Core Workflow

### 1. Understand the Request
- Identify the target dataset (by name, query, or "most downloaded" criteria)
- Clarify the output format requirements (check for format.json or user specifications)
- Determine the output filename and location

### 2. Search for Datasets (if needed)
- Use `huggingface-dataset_search` with appropriate query and sort parameters
- When user mentions "most downloaded," use `sort: "downloads"` and `limit: 10`
- Present search results to identify the best match

### 3. Read Format Specifications
- Check for format.json or similar specification files in the workspace
- Parse the JSON to understand required schema:
  - `data_source`: string identifier
  - `prompt`: list of message objects with role/content
  - `ability`: category string
  - `reward_model`: dict with style and ground_truth
  - `extra_info`: dict with additional fields

### 4. Get Dataset Details
- Use `huggingface-hub_repo_details` to examine dataset structure
- Note column names, data types, and sample content
- Verify the dataset contains required source fields

### 5. Download and Convert
- Use the bundled Python script `convert_dataset.py` for reliable conversion
- The script handles:
  - Loading dataset via Hugging Face datasets library
  - Mapping source columns to target format
  - Preserving all required schema elements
  - Saving to Parquet format with proper typing

### 6. Verify Output
- Check file creation and size
- Validate schema matches requirements
- Sample first few rows to confirm proper formatting
- Report completion with statistics

## Key Considerations

### Data Mapping
- Map `problem` → `prompt[0].content`
- Map `answer` → `reward_model.ground_truth`
- Map `solution` → `extra_info.solution`
- Add sequential `index` to `extra_info`
- Set constant fields (`data_source`, `ability`) as specified

### Error Handling
- If dataset doesn't contain required columns, inform user
- If format.json is missing or malformed, ask for clarification
- If download fails, check network connectivity and dataset accessibility

### Performance
- For large datasets, process in batches
- Use efficient data structures (pandas DataFrames)
- Compress output with Parquet for storage efficiency

## Common Triggers
- "download dataset from Hugging Face"
- "convert dataset to parquet"
- "format dataset for Verl/RLHF"
- "most downloaded dataset"
- "Hugging Face dataset with specific format"
