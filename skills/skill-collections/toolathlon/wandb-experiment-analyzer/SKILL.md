---
name: wandb-experiment-analyzer
description: When the user asks to analyze a Weights & Biases (wandb) project or experiment results, particularly when they need to identify the best-performing run based on validation metrics and find the optimal training step within that run. This skill handles querying wandb projects via GraphQL API, extracting run summaries and history data, comparing validation scores across experiments, analyzing step-level performance within the best run, and exporting results to structured formats like CSV files. It's triggered by requests involving wandb URLs, experiment comparison, validation performance analysis, or finding optimal training checkpoints.
---
# Instructions

## Overview
This skill analyzes wandb projects to identify the best-performing experiment based on validation metrics and finds the optimal training step within that experiment. It outputs results to a CSV file.

## Core Workflow

### 1. Parse User Request
- Extract the wandb project URL from the user's request (format: `https://wandb.ai/<entity>/<project>`)
- Identify the target validation metric (default: `val/test_score/` or similar validation score metrics)
- Determine output requirements (CSV file location, format)

### 2. Query Project Information
- Use the `wandb-query_wandb_tool` with GraphQL to fetch:
  - Project metadata (entity, project name, run count)
  - List of all runs with their summary metrics
  - Look for validation score metrics in summary data

### 3. Identify Best Experiment
- Parse summary metrics from all runs
- Extract validation scores (look for metrics like `val/test_score/`, `val_score`, `validation_score`)
- Compare scores across all runs to identify the best-performing experiment
- Note: Some runs may have null or missing validation scores

### 4. Fetch Detailed History for Best Run
- Query the history data for the best run using GraphQL
- Request sufficient samples to capture all validation score entries
- Handle large response data that may be truncated or saved to files

### 5. Analyze Step-Level Performance
- Parse the history data to extract validation scores and corresponding steps
- Filter out null/missing validation scores
- Sort by validation score in descending order
- Identify the step with the highest validation score

### 6. Generate Output
- Create a CSV file with columns: `best_experiment_name`, `best_step`, `best_val_score`
- Save to the workspace directory as specified by the user
- Provide a summary of findings to the user

## Key Considerations

### Validation Metric Identification
- The validation metric name may vary across projects
- Common patterns: `val/test_score/`, `val_score`, `validation/score`, `eval/score`
- Check summary metrics first to identify the correct metric name

### Data Handling
- Large history responses may be truncated; use the `local-search_overlong_tooloutput` tool to search within saved files
- Some steps may have null validation scores (only recorded at evaluation intervals)
- Ensure proper JSON parsing of nested data structures

### Error Handling
- Handle missing or inaccessible projects/runs gracefully
- Provide clear error messages if validation metrics cannot be found
- Handle authentication/permission issues with wandb API

## Tools Required
- `wandb-query_wandb_tool`: For GraphQL queries to wandb API
- `local-search_overlong_tooloutput`: For searching within large saved tool outputs
- `filesystem-write_file`: For creating CSV output files
- `filesystem-read_file`: For reading saved data files
- `terminal-run_command`: For running analysis scripts (when needed)

## Output Format
The skill produces a CSV file with the following structure:
