---
name: academic-performance-analyzer
description: When the user needs to monitor student academic performance by comparing current test scores against historical averages to identify significant performance declines. This skill reads CSV files containing latest scores, queries historical data from BigQuery tables, calculates percentage drops, identifies at-risk students based on configurable thresholds (e.g., >25% drop), and generates comprehensive reports. Triggers include requests for academic warning systems, student performance analysis, score trend monitoring, or identifying students needing intervention.
---
# Academic Performance Analyzer

## Overview
This skill automates the detection of significant declines in student academic performance by comparing the latest test scores against their historical averages. It identifies students requiring academic intervention and can trigger immediate counselor notifications for critical cases.

## Core Workflow
1. **Input Processing**: Read the latest student scores from a CSV file.
2. **Historical Analysis**: Query and aggregate historical test scores from multiple BigQuery tables.
3. **Performance Calculation**: Compute the percentage drop between historical average and latest score for each student.
4. **Threshold Filtering**: Identify students exceeding warning thresholds (configurable).
5. **Output Generation**: Create a CSV report of at-risk students.
6. **Critical Alerting**: Write critical warning logs for students exceeding severe thresholds.

## Required Inputs
- **Latest Scores CSV**: Path to a CSV file with columns: `student_id`, `name`, `score`
- **BigQuery Dataset**: Dataset containing historical score tables (e.g., `scores_2501`, `scores_2502`, etc.)
- **Log Bucket**: Cloud Logging bucket with name prefixed with `exam_log` for critical alerts

## Configuration Parameters
- `WARNING_THRESHOLD`: Percentage drop to trigger inclusion in report (default: 25%)
- `CRITICAL_THRESHOLD`: Percentage drop to trigger critical alerts (default: 45%)
- `OUTPUT_CSV`: Path for the generated report (default: `bad_student.csv`)

## Key Instructions

### 1. Validate Inputs
- Confirm the latest scores CSV file exists and has correct format
- Verify BigQuery dataset exists and contains expected score tables
- Check for available log buckets with `exam_log` prefix

### 2. Query Historical Data
- Use a UNION ALL query to combine all historical score tables
- Calculate average score per student across all historical tests
- Handle potential data truncation by querying in batches if needed

### 3. Perform Analysis
- Join latest scores with historical averages
- Calculate drop percentage: `((avg_score - latest_score) / avg_score) * 100`
- Filter students where drop percentage exceeds `WARNING_THRESHOLD`
- Sort results by drop percentage (descending)

### 4. Generate Outputs
- Create CSV with columns: `student_id`, `name`, `avg_score`, `latest_score`, `drop_percentage`
- Save to specified output path

### 5. Handle Critical Cases
- For students exceeding `CRITICAL_THRESHOLD`:
  - Write CRITICAL severity log to the `exam_log` bucket
  - Include student name and ID in log message
  - Format: "CRITICAL ACADEMIC WARNING: Student [Name] (ID: [ID]) has experienced a severe score drop of [X]% compared to their historical average. Immediate counselor notification required."

## Error Handling
- Gracefully handle missing tables or files
- Validate data types and ranges
- Log errors appropriately without exposing sensitive information

## Notes
- Historical tables may have prefixes (e.g., `scores_2501` instead of just `2501`)
- Output CSV should be sorted by severity (highest drop percentage first)
- Only consider logs written during current execution (ignore pre-existing entries)
