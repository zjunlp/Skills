---
name: csv-data-exporter
description: When the user needs to export analytical results or filtered datasets to CSV files for reporting or further processing. This skill formats data with appropriate headers, structures columns logically, handles numerical precision, sorts data meaningfully, and writes to specified file paths. Triggers include requests to save analysis results, create export files, generate reports in CSV format, or prepare data for external systems.
---
# Instructions

## Primary Objective
Export analytical results or filtered datasets to a CSV file with proper formatting, headers, and sorting.

## Core Workflow

### 1. Understand the Export Request
- Identify the source data to be exported (e.g., query results, filtered lists, analysis outputs)
- Clarify the desired output format: column names, data types, sorting order
- Determine the destination file path

### 2. Prepare Data for Export
- **Structure Data Logically**: Organize columns in a meaningful sequence (ID, name, metrics, timestamps)
- **Format Numerical Values**: Apply appropriate rounding/precision (typically 2 decimal places for percentages/float values)
- **Handle Missing Data**: Decide on representation for null/empty values (empty string or "N/A")
- **Sort Data Meaningfully**: Order by primary metric (descending for severity/drop percentages, ascending for IDs/names)

### 3. Generate CSV File
- Use `filesystem-write_file` to create the CSV
- Include header row with descriptive column names
- Ensure proper CSV formatting (comma-separated, proper escaping if needed)
- Write all data rows according to the prepared structure

### 4. Quality Verification
- Read back the created file using `filesystem-read_file` to verify content
- Check for correct row count and data integrity
- Confirm proper formatting and sorting

## Common Patterns from Trajectory

### Pattern A: Exporting Query Results with Calculations
When exporting results that involve calculations (like percentage drops):
1. Run the analytical query to get the dataset
2. Perform any necessary calculations/transformations in the query itself
3. Format numerical results (e.g., `ROUND(value, 2)`)
4. Sort by the key metric (e.g., `ORDER BY drop_percentage DESC`)
5. Export with columns: identifier, name, metrics, calculated values

### Pattern B: Multi-Source Data Integration
When combining data from multiple sources (e.g., CSV + database):
1. Read local CSV files using `filesystem-read_file`
2. Query database tables using appropriate services (e.g., `google-cloud-bigquery_run_query`)
3. Join/merge datasets using common identifiers
4. Apply filtering criteria (e.g., `WHERE drop_percentage > threshold`)
5. Export the integrated result set

### Pattern C: Conditional Logging with Exports
When export triggers additional actions (like critical alerts):
1. Identify threshold conditions within the exported data
2. For records meeting critical criteria, create log entries
3. Use appropriate logging services (e.g., `google-cloud-logging_write_log`)
4. Include identifying information (IDs, names) in log messages
5. Set appropriate severity levels (e.g., "CRITICAL" for urgent notifications)

## Best Practices

### File Naming
- Use descriptive names: `bad_student.csv`, `performance_report.csv`, `export_[date].csv`
- Place in appropriate directories (e.g., `/workspace/dumps/workspace/`)

### Data Presentation
- Include all relevant context columns (IDs, names, timestamps)
- Add calculated metrics as separate columns
- Use clear, human-readable column headers
- Maintain consistent formatting throughout

### Error Handling
- Verify file write operations succeeded
- Check that expected row counts match
- Validate data integrity after export

## Related Skills
- For data analysis before export: `data-analyzer`
- For database querying: `bigquery-query-executor`
- For log management: `logging-manager`
