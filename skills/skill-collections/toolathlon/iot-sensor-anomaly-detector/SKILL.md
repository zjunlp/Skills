---
name: iot-sensor-anomaly-detector
description: When the user needs to identify abnormal sensor readings by comparing real-time IoT sensor data against predefined operating parameter ranges. This skill queries time-series sensor data from BigQuery, loads parameter ranges from Excel configuration files, performs range validation, and generates anomaly reports in CSV format with timestamp, machine_id, sensor_type, reading, and normal_range fields. Triggers include IoT sensor data analysis, anomaly detection, operating parameter validation, factory monitoring, BigQuery sensor queries, Excel configuration file processing, and CSV report generation.
---
# Instructions

## Overview
This skill identifies sensor readings that fall outside predefined normal operating ranges. It queries time-series sensor data from BigQuery, compares readings against parameter ranges defined in an Excel configuration file, and generates a CSV anomaly report.

## Core Workflow

### 1. Initial Setup & Discovery
- **Locate the Excel configuration file**: Search for `machine_operating_parameters.xlsx` in the workspace.
- **Verify BigQuery dataset**: Confirm the `machine_operating` dataset exists.
- **Identify Cloud Storage bucket**: Find the bucket with prefix `iot_anomaly_reports` for report upload.

### 2. Data Acquisition
- **Read parameter ranges**: Load the "Operating Parameters" sheet from the Excel file. Extract `machine_id`, `sensor_type`, `min_value`, and `max_value` columns.
- **Query sensor data**: Execute a BigQuery query to retrieve sensor readings (`timestamp`, `machine_id`, `sensor_type`, `reading`) for the specified time range.

### 3. Data Processing & Anomaly Detection
- **Merge datasets**: Join sensor data with parameter ranges using `machine_id` and `sensor_type` as keys.
- **Identify anomalies**: Flag readings where `reading < min_value` OR `reading > max_value`.
- **Format output**: Create `normal_range` column as "min - max" string.

### 4. Report Generation & Delivery
- **Create CSV report**: Generate `anomaly_report.csv` with columns: `timestamp`, `machine_id`, `sensor_type`, `reading`, `normal_range`.
- **Save locally**: Store the report in the workspace.
- **Upload to Cloud Storage**: Transfer the report to the identified `iot_anomaly_reports` bucket.

### 5. Cleanup
- Remove any temporary files or tables created during processing.

## Key Requirements
- **Timestamp format**: Preserve the exact timestamp format from BigQuery (e.g., `2025-08-19 11:33:59.878906+00`).
- **Error handling**: Gracefully handle missing sheets in Excel, unmatched sensor-parameter pairs, and data type mismatches.
- **Efficiency**: Process potentially large datasets efficiently using appropriate data structures.

## Common Variations
- **Different time ranges**: The skill should adapt to user-specified start/end times.
- **Alternative file locations**: The Excel file may be in different directories.
- **Multiple sensor types**: The skill handles various sensor types defined in the configuration.

## Output
The primary output is `anomaly_report.csv` containing all readings outside normal ranges, sorted by timestamp.
