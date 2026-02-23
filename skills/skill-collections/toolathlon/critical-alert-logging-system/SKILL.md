---
name: critical-alert-logging-system
description: When the user needs to create automated alerting systems with severity-based logging for critical events. This skill writes structured log entries to Google Cloud Logging buckets, formats critical warning messages with relevant identifiers (student IDs, names), uses appropriate severity levels (CRITICAL, WARNING), and integrates with notification systems. Triggers include requests for automated alerting, critical event logging, system notifications, or creating audit trails for important events.
---
# Instructions

## Overview
This skill automates the detection of critical events (e.g., significant performance drops) and logs structured, severity-based alerts to a Google Cloud Logging bucket. It is designed for scenarios requiring automated alerting, audit trails, and integration with notification systems.

## Core Workflow
1.  **Data Acquisition & Analysis**
    *   Read the latest event/score data from a provided source (e.g., CSV file).
    *   Query historical data from a data warehouse (e.g., Google BigQuery) to establish a baseline or calculate trends.
    *   Perform comparative analysis (e.g., calculate percentage change, identify thresholds).

2.  **Threshold Evaluation & Categorization**
    *   Apply business logic to categorize events based on severity thresholds (e.g., >25% drop = WARNING, >45% drop = CRITICAL).
    *   Generate a structured list of all events exceeding the primary threshold.

3.  **Output Generation**
    *   **Primary Output:** Create a summary file (e.g., CSV) listing all identified events with relevant details (IDs, names, metrics, calculated drop).
    *   **Critical Alerting:** For events exceeding the highest severity threshold, write a structured log entry to the specified Google Cloud Logging bucket.

## Execution Steps

### 1. Initial Setup & Discovery
*   **Identify Input Source:** Locate and read the file containing the latest data (e.g., `latest_quiz_scores.csv`).
*   **Explore Data Warehouse:** Confirm the target dataset exists in BigQuery and list the relevant historical tables.
*   **Locate Logging Destination:** Identify the correct Google Cloud Logging bucket. Look for buckets with names matching the provided prefix (e.g., `exam_log`).

### 2. Historical Baseline Calculation
*   Construct a SQL query to calculate the historical average/metric for each entity (e.g., student) by combining all relevant source tables.
*   Execute the query in BigQuery. Handle potential data truncation in outputs by requesting all rows or batching queries if necessary.

### 3. Comparative Analysis
*   Join the latest data with the calculated historical averages.
*   Compute the performance metric (e.g., `(historical_avg - latest_score) / historical_avg`).
*   Filter results to isolate entities where the metric exceeds the defined warning threshold (e.g., >0.25).

### 4. Generate Primary Output File
*   Format the filtered results into a CSV file.
*   Include essential columns: identifier, name, historical average, latest value, calculated percentage.
*   Save the file to the specified path (e.g., `bad_student.csv`).

### 5. Critical Event Logging
*   From the primary results, filter again to isolate events exceeding the critical threshold (e.g., >0.45).
*   For each critical event:
    *   Format a clear, actionable log message. Include:
        *   **Severity Level:** `CRITICAL`
        *   **Event Description:** e.g., "CRITICAL ACADEMIC WARNING"
        *   **Identifying Information:** Name and ID of the entity.
        *   **Quantitative Detail:** The calculated drop percentage.
        *   **Action Required:** e.g., "Immediate counselor notification required."
    *   Write the formatted log entry to the identified Google Cloud Logging bucket using the `google-cloud-logging_write_log` tool.

### 6. Verification & Completion
*   Read back the generated output file to confirm it was saved correctly.
*   Provide a final summary confirming:
    *   The total number of events identified.
    *   The number of critical events logged.
    *   The location of the output file.
    *   The name of the log bucket used for alerts.

## Key Design Notes
*   **Modularity:** The analysis logic (BigQuery queries, percentage calculations) is separate from the alerting logic (log formatting and writing).
*   **Configurability:** Thresholds (25%, 45%) and log message templates should be considered parameters based on the user's request.
*   **Idempotency:** The skill does not consider log entries existing before execution, ensuring only new alerts are written for the current analysis run.
