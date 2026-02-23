# Google Cloud Logging Guidelines for Critical Alerting

## Log Entry Structure
When writing critical alerts to Google Cloud Logging, structure your log entries for maximum utility:

### Essential Components
1.  **Severity Level:** Use `CRITICAL` for events requiring immediate human intervention.
2.  **Clear Identification:** Include unique identifiers (Student ID, User ID, Transaction ID).
3.  **Human-Readable Message:** Describe what happened, why it's critical, and required action.
4.  **Quantitative Context:** Include relevant metrics (percentage drop, absolute values, thresholds).

### Message Template
