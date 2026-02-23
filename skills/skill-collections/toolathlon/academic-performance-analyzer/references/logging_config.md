# Cloud Logging Configuration

## Log Bucket Requirements
- **Name Pattern**: Must start with `exam_log` (e.g., `exam_log-af5170942c50`)
- **Access**: Requires write permissions for the service account
- **Retention**: Typically 30 days (configurable)

## Critical Alert Format

### Log Entry Structure
- **Log Name**: Use the exact bucket name (e.g., `exam_log-af5170942c50`)
- **Severity**: `CRITICAL`
- **Message Template**: 
  