# Kubernetes Annotation Schema

## Required Annotation
This skill expects deployments to have the following annotation:

### `app-version-release-date`
- **Purpose**: Stores the release date of the application version running in the deployment
- **Format**: ISO 8601 timestamp (UTC)
- **Example**: `2025-08-30T09:45:14Z`
- **Location**: Deployment metadata annotations

## Annotation Extraction
The annotation can be found in the deployment description:
