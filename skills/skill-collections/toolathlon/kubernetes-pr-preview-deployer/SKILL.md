---
name: kubernetes-pr-preview-deployer
description: Deploys a specific Git branch to Kubernetes for preview/testing. Handles cloning repositories, creating namespaces, applying Kubernetes manifests (like preview.yaml), creating ConfigMaps from application files, and exposing the application via NodePort for local access. After deployment, it can run tests and generate a test report.
---
# Instructions

## Overview
This skill deploys a Git branch to a Kubernetes preview environment. It follows a sequence of operations observed in the trajectory: clone repository, inspect structure, create Kubernetes resources from manifests and application files, expose the service, run tests, and generate a report.

## Core Workflow

### 1. Repository Setup
- **Clone the target branch** from the specified Git repository into the workspace.
- **Inspect the repository structure** to locate key files: `preview.yaml` (Kubernetes manifest), `src/index.html` (application file for ConfigMap), and test files.

### 2. Kubernetes Deployment
- **Create the namespace** specified in `preview.yaml` (e.g., `pr-preview-123`). If it already exists, proceed.
- **Apply the main Kubernetes manifest** (`preview.yaml`) to create the Deployment and ClusterIP Service. Use the `--namespace` flag to target the correct namespace.
- **Create a ConfigMap** from the application's static file (e.g., `src/index.html`). The ConfigMap name must match the `configMap.name` referenced in the Deployment's `volumes` section (e.g., `frontend-app-pr123-html`).
- **Create a NodePort Service** to expose the application on a specific host port (e.g., `30123`). This service should select the same Pod labels as the Deployment. Use type `NodePort` and specify the `nodePort` field.

### 3. Verification & Testing
- **Verify the Pod is running** and the Service is accessible at `http://localhost:<nodePort>`.
- **Run the test suite** if requested. The trajectory shows a Playwright test script (`tests/checkout.spec.js`) that expects the `APP_URL` environment variable or defaults to `http://localhost:30123`.
- **Generate a test report** by executing the tests and filling a template report file (e.g., `test-results-report.md`). Update the result cells (✅/❌) and the summary statistics (Total, Passed, Failed). Save the completed report to the workspace root.

## Key Decisions & Edge Cases
- **Namespace Handling**: The namespace must be created before applying manifests that reference it. If creation fails because it exists, continue.
- **ConfigMap Creation**: Must be created *after* the namespace exists. Use `kubectl create configmap <name> --from-file=<key>=<path>`.
- **NodePort Service**: Create a separate Service of type `NodePort` in addition to any ClusterIP service in the manifest. This ensures long-term host access.
- **Test Execution**: If the test environment lacks `npx` or Playwright, you may need to run tests manually via browser automation (as shown in the trajectory) and record results.
- **Tax Calculation Bug**: Note that the example application has a bug: tax is calculated on the subtotal *before* discount, while the test expects it to be calculated *after* discount. The test report should reflect this failure.

## Required Tools
- `terminal-run_command` for shell operations (git, curl, file inspection).
- `filesystem-*` tools for reading/writing files and directory trees.
- `k8s-kubectl_*` tools for all Kubernetes operations.
- `playwright_with_chunk-browser_*` tools for browser-based testing if automated test runners are unavailable.

## Output
- A running Kubernetes application accessible at `http://localhost:<specified_port>`.
- A filled test report saved in the workspace root (e.g., `filled-test-results-report.md`).
