---
name: test-results-reporter
description: When the user needs to execute test suites and generate formatted test reports with pass/fail status, summary statistics, and detailed results. This skill involves reading test specifications, running tests, evaluating outcomes, and creating structured markdown reports with proper formatting and summary metrics.
---
# Instructions

## Overview
This skill executes a test suite against a deployed application and generates a formatted markdown report. It follows a sequence of: 1) verifying the test environment and specifications, 2) systematically running each test, 3) evaluating the results against expected outcomes, and 4) compiling a final report with pass/fail status and summary statistics.

## Core Workflow

### 1. Initialize & Verify Environment
*   **Locate Test Files:** Identify the test specification file (e.g., `checkout.spec.js`) and the report template file (e.g., `test-results-report.md`). Read their contents to understand the test cases and the required report structure.
*   **Verify Deployment:** Ensure the target application (e.g., a web service) is accessible at the specified URL (e.g., `http://localhost:30123`). Use a simple HTTP request to confirm connectivity.
*   **Prepare Workspace:** Clear any persisted state from previous test runs (e.g., browser `localStorage`) to ensure test isolation.

### 2. Execute Test Suite
*   **Follow Test Specification:** Execute tests in the order they are defined. For web applications, use browser automation tools (like Playwright) to interact with the UI.
*   **Test Categories:** Typically handle two main categories:
    *   **Functional/UI Tests:** Simulate user actions (clicks, form input, navigation) and assert UI state changes.
    *   **API Health Checks:** Verify HTTP status codes and response headers.
*   **Record Outcomes:** For each test case, determine a **Pass (✅)** or **Fail (❌)** result based on whether the actual outcome matches the expected behavior defined in the test spec.

### 3. Generate the Test Report
*   **Populate Template:** Fill the pre-defined markdown report template. **Only** update the `Result` column cells with ✅ or ❌ and the `Summary Statistics` section (`Total Tests`, `Passed`, `Failed`). Do not modify any other text, structure, or headings.
*   **Save Output:** Write the completed report to the specified location in the workspace (e.g., `filled-test-results-report.md`).

## Key Decision Points & Error Handling
*   **Test Failure Interpretation:** A test fails if the actual result deviates from the expected assertion in the specification, even if the application is functionally working (e.g., a business logic implementation differs from the test's expectation).
*   **Environment Issues:** If the application is not accessible, pause test execution and report the deployment issue before proceeding.
*   **Strict Template Adherence:** The report template is immutable. Only fill in the designated result cells and summary numbers.

## Final Verification
*   Confirm the generated report file exists and is correctly formatted.
*   Optionally, provide a brief verbal summary of the test run (total tests, pass/fail count, and note any critical failures).
