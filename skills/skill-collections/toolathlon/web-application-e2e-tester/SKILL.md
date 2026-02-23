---
name: web-application-e2e-tester
description: When the user needs to run end-to-end tests on a web application, especially for e-commerce checkout flows, and requires manual or automated testing of UI functionality like adding items to cart, applying coupons, form validation, and responsive design. This skill includes browser automation, test execution, and validation of test cases against a live application.
---
# Instructions

## Objective
Deploy a web application to a Kubernetes preview environment, expose it via NodePort, and execute a comprehensive suite of end-to-end (E2E) tests. The final deliverable is a filled test results report.

## Prerequisites
1.  **Target Application:** A Git repository containing the application source code, a Kubernetes deployment manifest (`preview.yaml`), and a test suite.
2.  **Kubernetes Cluster:** Access to a `kubectl` configured cluster.
3.  **Browser Automation:** Access to a browser automation tool (e.g., Playwright via `playwright_with_chunk` tools).
4.  **Test Report Template:** A markdown file (`test-results-report.md`) with a predefined table structure for results.

## Core Workflow

### Phase 1: Environment Setup & Deployment
1.  **Clone Repository:** Clone the target Git repository, specifying the required branch (e.g., `feature/pr-123`).
2.  **Inspect Structure:** Examine the repository to locate key files: the Kubernetes manifest (`preview.yaml`), the test script, the test report template, and the application's main HTML file.
3.  **Deploy to Kubernetes:**
    a. Create the namespace defined in `preview.yaml` (if it doesn't exist).
    b. Apply the `preview.yaml` manifest to create the Deployment and ClusterIP Service.
    c. Create a ConfigMap from the application's `index.html` file, matching the volume mount specified in the deployment.
    d. Create a **NodePort Service** to expose the application on a specific host port (e.g., `30123`) for long-term external access. Ensure the selector matches the deployment's pod labels.
4.  **Verify Deployment:** Confirm the Pod is `Running` and the application is accessible via `curl` or browser at the NodePort URL (e.g., `http://localhost:30123`).

### Phase 2: Test Execution & Validation
**Approach:** Manually execute each test case defined in the test script using browser automation, simulating user interactions.
1.  **Prepare Test Environment:** Clear browser `localStorage` before critical test sequences to ensure a clean state.
2.  **Execute Test Cases:** For each test function in the suite (e.g., `should load the homepage`, `should add products to cart`):
    a. Navigate to the application URL.
    b. Perform the described user actions (click buttons, fill forms, change viewport size).
    c. Use browser automation tools to assert expected outcomes (check element visibility, text content, form validation messages).
    d. Record the result as **PASS (✅)** or **FAIL (❌)**.
3.  **Key Test Categories to Cover:**
    *   **Page Load & Metadata:** Title, headers, PR information display.
    *   **Shopping Cart Logic:** Add/remove items, total price calculation.
    *   **Business Rules:** Free shipping thresholds, coupon application (`SAVE10`), tax calculation logic.
    *   **Checkout Process:** Form filling, submission, order success confirmation.
    *   **Form Validation:** HTML5 validation for required fields.
    *   **Responsive Design:** Visibility of core components at mobile (`375x667`) and tablet (`768x1024`) viewports.
    *   **Performance & API Health:** Page load speed (<3s), HTTP `200` status, correct `Content-Type` header.

### Phase 3: Reporting
1.  **Fill Test Report:** Update the `test-results-report.md` template. **Only** replace the empty cells in the `Result` column with `✅` or `❌` and update the `Summary Statistics` (Total, Passed, Failed) at the bottom. Do not modify any other text, structure, or test names.
2.  **Save Final Report:** Save the completed report to the workspace root as `filled-test-results-report.md`.

## Critical Notes & Edge Cases
*   **Tax Calculation Bug:** The example trajectory revealed a test failure where tax was calculated on the subtotal *before* discount, but the test expected it to be calculated *after* discount. Document such discrepancies clearly in the report.
*   **State Persistence:** The application may use `localStorage`. Use `playwright_with_chunk-browser_evaluate` to run `localStorage.clear()` between tests to ensure isolation.
*   **Form `novalidate` Attribute:** If a form has the `novalidate` attribute, HTML5 validation pop-ups will not block submission. Use `element.validationMessage` to check for validation messages instead.
*   **Service Configuration:** The initial `preview.yaml` likely creates a `ClusterIP` service. You must create a separate `NodePort` Service to expose the application on the host machine.
