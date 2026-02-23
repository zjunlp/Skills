# E2E Test Execution Guide

## Test Case Mapping (Playwright Actions)
This guide maps common test descriptions to concrete browser automation actions.

| Test Step Description | Suggested Playwright Action |
| :--- | :--- |
| Navigate to application homepage | `playwright_with_chunk-browser_navigate` to the NodePort URL (e.g., `http://localhost:30123`) |
| Click the first "Add to Cart" button | `playwright_with_chunk-browser_click` on the element with role `button` and name `"Add to Cart"`, using `.first()` |
| Fill "Full Name" field with "John Doe" | `playwright_with_chunk-browser_type` into the textbox with name `"Full Name"` |
| Select "Express" shipping ($15.00) | `playwright_with_chunk-browser_click` on the combobox and select the option with text `"Express - $15.00"` |
| Apply coupon "SAVE10" | 1. `playwright_with_chunk-browser_type` into the "Coupon" textbox.<br>2. `playwright_with_chunk-browser_click` the "Apply" button. |
| Check for success message "Order Successful!" | Verify an element containing that text becomes `visible` after form submission. |
| Test mobile viewport (375x667) | `playwright_with_chunk-browser_resize` to `{width: 375, height: 667}` |
| Clear browser localStorage | `playwright_with_chunk-browser_evaluate` with function `() => { localStorage.clear(); }` |
| Get HTML5 validation message | `playwright_with_chunk-browser_evaluate` on an input element, running `(element) => element.validationMessage` |

## Assertion Patterns
*   **Text Content:** "Cart is empty" → Check if the cart container's text matches.
*   **Numerical Calculation:** "Total: $79.98" → Check if the total element contains the exact string after adding specific items.
*   **Element State:** "Submit button should be disabled" → Check the `disabled` property.
*   **Visibility:** "Success message should appear" → Check `display` style or visibility.

## Common Pitfalls
1.  **Dynamic Selectors:** Prefer `getByRole` or `getByText` over fragile CSS selectors that may change.
2.  **Timing:** Use appropriate waits (`networkidle`) for page loads after actions like form submission.
3.  **Test Independence:** Always clear `localStorage` and refresh the page (`navigate`) between tests that depend on cart state.
