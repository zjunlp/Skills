# Test Execution Guide for Preview Deployments

## Test Structure Observed
The trajectory includes a Playwright test file `tests/checkout.spec.js` with:

1. **Environment Variable**: Uses `process.env.APP_URL` or defaults to `http://localhost:30123`.
2. **Test Categories**:
   - E-Commerce Checkout Flow (11 functional tests)
   - API Health Checks (2 API tests)
3. **Report Template**: A markdown file `test-results-report.md` with a table to fill.

## Running Tests Without Full Playwright Setup
If `npx` or Playwright is not available in the environment, simulate test execution via browser automation:

### Manual Test Steps (as in trajectory)
1. Navigate to `http://localhost:<nodePort>`.
2. For each test case:
   - Perform the required interactions (click buttons, fill forms).
   - Observe the results.
   - Record ✅ (pass) or ❌ (fail).

### Critical Test Cases & Verification Points

#### 1. Homepage Load
- Title contains "Frontend App - E-Commerce Demo".
- PR info banner shows branch name and PR number.

#### 2. Cart Operations
- Add products: Cart updates with correct items and total.
- Remove products: Item removed, total recalculated.
- Free shipping message: Updates based on total ($50 threshold).

#### 3. Checkout Process
- Fill all required fields (name, email, address, card).
- Submit form shows success message with Order ID (format `ORD-XXXXX`).

#### 4. Coupon & Tax
- Apply coupon `SAVE10`: Shows "Coupon applied: 10% off", discount calculated.
- **Tax Bug**: Test expects tax calculated *after* discount (10% of `subtotal - discount`). The actual code calculates tax on subtotal before discount. This test will fail.

#### 5. Responsive Design
- Resize viewport to mobile (375x667) and tablet (768x1024).
- Verify key elements remain visible.

#### 6. API Health
- `curl -I` returns HTTP 200.
- Headers include `Content-Type: text/html`.

#### 7. Performance
- Page loads in < 3 seconds (use `curl -w "%{time_total}"`).

## Filling the Test Report
The report template has strict formatting:
- Only replace empty cells with ✅ or ❌.
- Update the `Summary Statistics` at the bottom: Total, Passed, Failed.
- Do not modify any other text, headers, or table structure.

Example completed report:
