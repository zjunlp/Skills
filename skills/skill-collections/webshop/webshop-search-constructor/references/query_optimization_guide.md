# Query Optimization Guide for E-commerce Search

## Principles
Effective search queries on platforms like WebShop balance specificity with breadth.

1.  **Order Matters:** Place the most critical, distinguishing term first. Typically, this is the product category.
2.  **Conciseness:** Avoid stop words (the, a, an, for). Use commas to separate key concepts, which many search engines interpret as "AND" operators.
3.  **Natural Language Filters:** For price, use phrases like "less than $X" which are commonly parsed by search engines. Avoid complex symbols like `<`.
4.  **Attribute Prioritization:** Include quantifiable attributes (e.g., "10 slots") before qualitative ones (e.g., "easy to clean").

## Example Transformations
*   **Criteria:** `product='laptop', attributes=['16GB RAM', 'lightweight'], price_limit=1200`
*   **Optimized Query:** `laptop, 16GB RAM, lightweight, less than $1200`

*   **Criteria:** `product='coffee maker', attributes=['programmable'], price_limit=80.00`
*   **Optimized Query:** `coffee maker, programmable, less than $80.00`

## Platform-Specific Notes
*   The `search[query]` action format is specific to the WebShop environment used in the trajectory.
*   The comma-separated format (`term1, term2, term3`) is a robust pattern that works across many e-commerce search engines.
*   Always verify the available actions in the observation. If `search` is not available, this skill should not be invoked.
