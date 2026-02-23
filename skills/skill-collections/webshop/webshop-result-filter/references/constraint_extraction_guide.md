# Guide: Extracting Constraints from User Instructions

This guide helps parse natural language shopping instructions into structured filters for the `webshop-result-filter` skill.

## Common Constraint Patterns

### 1. Price Constraints
*   **Maximum Price:** `lower than X dollars`, `under X`, `less than X`, `below X`, `cheaper than X`
    *   Extract: `max_price = X`
*   **Price Range:** `between X and Y dollars`
    *   Extract: `min_price = X`, `max_price = Y`
*   **Exact Price:** `around X dollars`, `about X`
    *   Extract: `target_price = X` (consider a small tolerance, e.g., ±10%)

### 2. Feature/Attribute Constraints
*   **Direct Keywords:** `natural looking`, `long`, `clip-in`, `synthetic`, `human hair`
    *   Extract: Add keyword to `required_keywords` list.
*   **Implied Keywords:** `looks real` → implies `natural`.
*   **Negations:** `not synthetic` → requires checking for the *absence* of a keyword.

### 3. Quality/Rating Constraints
*   **Minimum Rating:** `highly rated`, `well reviewed`, `rated 4 stars or above`
    *   Extract: `min_rating = 4.0` (map phrases to numerical values).

## Parsing Examples

| User Instruction | Extracted Constraints |
| :--- | :--- |
| "i need a long clip-in hair extension which is natural looking, and price lower than 40.00 dollars" | `max_price: 40.00`<br>`required_keywords: ["natural", "long", "clip-in"]` |
| "find a red dress under $50 with good reviews" | `max_price: 50.00`<br>`required_keywords: ["red", "dress"]`<br>`min_rating: 4.0` (assuming "good" = 4+) |
| "wireless headphones between $100 and $200" | `min_price: 100.00`<br>`max_price: 200.00`<br>`required_keywords: ["wireless", "headphones"]` |

## Implementation Notes
*   Use case-insensitive matching for keywords.
*   Some constraints may be soft (preferences) vs. hard (requirements). This skill focuses on hard requirements.
*   When in doubt, be conservative. If a constraint is ambiguous, it's better to filter it out than to include a non-matching product.
