# Evaluation Logic Reference

## Parsing Rules for WebShop Observations
The WebShop environment presents data in a specific text format. Key parsing assumptions:
1.  The observation is split by the `[SEP]` delimiter.
2.  The user's instruction follows the token `Instruction:`.
3.  Product listings typically follow this pattern on a search page:
    `[ASIN] [Title] [Price]`
    *   **ASIN:** A 10-character alphanumeric identifier, usually starting with 'B' (e.g., `B09NYFDNVX`).
    *   **Title:** The full product name.
    *   **Price:** A string starting with `$` followed by a number (e.g., `$17.99`).

## Requirement Extraction Heuristics
- **Price Limit:** Look for phrases like "lower than X dollars", "under X", "less than X". Extract the numerical value `X`.
- **Feature Keywords:** Identify nouns and adjectives describing the desired product features. Common stop words (I, need, some, that, also, and) are filtered out.
    *   *Example:* From "teeth whitening that also freshens breath", keywords become: `['teeth', 'whitening', 'freshens', 'breath']`.

## Product Scoring & Selection
A product is considered **suitable** if it passes two filters:
1.  **Price Filter:** Product price <= User's price limit.
2.  **Keyword Filter:** The product title contains all extracted feature keywords (case-insensitive).

If multiple products are suitable, they are **sorted by price (ascending)** to prioritize budget-friendly options. The top result is selected for the next action (`click[ASIN]`).

## Fallback Behavior
If the script finds **no suitable products** on the current page, the skill should guide the agent to:
1.  Consider refining the search query using the `search[new keywords]` action.
2.  Or, if available, navigate to the next page of results (`click[Next >]`).
