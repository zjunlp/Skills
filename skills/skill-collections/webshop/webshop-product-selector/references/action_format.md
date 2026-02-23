# WebShop Action Format Reference

## Valid Action Types
The WebShop environment accepts two primary action formats:

1.  `search[keywords]`
    *   Initiates a product search.
    *   `keywords`: A string of search terms (e.g., `"double horn bluetooth speaker"`).

2.  `click[value]`
    *   Clicks on an interactive element on the page.
    *   `value`: Must be an **exact match** of a value present in the current observation's list of available actions.
    *   Common `value` types:
        *   **Product IDs**: Typically a 10-character alphanumeric string (e.g., `b093c1thvp`). Appear in search results.
        *   **UI Elements**: Phrases like `"buy now"`, `"description"`, `"back to search"`, `"next >"`.

## Critical Rule for `click[value]`
The `value` for a click action **must be copied verbatim** from the observation text. For product IDs, the observation often shows them in uppercase (e.g., `B093C1THVP`), but the clickable action typically expects the lowercase version (e.g., `b093c1thvp`). Always check the format in the trajectory or environment feedback.

## Observation Structure
Observations are text blocks with `[SEP]` delimiters. Key sections include:
*   **Instruction**: The user's original request.
*   **Page Info**: e.g., `Page 1 (Total results: 50)`.
*   **Product Listings**: Pattern is usually `[SEP] ProductID [SEP] Title [SEP] Price [SEP]`.
*   **Clickable Actions**: A list of available `click[value]` targets at the end of the observation.

## Skill-Specific Guidance
The `webshop-product-selector` skill outputs a `click[product_id]` action. Ensure the selected `product_id` is present in the current observation before executing.
