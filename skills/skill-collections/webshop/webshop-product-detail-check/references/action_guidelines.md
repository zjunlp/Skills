# WebShop Action Guidelines

## Action Format
The environment requires actions in one of two strict formats:
*   `search[keywords]`
*   `click[value]`

The `value` in a `click` action must be an **exact match** (case-insensitive) of an item in the list of available actions presented in the observation.

## Available Actions Identification
Available actions are typically indicated in the observation by:
1.  **Clickable Buttons/Text:** Items like `Back to Search`, `< Prev`, `Next >`, `Buy Now`, `Description`.
2.  **Product ASINs:** Alphanumeric codes starting with 'B0' (e.g., `B09C337K8S`) that are clickable in search results.
3.  **Page Navigation:** `Page X (Total results: Y)`.

## Invalid Actions
If the specified `value` for a `click` action is not present in the current observation, the action is invalid and the agent performs nothing, wasting a step.

## Action Strategy for This Skill
This skill (`webshop-product-detail-check`) operates on a product detail page. The primary actionable outputs are:
1.  `click[buy now]`: If the product is verified as suitable.
2.  `click[back to search]`: If the product is not suitable, to return to the search results.

Always verify the target string (e.g., "buy now", "back to search") is present in the observation before recommending the action.
