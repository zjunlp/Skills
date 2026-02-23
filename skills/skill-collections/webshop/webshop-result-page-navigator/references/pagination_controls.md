# Reference: WebShop Pagination Controls

## Context
This document details the structure and identifiers for pagination elements within the WebShop environment, as observed in the provided trajectory.

## Observed Pagination Format
Pagination controls appear within the observation text, following the page information header. The format is consistent:
`[SEP] < Prev [SEP] Next > [SEP]`

## Control Identification
*   **"Next >"**: The control to advance to the next page of results. The exact clickable value is `next >`.
*   **"< Prev"**: The control to return to the previous page of results. The exact clickable value is `< prev`.

## Action Syntax
The correct action format to use with these controls is:
*   `click[next >]`
*   `click[< prev]`

## Important Notes
1.  These controls are only present when there is a previous or next page to navigate to (e.g., not on Page 1 of 1).
2.  The skill should only be invoked after a conscious evaluation of the current page's contents.
3.  The primary agent logic is responsible for parsing the `Observation` to confirm the presence of these controls before the skill executes.
