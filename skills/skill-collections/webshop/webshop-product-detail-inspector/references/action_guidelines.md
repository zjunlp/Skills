# Action Guidelines for WebShop Navigation

## Valid Action Patterns
The environment only accepts actions in these formats:
*   `search[keywords]`
*   `click[value]`

## Click Action Rules
1.  **Exact Match Required:** The `value` inside `click[]` must be an exact, case-sensitive string match for a clickable element shown in the observation.
2.  **Common Clickable Elements:**
    *   Product IDs (e.g., `B08VJB28BL`)
    *   Option buttons (e.g., `10 slots`, `Large`, `Blue`)
    *   Navigation (e.g., `Back to Search`, `< Prev`, `Next >`)
    *   Page actions (e.g., `Buy Now`, `Add to Cart`, `Description`)

## Search Action Rules
1.  **Use when:** You need to find new products. This is typically done from the main search page.
2.  **Keyword Strategy:** Combine key product terms from the instruction. Be specific but not overly long.
    *   Good: `search[jewelry box 10 slots under $60]`
    *   Less Effective: `search[box]`

## Error Handling
If you attempt an invalid action (e.g., `click[Something Not Listed]`), the environment will ignore it and the state will not change. Always verify the available actions list in the observation before responding.
