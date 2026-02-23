# Strategy Documentation for Option Selection

## Overview
This document details the decision-making strategies for the `webshop-option-selector` skill.

## Primary Strategy: Cost Minimization
**When to Use:** Default strategy when no user preference is detected.
**Logic:** Select the option value presumed to be the lowest cost. In e-commerce, this is often:
- The smallest quantity (e.g., "1pcs" vs "2pcs").
- The base model or standard variant.
- The default color or size.

**Implementation Note:** The current script uses a simple heuristic: select the **first listed value** for an option category, as product pages often list options in ascending order of price or quantity. For a production system, this would require integrating with a live pricing API or parsing dynamic price tags.

## Secondary Strategy: User Preference Matching
**When to Use:** When the user's original instruction contains explicit keywords matching an available option.
**Logic:**
1. Extract keywords from the instruction (e.g., "large", "blue", "2 pack").
2. For each configurable option category, check if any value contains a matching keyword (case-insensitive).
3. Select the first matching value.

**Precedence:** User preference overrides cost minimization.

## Edge Cases & Handling
1. **No Options Found:** Skill does not trigger. Return no action.
2. **Multiple Categories:** Process categories in the order they appear on the page. The script currently selects for the first category only; for multiple categories, the agent should loop.
3. **Price Constraint Violation:** The skill assumes the agent has already validated the product price against the user's budget (e.g., "under $40") before reaching the option page. If an option selection would exceed the budget, the agent should backtrack.
4. **Unclear Cost Difference:** If the price impact of options is unknown, default to the first option.

## Integration with Agent
The agent should:
1. Check the trigger condition (product detail page with options).
2. Call the `parse_and_select.py` script (or its logic) with the current observation and the original user instruction.
3. Execute the returned `click[value]` action.
4. Observe the resulting state (updated price, new options) and decide on subsequent actions (e.g., select another option, click "Buy Now").
