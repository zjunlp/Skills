# Common Heating Appliances in ALFWorld

This reference lists typical appliances used for the `heat` action and their common states.

## Primary Appliances
*   **microwave {N}**
    *   **Default State:** Usually closed.
    *   **Preparation:** Often requires opening. May contain other items that need removal.
    *   **Action:** `heat {obj} with microwave {N}`
*   **stoveburner {N}**
    *   **Default State:** Typically off (not toggled).
    *   **Preparation:** May need to be toggled on first (`toggle stoveburner {N}`). Requires a pot or pan for most foods.
    *   **Action:** `heat {obj} with stoveburner {N}` (often requires the object to be in a container).

## Secondary/Alternative Appliances
*   **toaster {N}:** For bread-like items.
*   **oven {N}:** For baking (less common in simple tasks).

## Appliance States & Keywords
When observing the environment, look for these phrases:
*   `is closed` / `is open`: For microwaves, ovens.
*   `is off` / `is on`: For stoveburners.
*   `In it, you see...`: Indicates contents of an open receptacle.
*   `Nothing happened.`: The previous action was invalid. Re-evaluate the object, appliance, or preconditions.

## Skill Execution Notes
1.  The `heat` action is often an abstraction. You may not need to explicitly place the object inside the appliance first.
2.  If an appliance is occupied, the task may or may not require you to empty it. Use context from the goal description (e.g., "heat some egg" vs. "prepare a meal by heating an egg").
3.  Always verify the object is in your inventory or at the target location after heating.
