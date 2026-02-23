---
name: alfworld-receptacle-finder
description: This skill searches for a suitable empty or appropriately occupied receptacle (like a shelf) to place an object. It should be triggered when the agent needs to store or place an object and must evaluate available receptacles. The skill examines candidate receptacles and identifies one meeting the placement criteria.
---
# Instructions

**Trigger:** Use this skill when you have an object in your inventory that needs to be stored or placed, and you must find a suitable receptacle (e.g., shelf, table, basin).

## Core Procedure
1.  **Identify Target Object:** Confirm the object you are holding and its state (e.g., `soapbar 1`, `clean`).
2.  **Survey Environment:** Use `go to {recep}` actions to navigate to and visually inspect candidate receptacles from the observation list.
3.  **Evaluate Candidates:** For each receptacle, assess its suitability:
    *   **Primary Goal:** Find an **empty receptacle**.
    *   **Fallback:** If no empty receptacle is available, identify one that can logically accommodate the object without conflict (e.g., a shelf with a tissuebox may still have space for a soapbar).
4.  **Select & Place:** Navigate to the chosen suitable receptacle and execute `put {obj} in/on {recep}`.

## Key Considerations
*   **Object State:** Ensure the object meets any required pre-conditions (e.g., `clean`) before searching for a receptacle. Use skills like `clean` if necessary.
*   **Efficiency:** Prioritize receptacles that are logically appropriate for the object (e.g., a `shelf` for a `soapbar` over a `toilet`).
*   **Observation Parsing:** Carefully read the observation after each `go to` action. The line "On the {recep}, you see..." lists its contents. "you see nothing." indicates it is empty.
*   **Invalid Actions:** If an action fails ("Nothing happened"), re-evaluate your target and try a different receptacle or confirm the object is in your inventory.

## Example Trajectory (from `references/example_trajectory.md`)
For a detailed example of this skill in action, including handling object pre-cleaning and sequential shelf evaluation, refer to the bundled reference.
