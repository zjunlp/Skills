# Example Trajectory: Examine Pillow with Desklamp

This document breaks down the provided execution trajectory to illustrate the skill's logic.

## Trajectory Steps
1.  **Goal:** "examine the pillow with the desklamp."
2.  **Step 1 (Locate Object):** `go to sofa 1` -> No pillow found.
3.  **Step 2 (Locate Object):** `go to armchair 1` -> `pillow 1` found.
4.  **Step 3 (Acquire Object):** `take pillow 1 from armchair 1` -> Success.
5.  **Step 4 (Locate Tool):** `go to sidetable 1` -> `desklamp 1` found.
6.  **Step 5 (Execute Interaction):** `use desklamp 1` -> (Implied success).

## Key Inferences for Skill Design
- **Object First:** The agent prioritizes finding and acquiring the *target object* (`pillow`) before seeking the *tool* (`desklamp`).
- **Tool Location:** Tools are typically found on surfaces like `sidetable`, `shelf`, or `drawer`.
- **Final Action:** The verb `use` is applied directly to the tool when the context is "examine X with Y".
- **Error Handling:** The agent seamlessly switches search locations (`sofa` -> `armchair`) upon not finding the object.

## Abstracted Skill Flow
1.  **Acquire Target:** `Navigate -> [Search Receptacles] -> Take Object`
2.  **Locate Tool:** `Navigate -> Identify Tool`
3.  **Interact:** `Perform({interaction_verb}, {tool})`
