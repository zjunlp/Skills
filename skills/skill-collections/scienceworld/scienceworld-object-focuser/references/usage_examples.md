# Usage Examples & Patterns

## Example 1: Basic Focus & Move (from Trajectory)
**Task:** "Find a living thing. First, focus on the thing. Then, move it to the orange box in the workshop."

**Agent Execution:**
1.  Explore (`look around`) multiple rooms (living room, outside).
2.  **Identify Target:** In `outside`, observe `a dove egg`.
3.  **Apply Skill:** `focus on dove egg`
4.  **Result:** Observation confirms focus.
5.  **Next Action:** `pick up dove egg`, then `move dove egg to orange box`.

**Key Takeaway:** The skill is applied immediately after identifying the task-relevant object.

## Example 2: Focus as a Prerequisite
**Task:** "Create a circuit. Connect the battery to the light bulb."

**Potential Execution:**
1.  Locate `battery` and `light bulb` via `examine`.
2.  The task may require focusing on components before `connect`.
3.  **Apply Skill:** `focus on battery`
4.  **Next Action:** `connect battery to light bulb`

**Key Takeaway:** `focus` can be a required step before specific interaction verbs like `connect`.

## Common Patterns
*   **Exploration -> Identification -> Focus -> Manipulation:** The standard workflow.
*   **Re-focusing:** If you switch targets mid-task, you may need to `focus on` the new object.
*   **Implicit Focus:** Some objects may not require explicit focus. Use the skill when the task explicitly states it, or if a manipulation action fails without it.

## Anti-Patterns to Avoid
*   **Focusing on Non-Objects:** Do not `focus on` substances (e.g., `air`, `water`) or rooms.
*   **Incorrect Name:** `focus on dove` (incorrect) vs. `focus on dove egg` (correct). Always use the full name from the observation.
*   **Premature Focus:** Do not focus before confirming the object meets the task criteria.
