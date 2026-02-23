---
name: scienceworld-container-inspector
description: This skill performs a 'look at' action to inspect the contents of a specific container or device. It should be triggered when the agent needs to verify what is inside a container (e.g., checking if lead is in the blast furnace) or monitor the state of contents (e.g., solid vs. liquid). The skill outputs a detailed list of contents and their states, providing essential feedback for process monitoring.
---
# Skill: Container/Device Inspector

## Purpose
Execute a `look at` action on a specified container or device to retrieve a detailed observation of its contents and their states. This is a critical monitoring skill for verifying process steps, such as confirming an item is present inside a furnace or checking the phase (solid/liquid) of a substance.

## When to Use
Trigger this skill when you need to:
1.  **Verify Placement:** Confirm an object has been successfully moved into a target container (e.g., "Is the metal pot in the blast furnace?").
2.  **Monitor State Change:** Check if the contents of a container have changed state due to a process (e.g., "Has the lead melted into liquid lead?").
3.  **Inventory Check:** List all items currently stored inside an open container or device.

## Core Instruction
**Always use the exact action:** `look at <CONTAINER_NAME>`
-   Replace `<CONTAINER_NAME>` with the precise name of the target object (e.g., `blast furnace`, `cupboard`, `tin cup`).
-   This action returns a structured observation detailing the container's state and a list of its contents.

## Expected Output & Interpretation
The observation will follow this general pattern:
`[CONTAINER], which is [STATE]. The [CONTAINER] door is [OPEN/CLOSED]. In the [CONTAINER] is: \n\t[LIST OF CONTENTS]`

**Key Information to Extract:**
1.  **Container State:** Is it activated/turned on? (e.g., "which is turned on").
2.  **Door State:** Is it open or closed? You can only see contents if the door is open.
3.  **Contents List:** A nested list of all objects inside. Pay close attention to:
    -   **Item Names:** (e.g., `a metal pot`).
    -   **Substance States:** The description may reveal state changes (e.g., `a substance called liquid lead` vs. `a substance called lead`).
    -   **Nested Containers:** Contents may themselves be containers (e.g., `a metal pot (containing a substance called lead)`).

## Example from Trajectory
**Action:** `look at blast furnace`
**Observation:** `a blast furnace, which is turned on. The blast furnace door is open. In the blast furnace is: \n\ta metal pot (containing a substance called liquid lead)`
**Interpretation:** The blast furnace is active and open. It contains one item: a metal pot, which itself contains liquid lead. This confirms the heating process was successful and the lead has melted.

## Integration with Other Actions
This skill is often used in a sequence:
1.  `move <ITEM> to <CONTAINER>` (Place an item).
2.  `look at <CONTAINER>` (This skill - verify placement).
3.  `activate <CONTAINER>` (Start a process).
4.  `look at <CONTAINER>` (This skill - monitor state change).
5.  `use <TOOL> on <CONTENT>` (Proceed based on observed state).

## Error Handling
-   If the `look at` action fails or returns an unexpected result, first verify the object's name is correct by using `look around` or `examine`.
-   If the container door is closed, you must `open` it before using this skill.
-   The skill only inspects; it does not manipulate contents. Use `pick up`, `move`, or `use` for manipulation.

---
*For detailed examples of container interactions and state transitions, see the reference documentation.*
