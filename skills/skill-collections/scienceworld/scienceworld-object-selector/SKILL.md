---
name: scienceworld-object-selector
description: Selects appropriate objects from available options based on task criteria. Use this skill when you need to choose a specific object type (e.g., non-living thing) from multiple candidates. It evaluates object properties against task requirements and selects the most suitable candidate for further manipulation.
---
# Instructions

Use this skill to identify and select an object from a list of candidates that matches a given task requirement (e.g., "non-living thing", "electrical component", "container").

## Core Workflow
1.  **Observe & Parse:** Use `look around` to get a list of all visible objects in your current location.
2.  **Evaluate Candidates:** For each object, determine if it matches the task's criteria. Refer to the `references/object_properties.md` for common classifications.
3.  **Select & Focus:** Choose the most suitable candidate. Use `focus on [OBJECT]` to signal your intent and proceed with the next task step (e.g., `pick up`, `move`, `use`).

## Key Principles
*   **Conciseness:** Choose the first suitable object unless the task implies a specific preference (e.g., "largest", "closest").
*   **Verification:** If uncertain about an object's properties, use `examine [OBJECT]` for more detail before selecting.
*   **Task Alignment:** Your selection should enable the *next action* in the task sequence.

## Example (from trajectory)
**Task:** "Find a non-living thing. First, focus on the thing. Then, move it to the purple box."
1.  `look around` reveals: `purple box`, `table`, `battery`, `black wire`, `blue light bulb`, `red light bulb`, `red wire`, `switch`, `violet light bulb`, `yellow wire`, `ultra low temperature freezer`.
2.  Evaluate: The `purple box` is the target location. The `table` and `freezer` are furniture. All other items (`battery`, `wires`, `bulbs`, `switch`) are non-living.
3.  Select: `black wire` is a straightforward, portable non-living object.
4.  Action: `focus on black wire` -> `move black wire to purple box`.
