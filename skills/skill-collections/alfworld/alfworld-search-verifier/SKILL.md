---
name: alfworld-search-verifier
description: This skill re-examines previously visited locations to confirm the absence of a target object or to check for overlooked items. It should be triggered when an initial search fails to find enough objects or when double-checking is required for task completion. The skill systematically revisits receptacles, ensuring no viable location is missed before concluding the search.
---
# Instructions

**Trigger Condition:** Use this skill when an initial search for a target object (e.g., `cellphone`) has been performed but the required quantity has not been found, or when you need to verify that no viable location was missed before task failure.

## Core Procedure

1.  **Initialize Tracking:** Maintain a mental or explicit list of all receptacles (`recep`) you have already visited during your initial search (e.g., `sidetable 1`, `shelf 1-8`, `drawer 1`, `drawer 2`, `desk 1`).

2.  **Systematic Revisit:** Do not search new, unvisited locations. Instead, systematically return to each previously visited receptacle in a logical order (e.g., by proximity or by the original search sequence).
    *   **Action:** Use `go to {recep}` to navigate.
    *   **For Closed Receptacles:** If a receptacle was closed during the initial search and you opened it, it may be closed again. Re-open it to verify its contents.
        *   **Action:** Use `open {recep}`.
    *   **For Open/Empty Receptacles:** Observe the contents again. The state may have changed, or an item may have been overlooked.
    *   **Action:** No action needed; read the `Observation`.

3.  **Verification Logic:** For each revisited location:
    *   If the target object is now present, retrieve it (`take {obj} from {recep}`) and proceed with the main task.
    *   If the location is confirmed to lack the target object, note it as thoroughly checked and move to the next location on your list.
    *   If you encounter a closed receptacle you previously assumed was empty, opening it is a critical verification step.

4.  **Conclusion:** After revisiting all locations on your list:
    *   **If the target object was found:** Integrate it into your main task plan.
    *   **If the target object was not found:** You can conclusively report that the object is not available in the searched area and adjust your task strategy accordingly (e.g., consider task failure or exploring a new, unsearched area).

## Thought Process Template
When executing this skill, structure your reasoning as follows:
