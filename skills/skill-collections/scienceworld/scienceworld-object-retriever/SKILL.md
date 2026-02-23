---
name: scienceworld-object-retriever
description: This skill acquires a specified object by moving it from the environment into the agent's inventory. It should be triggered when a task requires an object to be manipulated, tested, or transported. The skill uses the 'pick up OBJ' action on a target object identified in the room, making it available for further actions in the inventory.
---
# Instructions

## Primary Objective
Acquire a specified target object from the environment and place it into your inventory.

## Core Procedure
1.  **Locate the Object:** Use the `look around` action to survey the current room. Identify the target object from the description.
2.  **Acquire the Object:** Use the `pick up OBJ` action on the identified target object. Replace `OBJ` with the exact name of the object as seen in the environment description.
3.  **Verification:** Confirm the object is now in your inventory by checking the observation feedback from the `pick up` action.

## Key Notes
*   This skill is the foundational step for any task requiring physical interaction with an object (e.g., testing, moving, using).
*   Ensure you are in the correct room containing the object before attempting to pick it up. Use `teleport to LOC` if necessary.
*   The object must be accessible and not inside a closed container. If a container is closed, use `open OBJ` first.
*   If the initial `look around` does not reveal the object, you may need to `examine` specific containers or furniture (e.g., `examine table`) to find it.
*   After acquisition, the object is ready for the next skill in the task sequence (e.g., testing conductivity, placing in a specific box).
