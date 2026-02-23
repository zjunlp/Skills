# Action Primer for Object Fetching

## Primary Action
`pick up OBJ`
*   **Effect:** Moves the specified object from the environment into the agent's inventory.
*   **Precondition:** The object must be present in the current room and accessible (not fixed, not inside a *closed* container).
*   **Postcondition:** The object is listed in the agent's inventory and can be referenced as `OBJ in inventory`.

## Supporting Actions
*   `look around`: **CRITICAL.** Use this to scan the current room and identify all visible objects and their states before attempting to pick anything up.
*   `look at OBJ`: Use if you need to inspect the contents of a container (e.g., a cupboard, a pot) to find your target object.
*   `focus on OBJ`: Use after a successful `pick up` to signal that the fetched item is the current subject of intent for the overarching task.
*   `teleport to LOC`: Use to navigate to the room where the target object is located before beginning the fetch sequence.

## Common Pitfalls & Assumptions
*   **Containers are Open:** The trajectory states "All containers in the environment have already been opened." Therefore, you can directly `pick up` items from within containers (e.g., from a pot, a cupboard) without needing to `open` them first.
*   **Object Names:** Use the exact name as shown in observations. For example, if the observation says `a tin cup (containing a substance called lead)`, the object to pick up would be the `tin cup`, not `lead`. To fetch the substance itself, you might need to `move` it to another container first.
*   **Inventory Management:** The skill is designed for single-object fetching. For complex tasks requiring multiple items, chain multiple executions of this skill.
