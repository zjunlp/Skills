# Action Primer for scienceWorld

## Relevant Actions for This Skill

### Primary Action
*   **`move OBJ to OBJ`**: Transfers an object from your inventory or current location to a container or surface.
    *   Format: `move <object_name> to <container_name>`
    *   Example: `move dove egg to orange box`
    *   **Critical:** The object must be accessible (in inventory or in the room).

### Prerequisite & Support Actions
*   **`teleport to LOC`**: Instantly moves the agent to a named room.
    *   Format: `teleport to <room_name>`
    *   Example: `teleport to workshop`
    *   Rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway.

*   **`focus on OBJ`**: Signals intent on a task-relevant object. Use to confirm you have identified the correct item before picking it up or moving it.
    *   Format: `focus on <object_name>`
    *   Example: `focus on dove egg`

*   **`pick up OBJ`**: Moves an object from the environment into your inventory. Required before you can `move` an item that is not already in your inventory.
    *   Format: `pick up <object_name>`
    *   Example: `pick up dove egg`

*   **`look around`**: Describes the current room and lists visible objects and containers. Use to locate the target container.
    *   Format: `look around`

## Common Object Naming Patterns
*   Objects may be referred to with articles (e.g., "a dove egg", "the orange box"). The action command typically uses the noun phrase without the article.
*   Compound names are used directly (e.g., `dove egg`, `orange box`).
*   If an object is described as being "on" something, it is a separate object (e.g., "a white pillow" on "a couch").
