# Action Primer for Object Retrieval

## Essential Actions
*   `look around`: Describes the current room and lists all visible objects and agents. Use this first to find your target.
*   `pick up OBJ`: Moves the specified object from the environment into your inventory. This is the core action of this skill.
*   `teleport to LOC`: Instantly moves you to a named room (e.g., `teleport to workshop`). Use if the object is known to be in another location.

## Supporting Actions (Use as Needed)
*   `examine OBJ`: Provides a detailed description of a specific object or piece of furniture, which may list its contents.
*   `open OBJ`: Opens a closed container (e.g., a cabinet, box) so its contents can be seen and accessed.
*   `look at OBJ`: Describes the contents of a container. Use after opening it or if it's already open.

## Common Object Locations
Objects are typically:
*   Loose in the room.
*   On surfaces (e.g., `on the table`).
*   Inside open containers (e.g., `in the blue box`).
*   Inside closed containers (requires `open` first).
