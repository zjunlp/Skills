# Action Primer for ScienceWorld Environment

This reference details common actions used in conjunction with the `scienceworld-room-explorer` skill for task execution.

## Foundational Action
*   `look around`: Describes the current room, listing all visible objects, their states, and contents. **This is the core action of the scienceworld-room-explorer skill.**

## Common Follow-up Actions
After using `scienceworld-room-explorer`, you will typically use other actions to manipulate the discovered objects.

### Object Interaction
*   `examine OBJ`: Get a detailed description of a specific object's properties.
*   `look at OBJ`: List the contents of a container.
*   `pick up OBJ`: Move an object into your inventory.
*   `move OBJ to OBJ`: Place an object into a container or onto a surface.
*   `use OBJ [on OBJ]`: Operate a device or apply an item.

### Electrical & Device Manipulation
*   `connect OBJ to OBJ`: Link two electrical components (e.g., wire terminal to battery terminal).
*   `disconnect OBJ`: Remove a connection from a component.
*   `activate OBJ` / `deactivate OBJ`: Turn a device on or off.
*   `focus on OBJ`: Signal intent to interact with a specific task-relevant object.

### Navigation
*   `teleport to LOC`: Instantly move to another room.

## Action Sequencing Pattern
A standard task execution pattern often follows this flow:
1.  **Explore:** Use `scienceworld-room-explorer` (`look around`) to understand the room.
2.  **Locate:** Identify the target object(s) from the observation.
3.  **Acquire:** Use `pick up` or navigate to the object.
4.  **Prepare:** Use `examine`, `connect`, or other actions to set up the object for the task.
5.  **Execute:** Use `use`, `activate`, `mix`, etc., to perform the core task action.
6.  **Conclude:** Use `move` or other actions to complete the task (e.g., placing an object in a specified box).
