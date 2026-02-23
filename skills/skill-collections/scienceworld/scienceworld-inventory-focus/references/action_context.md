# Action Context for ScienceWorld Environment

## Relevant Actions for This Skill
*   `pick up OBJ`: Moves an object from the environment to the agent's inventory. A prerequisite for using the `focus` skill on that object.
*   `focus on OBJ`: The core action of this skill. When the object is in inventory, use the format `focus on OBJ in inventory`.

## Environment Assumptions (From Trajectory)
The provided trajectory operates under specific world assumptions that simplify task execution:
*   **Teleportation:** You can `teleport to LOC` instantly between named rooms.
*   **Open Containers:** All containers (cupboards, drawers, fridges) are already open. You can directly see and take items from them.
*   **Inventory:** The agent has an implicit inventory. Picked-up items are moved "to the inventory" and can be focused on or used from there.

## Common Object Names
Based on the trajectory, be precise with object names:
*   `thermometer`
*   `unknown substance B`
*   `blue box`
*   `green box`
*   `lighter`, `stopwatch`, `glass jar`, etc.
