---
name: scienceworld-inventory-manager
description: This skill handles picking up objects from the environment into the agent's inventory or moving them out. It should be triggered when the agent needs to acquire an object for later use or place an object into a container. The input is an object name and location, and the output is the object being transferred to or from inventory.
---
# Instructions

## Purpose
Use this skill to manage the agent's inventory by picking up objects from the environment or placing them into containers. This is a foundational action for acquiring tools, materials, or evidence needed for tasks.

## When to Use
- You need to acquire an object for later use in an experiment or task.
- You need to place an object into a specific container (e.g., a box, a table, a room) as part of a task requirement.
- The object's location is known or can be inferred from context.

## Core Action Pattern
The primary action is `pick up OBJ` to acquire an object or `move OBJ to OBJ` to place it into a container. The exact syntax may vary slightly based on the environment's action space (e.g., `pick up metal pot containing nothing in kitchen` vs. `move metal pot to blue box`).

## Procedure
1.  **Locate the Object:** Ensure you are in the correct room or that the object is in your immediate vicinity. Use `look around` or `examine` if needed.
2.  **Acquire the Object:** If the object is not in your inventory, use the `pick up` action with the correct object identifier and location.
3.  **Place the Object (if required):** If the task requires placing the object into a container, use the `move` action with the target container's name.

## Key Considerations
- **Object State:** Some objects may be described as "containing nothing." Include this in the action if the environment's grammar requires it (e.g., `pick up metal pot containing nothing in kitchen`).
- **Container Targets:** When moving an object to a container like a box, use the container's name and color if specified (e.g., `blue box`, `orange box`).
- **Inventory Management:** You can only hold one item in your inventory at a time in the provided trajectory. Plan your actions accordingly.

## Example from Trajectory
**Goal:** Acquire the metal pot from the kitchen.
- Action: `pick up metal pot containing nothing in kitchen`
- Result: "You move the metal pot to the inventory."

**Goal:** Place the conductive metal pot into the correct box.
- Action: `move metal pot to blue box`
- Result: The object is transferred from inventory (or the room) to the specified container.

For complex object interactions or environment-specific grammar, consult the bundled reference.
