# Action Reference for ScienceWorld Conditional Placement

This guide details the common actions used in this skill within the ScienceWorld environment.

## Navigation & Exploration
*   `teleport to LOC`: Instantly move to a named room (e.g., `teleport to kitchen`, `teleport to bedroom`). Use this for efficient travel.
*   `look around`: Get a description of the current room and all visible objects. Use this to find items.

## Object Interaction
*   `pick up OBJ`: Move a visible object into your inventory. Use this for the measurement tool and the target object.
*   `examine OBJ`: Get a detailed description of a specific object. Useful for verification.
*   `look at OBJ`: List the contents of a container. Use this to inspect the target boxes.
*   `use OBJ [on OBJ]`: Apply a tool to an object. The primary measurement action (e.g., `use thermometer on metal fork`). Objects are typically referenced from inventory.
*   `move OBJ to OBJ`: Move an object from your inventory into a container (e.g., `move metal fork to blue box`). This is the final placement action.

## Skill-Specific Notes
*   **Containers are Open:** Do not use `open` or `close`. All cupboards, boxes, etc., are already accessible.
*   **Inventory Use:** After picking up objects, they are referred to as `OBJ in inventory` in some action descriptions, but the standard syntax `use thermometer on metal fork` works when both are in inventory.
*   **Focus Action:** The `focus on OBJ` action from the trajectory is a signal of intent and may not be necessary in all environment implementations. The core functional actions are `pick up`, `use`, and `move`.
