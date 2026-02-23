# Action Primer for ScienceWorld Environment

This document lists common actions used in conjunction with the `scienceworld-instruction-reader` skill to execute tasks based on extracted instructions.

## Navigation & Exploration
*   `teleport to LOC`: Instantly move to a named room (e.g., `teleport to workshop`).
*   `look around`: Observe the current room's contents and exits.

## Inventory & Object Manipulation
*   `pick up OBJ`: Move an object into your inventory.
*   `examine OBJ`: Get a detailed description of an object, including its contents if it's a container.
*   `move OBJ to OBJ`: Place an object into a container or onto a surface.

## Task Execution (Common for Chemistry/Experiments)
*   `mix OBJ`: Chemically combine the contents of a container.
*   `pour OBJ into OBJ`: Transfer a liquid between containers.
*   `use OBJ [on OBJ]`: Employ a tool or device.
*   `activate OBJ` / `deactivate OBJ`: Turn a device (like a sink) on or off.

## Task Completion
*   `focus on OBJ`: Signal intent on a final task object to indicate completion.
