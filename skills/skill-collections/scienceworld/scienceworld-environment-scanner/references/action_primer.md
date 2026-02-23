# ScienceWorld Action Primer for Environment Scanning

## Primary Scanning Action
*   `look around`: Describes the current room, listing all visible objects, substances, and doors to other rooms. This is the core action for the `scienceworld-environment-scanner` skill.

## Context for Other Common Actions
*(This section is for reference when planning post-scan steps. Do not use these during the scan itself.)*

### Navigation
*   `teleport to LOC`: Instantly moves the agent to a named room (e.g., `teleport to kitchen`). Use after scanning to move towards a goal.

### Object Interaction
*   `examine OBJ`: Provides a detailed description of a specific object.
*   `look at OBJ`: Lists the contents inside a container.
*   `pick up OBJ`: Moves a freestanding object into the agent's inventory.

### Task Signaling
*   `focus on OBJ`: Explicitly marks an object as the current target of the task. Often required before manipulating key objects.

## Important Constraints (from Trajectory)
*   All containers are pre-opened. You can `look at` them directly without needing to `open` them first.
*   You can `teleport` to any known room in a single step.
