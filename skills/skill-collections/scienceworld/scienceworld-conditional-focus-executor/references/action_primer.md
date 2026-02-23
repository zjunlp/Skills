# ScienceWorld Action Primer for Conditional Tasks

## Core Actions Used in Trajectory
*   `teleport to <LOC>`: Move instantly to a named room.
*   `look around`: Observe all objects and doors in the current room.
*   `pick up <OBJ>`: Move a portable object into your inventory.
*   `focus on <OBJ>`: Signal intent or completion on a task-relevant object. **This is the key action for this skill.**
*   `move <OBJ> to <OBJ>`: Place an object inside a container or onto a surface.
*   `activate <OBJ>`: Turn on a device (e.g., stove).
*   `look at <OBJ>`: Inspect a specific object or container and its contents.
*   `examine <OBJ>`: Get a detailed description of an object's state.
*   `use <OBJ> on <OBJ>`: Employ a tool (e.g., thermometer) on a target.

## Object Interaction Patterns
1.  **Finding Objects:** Use `look around` upon entering a room.
2.  **Retrieving Objects:** If in an open container, `pick up` directly.
3.  **Preparing Experiments:** Often involves `move` items to appropriate containers (pots, bowls) and then to devices (stove).
4.  **Taking Measurements:** Use measurement tools (`use thermometer on chocolate`) on the target.
5.  **Conditional Completion:** The final step is always `focus on <COLOR> box` based on the result.

## Common Conditional Task Structure
The provided trajectory follows a classic pattern:
1.  **Locate & Setup:** Find tools and materials (`teleport`, `look around`, `pick up`).
2.  **Prepare Experiment:** Arrange items for the process (`move`).
3.  **Execute Process:** Activate devices (`activate`).
4.  **Monitor & Measure:** Repeatedly check state (`look at`, `examine`) and measure (`use ... on`).
5.  **Evaluate & Conclude:** Compare final measurement to threshold, then `focus on` the correct object.

## Key Assumptions (From Trajectory)
*   All containers (fridge, cupboard, etc.) are **already open**.
*   Objects inside open containers can be interacted with directly.
*   The `focus` action is the designated method to signal task completion for conditional outcomes.
