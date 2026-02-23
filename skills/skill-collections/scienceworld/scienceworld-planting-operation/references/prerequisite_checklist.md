# Prerequisite Checklist for Planting Operation

Execute the Planting Operation skill **only** when the following conditions are met:

## Mandatory Prerequisites
- [ ] **Seeds Acquired:** You have a source of seeds in your inventory or within reach (e.g., `seed jar` in inventory). Verify with `look at <seed_source>`.
- [ ] **Containers Prepared:** The target planting containers (e.g., `flower pot 1`, `flower pot 2`) exist in the environment and **contain soil**. Verify with `examine <container>`.
- [ ] **Containers Empty:** The containers should not already contain a plant or seed. They must be ready for new planting.

## Recommended Pre-Skill Actions
1.  **Acquire Seeds:** Use `pick up <seed_source>` or similar.
2.  **Prepare Soil:**
    - Locate a soil source (e.g., `outside` ground).
    - Use `shovel` to dig soil (`use shovel on ground`).
    - `pick up soil`.
    - `move soil to <container>` for each target container.
3.  **Positioning:** Be in the same location (room) as the target containers.

## Common Failure Points
*   **No Soil:** Attempting to plant in a container with only water or nothing.
*   **Already Planted:** Trying to plant in a container that already has a seed or plant.
*   **Wrong Location:** The seed source or containers are in a different room.
