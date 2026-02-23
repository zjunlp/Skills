# Action Primer for Planting

## Relevant Actions from the Trajectory
- `look around`: Describes the current room and its contents.
- `teleport to LOC`: Instantly moves the agent to a specified room.
- `pick up OBJ`: Moves an object from the environment into the agent's inventory.
- `move OBJ to OBJ`: Transfers an object from one container to another. This is the **core action** for planting.
- `focus on OBJ`: Signals intent to monitor or work on a specific object. Use after planting.

## Handling Ambiguous Actions
The environment may sometimes return an ambiguous request for a `move` action, presenting multiple identical-seeming options with indices (0, 1, 2...).

**Resolution Protocol:**
1.  The agent's command (`move avocado seed in seed jar to flower pot 1`) triggers the ambiguity.
2.  The environment responds with a numbered list of options.
3.  **Apply the skill's deterministic rule:** Respond with the action corresponding to the first index (`0`).
4.  This selects the standard transfer path and allows the action to proceed.

## Container Suitability Check
A container is suitable for planting if its description includes:
- `soil` (a substance)
- `water` (a substance)

Example from trajectory: `a flower pot 1 (containing soil, a substance called water)`
