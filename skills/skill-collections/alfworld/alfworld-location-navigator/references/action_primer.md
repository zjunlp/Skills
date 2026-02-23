# Alfworld Action Primer

## Core Navigation Action
The primary action used by this skill is:
- **`go to {recep}`**: Moves the agent to the specified receptacle or object location.

## Action Format Rules
1.  The action must be output exactly as: `go to <target_name>`
2.  `<target_name>` must match the environment's object naming exactly (e.g., "toiletpaperhanger 1", not "toilet paper hanger").
3.  The agent must be able to see the target in its current observation for the action to succeed.

## Common Navigation Patterns from Trajectory
1.  **Initial Approach**: `go to toiletpaperhanger 1` - Move to inspect a target.
2.  **Object Retrieval Path**: `go to toilet 1` - Move to a location containing a needed object.
3.  **Return for Placement**: `go to toiletpaperhanger 1` - Return to original target after obtaining object.

## Error Handling
- If the environment responds with "Nothing happened", the `go to` action likely failed because:
    - The target name was incorrect or misspelled.
    - The target is not currently visible/accessible from the agent's location.
    - The target is not a valid receptacle/object for navigation.
- Recommended response: Re-examine the observation for correct target names and accessible paths.
