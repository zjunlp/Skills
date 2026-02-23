# ALFWorld Action Primer for Receptacle Closing

## Available Actions Context
The agent operates with a discrete set of actions. The relevant action for this skill is:
- `close {recep}`: Closes the specified receptacle if it is currently open.

## Common Receptacle Types
*   `drawer #`
*   `cabinet #`
*   `fridge #`
*   `microwave #`
*   `box #`
*   `washingmachine #`

## Trajectory Logic Extract
The skill is derived from the following step in the provided trajectory:
1.  Agent opened `drawer 2` to inspect contents.
2.  Agent observed contents (`nothing`).
3.  Agent's **Thought** explicitly stated the tidiness rationale: *"I should close drawer 2 to keep the room tidy and continue my search."*
4.  Agent executed **Action:** `close drawer 2`.

This demonstrates the skill's trigger condition: **post-inspection cleanup**.
