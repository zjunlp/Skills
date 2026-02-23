# Primer: The 'focus on' Action

## Environmental Role
The `focus on OBJ` action is a special meta-action within this simulation environment. It does not manipulate the object physically but signals to the task system that the agent has successfully interacted with or produced the object relevant to the assigned goal.

## Semantics
*   **Completion Signal:** Its primary use is to mark a task as **complete**.
*   **Intent Declaration:** It can also be used to declare an intent to work on a specific object in multi-step tasks, though its most critical function is finalization.
*   **Non-Destructive:** The action does not consume, move, or alter the object.

## Handling Ambiguity
The environment often requires disambiguation when multiple instances of a target object exist. The standard pattern is:
1.  Agent sends: `focus on <OBJECT> on <CONTAINER>`
2.  Environment responds with an "Ambiguous request" and a numbered list.
3.  Agent must respond with a single number (e.g., `0`).

**Best Practice:** When the goal is simple completion (e.g., "focus on the grown banana"), selecting the first valid option (`0`) is the correct and sufficient strategy, as demonstrated in the provided trajectory.
