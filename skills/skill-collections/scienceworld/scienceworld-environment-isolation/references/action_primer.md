# Action Primer for Environment Isolation

## Relevant Standard Actions
*   `look around`: **Critical.** Use this first to survey the room and identify all closable openings (doors, gates, windows).
*   `close OBJ`: The core action. `OBJ` must be the exact name of the opening as listed in the `look around` observation (e.g., "door to the outside").

## Handling Ambiguity
If an action (like `focus on OBJ`) returns an "Ambiguous request" prompt after isolation, it is unrelated to this skill. Resolve it by selecting the appropriate numeric option corresponding to your intended target.

## Sequencing with Other Skills
1.  **Preparation First:** Often, tasks like planting (`move seed to pot`) or setting up devices should be completed *before* isolation.
2.  **Isolation Middle:** Apply this skill after setup but before the critical waiting period for the process (e.g., pollination).
3.  **Monitoring After:** Use `wait` and periodic `look around` checks after isolation to observe outcomes.
