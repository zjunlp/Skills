# Trajectory Analysis: Task Completion Focus

## Source Context
This skill is derived from the final segment of a "grow a banana" task trajectory.

## Key Sequence
1.  **Prerequisite Achievement:** Multiple banana trees have reached the "reproducing stage" and are bearing fruit (`a banana`).
2.  **Verification:** Agent uses `look around` to confirm the presence of bananas.
3.  **Execution Attempt:** Agent issues `focus on banana on banana tree in the reproducing stage`.
4.  **Ambiguity:** Environment lists 4 possible banana targets across three flower pots.
5.  **Resolution:** Agent selects option `0`, corresponding to the banana in `flower pot 1`.

## Derived Rules
*   The skill is invoked **after** the core productive task is done.
*   The target object must be explicitly visible in the scene description.
*   Ambiguity is resolved by selecting the **first matching option** (`0`), establishing a simple, deterministic rule for the completion step.
*   This action is the terminal point of the task workflow.
