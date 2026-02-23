# Decision Workflow for Measurement-Based Tasks

This document outlines the thought process for planning and executing a measurement task, based on the successful trajectory.

## 1. Task Decomposition
*   **Input:** "Measure property of TARGET and perform ACTION_A if value > X, else ACTION_B."
*   **Decomposed Steps:**
    1.  Identify the required measurement `TOOL`.
    2.  Identify the `TARGET` object/substance.
    3.  Identify the location for `ACTION_A` and `ACTION_B` (e.g., `CONTAINER_A`, `CONTAINER_B`).
    4.  Identify the `DECISION_ROOM` where these containers are located.

## 2. Strategic Sequencing
**Optimal Order:** `Get TOOL` → `Get TARGET` → `Go to DECISION_ROOM` → `Measure` → `Act`.
*   **Why?** Measuring in the decision room minimizes unnecessary movement after obtaining the fragile measurement result, allowing immediate conditional action.

## 3. State Verification Checklist
Before executing `use TOOL on TARGET`, confirm:
- [ ] `TOOL` is in inventory. (Verify with `focus on TOOL`)
- [ ] `TARGET` is in inventory. (Verify with `focus on TARGET`)
- [ ] Current room is `DECISION_ROOM`.
- [ ] `CONTAINER_A` and `CONTAINER_B` are visible in the room (verified via `look around`).

## 4. Error Recovery Notes
*   **Tool/Target Not Found:** Use `teleport` to likely rooms (e.g., `kitchen`, `workshop` for tools; location specified in task for target) and `look around` systematically.
*   **Unparseable Observation:** If the `use` action doesn't return a clear number, try `examine TARGET` or `examine TOOL` for clues, or ensure the target is a valid substance for the tool.
*   **Wrong Room:** If you measure before reaching the decision room, simply `teleport to DECISION_ROOM` and proceed with the conditional action. The measurement result is retained.
