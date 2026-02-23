# Troubleshooting Guide: Temperature Regulation

## Common Issues & Solutions

### 1. Action Results in "Nothing happened"
*   **Object not found:** The `{obj}` or `{recep}` in your action may not be visible or may have a different identifier (e.g., `bread 2` instead of `bread 1`). Re-examine the latest observation list.
*   **Invalid syntax:** Ensure your action string exactly matches the available action format (e.g., `take bread 1 from countertop 2`, not `pick up bread`).
*   **Precondition not met:** You cannot `take` an object you are not near. You cannot `put` an object in a receptacle that is closed. Always `go to` the relevant location first and `open` closed receptacles.

### 2. Receptacle is Already Open/Closed
*   If you try to `open` an already open receptacle, it may yield "Nothing happened". Check the observation text (e.g., "The fridge 1 is open."). Skip the `open` action if already open.
*   Similarly, only `close` a receptacle if it was explicitly opened during this skill execution and the task logic requires it.

### 3. Object Not at Expected Initial Location
*   The trajectory showed `bread 1` on `countertop 2`. This may not always be true.
*   **Solution:** If the object is not at the planned location, you must first find it. Use the `go to` action to search common receptacles mentioned in the initial observation (e.g., other countertops, shelves, diningtable, cabinets). The skill must be flexible to this variation.

### 4. Heating vs. Cooling Logic
*   **Cooling:** Use `fridge 1`. Sequence: `open fridge` -> `put object in` -> `close fridge`.
*   **Heating:** Use `microwave 1`. Sequence: `open microwave` -> `put object in` -> `close microwave` -> `toggle microwave microwave`. The `toggle` action may be needed to start the heating process. Check the environment's action list and feedback.

## Flowchart for Robust Execution
