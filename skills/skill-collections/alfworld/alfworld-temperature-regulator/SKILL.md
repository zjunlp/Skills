---
name: alfworld-temperature-regulator
description: This skill manages the temperature state of an object by placing it into an appropriate environment (e.g., fridge for cooling, microwave for heating). It should be triggered when the task requires modifying an object's temperature property, such as cooling bread or heating food. The skill requires the object identifier and the target temperature-modifying receptacle as inputs and confirms the object's placement for the intended thermal effect.
---
# Instructions
This skill executes a sequence to change an object's temperature by placing it in a specific receptacle (e.g., fridge for cooling, microwave for heating) and then relocating it to a final target location.

## 1. Input Validation & Planning
*   **Inputs Required:** The `object` identifier (e.g., `bread 1`) and the `temperature_receptacle` identifier (e.g., `fridge 1` for cooling, `microwave 1` for heating). The final `target_receptacle` (e.g., `diningtable 1`) is also required.
*   **Verify** the provided object and receptacles exist in the agent's current observation. If not, the agent must first navigate to locate them.
*   **Plan** the sequence: Locate object -> Pick up object -> Navigate to temperature receptacle -> Open it (if closed) -> Place object inside -> Close receptacle (optional, based on environment feedback) -> Navigate to target receptacle -> Place object there.

## 2. Execution Sequence
Follow this core logic. Use deterministic scripts for error-prone steps (see `scripts/`).
1.  **Acquire Object:** `go to` the object's location, then `take {object} from {recep}`.
2.  **Apply Temperature Effect:**
    *   `go to {temperature_receptacle}`.
    *   If the receptacle is reported as "closed", `open {temperature_receptacle}`.
    *   `put {object} in/on {temperature_receptacle}`.
    *   (Optional) `close {temperature_receptacle}` if the environment or task logic suggests it (e.g., maintaining fridge temperature).
3.  **Deliver Object:** `go to {target_receptacle}`, then `put {object} in/on {target_receptacle}`.

## 3. Error Handling & Observations
*   If an action results in "Nothing happened", consult the troubleshooting guide in `references/troubleshooting.md`.
*   Always verify the state change after each action (e.g., "You pick up...", "You open...", "You put...").
*   If the object is not at the expected location, pause execution and re-scan the environment.

## 4. Completion
The skill is complete when the object has been placed into the `temperature_receptacle` and subsequently placed onto the `target_receptacle`. Confirm the final observation states the object is on the target.
