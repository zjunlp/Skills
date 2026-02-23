# Action Guide for ScienceWorld Temperature Measurement

## Essential Actions
| Action | Purpose | Example | Expected Observation Key Phrase |
| :--- | :--- | :--- | :--- |
| `pick up OBJ` | Acquire a tool or sample. | `pick up thermometer` | `You move the thermometer to the inventory.` |
| `move OBJ to OBJ` | Transfer a substance or item. | `move lead to metal pot` | `You move the lead to the metal pot.` |
| `activate OBJ` | Start a device. | `activate blast furnace` | `The blast furnace is now activated.` |
| `use OBJ on OBJ` | Apply a tool to a target. | `use thermometer in inventory on lead` | `the thermometer measures a temperature of X degrees celsius` |
| `teleport to LOC` | Instant movement between rooms. | `teleport to foundry` | `You teleport to the foundry.` |
| `look at OBJ` | Inspect contents of a container/device. | `look at blast furnace` | `In the blast furnace is: a metal pot (containing...)` |
| `focus on OBJ` | Signal intent or confirm target. | `focus on red box` | `You focus on the red box.` |

## Critical Observations for State Tracking
*   **Substance State Change:** `a substance called lead` → `a substance called liquid lead`. This indicates melting has occurred.
*   **Temperature Reading:** The output of the `use thermometer` action is the numerical result. **Example:** `the thermometer measures a temperature of 450 degrees celsius`.
*   **Device Status:** `which is turned on` vs. `which is turned off`. Confirm heaters are active.

## Common Pitfalls & Solutions
1.  **"No known action matches that input" when using thermometer.**
    *   **Cause:** The target substance name might have changed state (e.g., `lead` -> `liquid lead`) or is not directly accessible.
    *   **Solution:** Use `look at [container]` to check the exact name of the substance inside, then use that name (e.g., `use thermometer ... on liquid lead`).

2.  **Substance not heating.**
    *   **Cause:** The heating device may not be active, or the sample is not inside it.
    *   **Solution:** Verify with `look at [heating device]`. Ensure `activate [device]` was successful and the container is inside.

3.  **Cannot find the thermometer or substance.**
    *   **Cause:** Not in the correct room.
    *   **Solution:** Use `look around` to survey the current room. Use `teleport` to move to likely rooms (e.g., `kitchen` for tools, `foundry` for heaters).

## Workflow Triggers
Use this skill when the goal involves:
*   "measure the temperature of [substance]"
*   "find the melting point of [metal]"
*   "check if [material] is above/below [number] degrees"
