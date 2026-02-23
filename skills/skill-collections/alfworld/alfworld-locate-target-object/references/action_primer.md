# Action Primer for ALFWorld

This reference details the core actions used in the `alfworld-locate-target-object` skill.

## Primary Skill Actions
| Action Format | Purpose | Example in Skill Context |
| :--- | :--- | :--- |
| `go to {recep}` | Move the agent to a specific receptacle or appliance. | `go to fridge 1` |
| `open {recep}` | Open a closed receptacle to see or access its contents. | `open fridge 1` |

## Critical Observation Patterns
*   `The {recep} is closed.` -> You must `open` it.
*   `You open the {recep}. The {recep} is open. In it, you see...` -> The contents are now listed. Scan for your target object.
*   `Nothing happened.` -> The previous action was invalid (e.g., trying to open an already open receptacle). Re-assess the state.

## Object & Receptacle Notation
*   Objects and receptacles have unique identifiers (`potato 1`, `fridge 1`, `cabinet 3`).
*   Always use the full identifier as provided in observations.
