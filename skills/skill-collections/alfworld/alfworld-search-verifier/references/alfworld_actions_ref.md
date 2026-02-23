# Alfworld Action Reference

This document lists the core actions available to an agent in the Alfworld environment, as per the trajectory. Use this for quick reference when planning actions.

## Action List & Format
The agent must output actions strictly in the format: `Action: <action command>`.

| # | Action Template          | Description                                                                 | Example                     |
|---|--------------------------|-----------------------------------------------------------------------------|-----------------------------|
| 1 | `go to {recep}`          | Navigate to a specified receptacle.                                         | `go to sidetable 1`         |
| 2 | `take {obj} from {recep}`| Pick up an object from a receptacle.                                        | `take cellphone 3 from desk 1` |
| 3 | `put {obj} in/on {recep}`| Place a held object into or onto a receptacle.                              | `put cellphone 3 in/on bed 1` |
| 4 | `open {recep}`           | Open a closed receptacle (e.g., drawer, cabinet).                           | `open drawer 1`             |
| 5 | `close {recep}`          | Close an open receptacle.                                                   | `close drawer 1`            |
| 6 | `toggle {obj} {recep}`   | Toggle the state of an object (e.g., switch) associated with a receptacle.  | `toggle lamp sidetable 1`   |
| 7 | `clean {obj} with {recep}`| Clean an object using a receptacle/tool.                                    | `clean plate with sinkbasin 1`|
| 8 | `heat {obj} with {recep}`| Heat an object using a receptacle/appliance.                                | `heat soup with microwave 1`|
| 9 | `cool {obj} with {recep}`| Cool an object using a receptacle/appliance.                                | `cool wine with fridge 1`   |

## Key Notes for Search Verifier Skill
*   **{recep}:** A receptacle is a container or surface (e.g., `shelf 1`, `desk 1`, `drawer 2`, `bed 1`).
*   **{obj}:** An object is an item that can be manipulated (e.g., `cellphone 3`, `credit card 1`).
*   **Verification Relevance:** The `go to`, `open`, and `close` actions are most critical for the search-verifier skill to navigate and inspect receptacles.
*   **Observation:** After each action, the environment provides an `Observation:` which must be parsed to determine the next step.
*   **Invalid Actions:** If an action is invalid, the environment outputs `"Nothing happened"`. The verifier skill should typically not trigger invalid actions if following a known list of receptacles.
