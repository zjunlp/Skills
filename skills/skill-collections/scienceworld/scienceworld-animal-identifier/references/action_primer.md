# Action Primer for Animal Identification

## Essential Actions
The following actions from the ScienceWorld environment are core to this skill's execution.

### `teleport to <LOCATION>`
*   **Purpose:** Instantly move the agent to a named room.
*   **Usage:** Critical for initial positioning. Always use if the target animal is known to be elsewhere.
*   **Example:** `teleport to outside`

### `look around`
*   **Purpose:** Describes the current room and lists all objects, containers, and entities within it.
*   **Usage:** The primary method for discovering what animals are present. Must be executed in the target location.
*   **Output Format:** Lists items like `a baby dragonfly`, `a parrot egg`, `a baby wolf`.

### `focus on <OBJECT>`
*   **Purpose:** Signals the agent's intent to work on a specific task-relevant object.
*   **Usage:** The final step of the skill. The `<OBJECT>` must match the name from the `look around` observation exactly.
*   **Example:** `focus on parrot egg`
*   **Note:** This action does not physically manipulate the object but marks it for subsequent task steps.

## Observation Parsing Guide
When you receive an observation from `look around`, scan for lines containing biological entities. Common patterns include:
*   `a/an <descriptor> <animal>` (e.g., `a baby dragonfly`)
*   `a/an <animal> egg` (e.g., `a parrot egg`)
*   `a substance called <name>` (Typically not an animal)

Ignore non-biological items like furniture (`a chair`), tools (`an axe`), and generic substances (`air`, `water`).
