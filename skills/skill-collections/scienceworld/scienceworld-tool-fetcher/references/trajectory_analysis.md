# Skill Derivation: Trajectory Analysis

## Source Trajectory Summary
**Task:** Measure the melting point of chocolate.
**Key Tool-Fetching Segment:**
1.  Agent is in the `living room`. Uses `look around`.
2.  Not finding the tool, uses `teleport to kitchen`.
3.  In the `kitchen`, uses `look around`.
4.  Observation lists "a thermometer" directly in the room.
5.  Agent executes `pick up thermometer` successfully.

## Extracted Skill Logic
The trajectory demonstrates the core pattern for this skill:
1.  **Identify Need:** Task requires a "thermometer".
2.  **Navigate:** Teleport to probable location (`kitchen`).
3.  **Search:** Use `look around` to scan the room.
4.  **Locate:** Identify tool in observation list.
5.  **Acquire:** Execute `pick up [tool]`.

## Why This is a Separate Skill
*   **Reusable Pattern:** The sequence (teleport→look→pick up) is generic for fetching any specified tool.
*   **Fragile Core:** The exact match of the tool name in the observation and the `pick up` syntax is error-prone, suitable for scripting.
*   **Context Isolation:** The skill doesn't need to know about the broader experiment (melting chocolate), only about finding the tool.

## Excluded from Skill
*   Actions after tool acquisition (e.g., `focus on thermometer`, `use thermometer`).
*   Heating the chocolate or interpreting measurements.
*   Interacting with secondary objects (green/blue boxes).

## Design Validation
The skill package isolates the tool-fetching subroutine, allowing it to be invoked whenever a task says "you need a [tool]". The main `SKILL.md` provides high-level guidance, while the script handles the deterministic search logic.
