---
name: scienceworld-environment-isolation
description: Closes doors or openings to create a contained environment for controlled processes. Trigger this when you need to isolate a space (like a greenhouse) to optimize conditions for pollination or other environmental-sensitive tasks. This modifies room connectivity to create optimal conditions for specific processes.
---
# Instructions

## When to Use
Trigger this skill when you need to:
1.  **Optimize Pollination:** Isolate a room (like a greenhouse) to concentrate pollinators (e.g., bees) and prevent them from leaving, thereby increasing the chance of successful cross-pollination between plants.
2.  **Control Environmental Factors:** Create a stable, contained space to manage variables like temperature, humidity, or airflow for a sensitive process.
3.  **Prevent Contamination or Interference:** Seal off a room to protect an ongoing experiment or task from external disturbances.

## Core Action Sequence
1.  **Assess the Environment:** Use `look around` to identify all doors, windows, or other openings in the current room.
2.  **Execute Isolation:** For each identified opening (e.g., "door to the outside", "door to the hallway"), issue a `close` command.
    *   **Example:** `close door to outside`
    *   **Example:** `close door to hallway`
3.  **Verify Closure:** Perform a final `look around` to confirm all intended openings are now closed and the room is successfully isolated.

## Key Principles
*   **Completeness:** Ensure *all* exits are closed for full isolation. An open door can defeat the purpose.
*   **Context-Specific:** The specific doors to close depend entirely on the room's layout. Always verify via `look around`.
*   **Post-Isolation:** After isolation, monitor the contained process (e.g., plant growth) and use `wait` actions as needed to allow time for the optimized conditions to take effect.

## Notes
*   This skill modifies the **connectivity** of a space, not its internal state (like temperature controls).
*   The primary observed benefit in the trajectory was enhancing bee-mediated pollination by preventing bee escape.
*   Reversal (re-opening doors) is not part of this skill's core function but can be done using standard `open` commands if needed later.
