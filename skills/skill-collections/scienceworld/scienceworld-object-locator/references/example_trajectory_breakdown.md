# Example Trajectory Breakdown

This document breaks down the provided execution trajectory to illustrate the skill's logic.

## Task
"Measure the temperature of metal fork, which is located around the bedroom."

## Implicit Sub-Task: Locate the Thermometer
The agent does not know where the thermometer is, triggering the object location skill.

### Search Execution (For Thermometer)
1.  **Initial State:** Agent is in `workshop`.
2.  **Action:** `look around` (in workshop).
    *   **Observation:** No thermometer found.
3.  **Decision:** Teleport to a likely room (`kitchen`).
4.  **Action:** `teleport to kitchen`
5.  **Action:** `look around` (in kitchen).
    *   **Observation:** Thermometer found on the table.
6.  **Result:** Search successful. Object located in `kitchen`.

### Search Execution (For Metal Fork)
1.  **Known Location:** Task states "located around the bedroom."
2.  **Action:** `teleport to bedroom`
3.  **Action:** `look around` (in bedroom).
    *   **Observation:** Metal fork found.
4.  **Result:** Search successful. Object located in `bedroom`.

## Skill Principles Demonstrated
1.  **Systematic Search:** When the thermometer wasn't in the first room (workshop), the agent moved to the next likely candidate (kitchen).
2.  **Room Examination:** Used `look around` in each new room to get the object list.
3.  **Action Efficiency:** Used `teleport` for instant movement between search locations.
4.  **Parsing Simplicity:** Identified the target object by name in the observation text.

## Flowchart Summary
