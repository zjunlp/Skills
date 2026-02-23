# Example Execution: Placing a Clean Soapbar

This document provides a reference trajectory demonstrating the `alfworld-receptacle-finder` skill.

**Task:** `put a clean soapbar in shelf.`

**Agent Trajectory:**
1.  **Locate Object:** The agent first finds a `soapbar 1` on the `toilet 1` and takes it.
2.  **Pre-condition Check:** The agent ensures the soapbar is `clean` by using the `sinkbasin 1`.
3.  **Receptacle Search Initiated:** The skill is triggered. The agent begins surveying shelves.
    *   `go to shelf 1`: Observation shows `tissuebox 1`. Not empty, but could be a fallback.
    *   `go to shelf 2`: Observation shows `nothing`. **Suitable empty receptacle found.**
4.  **Placement:** The agent executes `put soapbar 1 in/on shelf 2`, completing the task.

**Key Skill Behaviors Demonstrated:**
*   **Sequential Inspection:** The agent checked shelves in order (`shelf 1`, then `shelf 2`).
*   **Empty Receptacle Priority:** `shelf 2` was chosen because it was empty, satisfying the primary goal.
*   **Integration with Pre-conditions:** The cleaning step was performed *before* the receptacle search, ensuring the object state ("clean soapbar") was correct.
