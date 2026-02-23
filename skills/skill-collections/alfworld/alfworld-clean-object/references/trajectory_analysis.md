# Trajectory Analysis & Skill Derivation

## Source Trajectory
**Task:** "put a clean potato in microwave"

**Key Steps:**
1. Agent acquires potato from fridge.
2. **Skill Trigger:** Task requires a *clean* potato.
3. Agent navigates to `sinkbasin 1`.
4. Agent executes: `clean potato 1 with sinkbasin 1`
5. Environment confirms: "You clean the potato 1 using the sinkbasin 1."
6. Agent proceeds to place the now-clean potato in the microwave.

## Skill Abstraction
The trajectory demonstrates a clear, reusable subroutine:
1. **Condition:** Object needs to be cleaned.
2. **Action:** `clean {obj} with {recep}`
3. **Location:** Typically a `sinkbasin`.
4. **Prerequisite:** Object must be in agent's inventory.
5. **Outcome:** Object state changes to "clean".

## Environment Notes
- The `clean` action is a discrete, atomic command in ALFWorld.
- The cleaning receptacle must be appropriate (e.g., `sinkbasin`, not `countertop`).
- The action consumes one turn and provides immediate success/failure feedback.
