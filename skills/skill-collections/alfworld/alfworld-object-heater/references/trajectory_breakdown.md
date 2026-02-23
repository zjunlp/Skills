# Trajectory Breakdown & Common Pitfalls

## Example Trajectory Analysis
The provided successful trajectory for heating a potato and placing it on a dining table follows this pattern:

1.  **Locate & Acquire Object:** `go to garbagecan 1` -> `take potato 3 from garbagecan 1`
2.  **Navigate to Appliance:** `go to microwave 1`
3.  **Prepare Appliance:** `open microwave 1` (Observation: "The microwave 1 is open.")
4.  **Execute Heating:** `heat potato 3 with microwave 1` (Observation: "You heat the potato 3 using the microwave 1.")
5.  **Deliver Object (Task Specific):** `go to diningtable 1` -> `put potato 3 in/on diningtable 1`

## Critical Observations & Lessons

1.  **State Verification is Key:** After `open microwave 1`, the observation confirmed "The microwave 1 is open. In it, you see nothing." This confirms the appliance is ready for the `heat` action.
2.  **Error Pattern - Premature Closing:** The trajectory shows an erroneous step: after opening the microwave, the agent issued `close microwave 1` without placing the object, which would have made the next `heat` action fail. The correct sequence is **open -> heat**.
3.  **Inventory Requirement:** The `heat` action (`heat potato 3 with microwave 1`) only succeeded because the potato was in the agent's inventory (`You pick up the potato 3 from the garbagecan 1.`). You cannot heat an object that is still in a receptacle.
4.  **"Nothing Happened" Response:** This is the environment's signal for an invalid action. If you receive this after a `heat` command, check:
    *   Is the appliance open/on?
    *   Is the target object in your inventory?
    *   Is the appliance compatible with the object?

## Action Template
The core heating action follows this strict template:
`heat {object_id} with {appliance_id}`

## Compatible Appliances
*   `microwave {n}`
*   `stoveburner {n}`
*   (Potentially other heating receptacles like `oven` in other scenarios)
