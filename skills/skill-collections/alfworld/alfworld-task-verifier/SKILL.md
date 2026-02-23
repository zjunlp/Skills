---
name: alfworld-task-verifier
description: This skill checks the current state against the task goal to determine if the objective has been met or if further actions are needed. It is triggered after completing a key sub-action, such as placing an object, to assess progress. The skill evaluates the observation feedback and the remaining requirements, outputting a decision to either continue searching for missing items or conclude the task.
---
# Instructions

Use this skill to verify task progress in an ALFWorld household environment. The skill is triggered after a key sub-action (e.g., `put {obj} in/on {recep}`) is performed.

## 1. Input Analysis
- **Input:** The most recent `Observation:` from the environment following an action.
- **Task Goal:** The original, full task description (e.g., "find two pen and put them in garbagecan").

## 2. Verification Logic
Analyze the observation to determine if the task's goal conditions are satisfied.
1.  **Parse the Goal:** Identify the target object(s) and the target receptacle from the task description.
2.  **Check the Observation:** Scrutinize the observation text for evidence that the required objects are present in the target receptacle.
    -   Positive evidence: Phrases like `"you see a pen 3"` located `"in/on the garbagecan 1"`.
    -   The presence of other items in the receptacle does not invalidate success.
3.  **Make a Decision:**
    -   **Task Complete:** If the observation confirms all required objects are in the target receptacle. Output: `"Verification: Task complete. No further action needed."`
    -   **Continue Task:** If the observation shows some, but not all, required objects are in the target receptacle, or if the target object was just placed elsewhere. Output: `"Verification: Task incomplete. Continue searching for [missing object(s)]."`
    -   **Action Invalid/No Change:** If the observation is `"Nothing happened"` or does not reflect the intended outcome of the last action. Output: `"Verification: Last action was ineffective. Re-assess and try a different approach."`

## 3. Output
Output only the verification decision in the specified format. Do not output the next action. This skill informs the planning for the *next* action.

**Example Outputs:**
- `Verification: Task complete. No further action needed.`
- `Verification: Task incomplete. Continue searching for pen 2.`
- `Verification: Last action was ineffective. Re-assess and try a different approach.`
