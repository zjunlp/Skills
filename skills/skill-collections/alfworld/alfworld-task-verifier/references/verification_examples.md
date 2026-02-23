# Verification Examples

Use these examples to understand how the skill should interpret observations.

## Scenario: Task is "find two pen and put them in garbagecan."

### Example 1: After first successful placement
- **Observation:** "You put the pen 3 in/on the garbagecan 1."
- **Verification:** "Verification: Task incomplete. Continue searching for pen."
- **Reasoning:** One pen is in the garbage can, but the goal requires two.

### Example 2: After second successful placement
- **Observation:** "On the garbagecan 1, you see a cd 1, and a pen 3." (Agent is at garbagecan after placing pen 2)
- **Verification:** "Verification: Task complete. No further action needed."
- **Reasoning:** Both pens (instances 2 and 3) are now in the target receptacle. The CD's presence is irrelevant.

### Example 3: After checking empty sidetable
- **Observation:** "On the sidetable 1, you see a tissuebox 1."
- **Verification:** "Verification: Task incomplete. Continue searching for pen."
- **Reasoning:** No pens are observed at this location. The search must continue.

### Example 4: Invalid action
- **Observation:** "Nothing happened."
- **Verification:** "Verification: Last action was ineffective. Re-assess and try a different approach."
- **Reasoning:** The previous action failed and must be reconsidered.

## Scenario: Task is "put the mug on the sidetable."

### Example 5: Success
- **Observation:** "You put the mug 1 in/on the sidetable 2."
- **Verification:** "Verification: Task complete. No further action needed."
- **Reasoning:** The single required object is on a sidetable (instance 2 satisfies "the sidetable").

### Example 6: Wrong receptacle
- **Observation:** "You put the mug 1 in/on the diningtable 1."
- **Verification:** "Verification: Task incomplete. Continue searching for sidetable."
- **Reasoning:** The mug is placed, but not on the specified receptacle type.

## Key Principles
1.  **Presence over Purity:** A receptacle containing the target objects plus other items still satisfies the goal.
2.  **Type over Instance:** "Pen" refers to the object type. Any pen instance (pen 1, pen 2, etc.) fulfills the requirement for that object type.
3.  **Context is Key:** The observation text must be read in the context of the agent's last action and location. An object listed in an observation is typically at the agent's current location.
