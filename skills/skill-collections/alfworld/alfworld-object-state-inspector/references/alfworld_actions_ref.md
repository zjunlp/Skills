# AlfWorld Action Reference for State Inspection

## Relevant Actions for This Skill
This skill is used in conjunction with the `go to` action. The inspection is performed by the environment's feedback *after* a successful navigation.

**Primary Pre-condition Action:**
*   `go to {recep}` - Moves the agent to a specific receptacle. The subsequent observation reveals the receptacle's state.

## Observation Patterns
The environment responds to `go to {recep}` with one of the following patterns:

1.  **Empty Receptacle:**
    *   `On the {recep}, you see nothing.`
    *   *Interpretation:* The target is empty. The required object is not here.

2.  **Populated Receptacle:**
    *   `On the {recep}, you see a {obj1} {id1}{, and a {objN} {idN}}*.`
    *   *Examples:*
        *   `On the toilet 1, you see a soapbottle 1, and a toiletpaper 1.`
        *   `On the countertop 1, you see a mug 1.`
    *   *Interpretation:* The target contains the listed object(s). The IDs (e.g., `1`) are important for subsequent `take` or `put` actions.

3.  **Invalid Action:**
    *   `Nothing happened.`
    *   *Interpretation:* The previous `go to` action failed. The receptacle name may be incorrect, or the agent cannot path to it.

## Skill Integration in Task Flow
A typical task sequence involving this skill:
