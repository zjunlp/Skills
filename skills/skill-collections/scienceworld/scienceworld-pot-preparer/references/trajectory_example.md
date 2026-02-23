# Trajectory Example: Pot Preparation
This reference shows the skill in use within the broader context of the "grow an avocado" task.

## Context
The agent is in the `greenhouse` and needs to prepare three flower pots (`flower pot 1`, `flower pot 2`, `flower pot 3`) for planting avocado seeds. Soil has been previously collected and is in the agent's inventory.

## Skill Execution Sequence
1.  **Prepare Flower Pot 1**
    *   **Action**: `move soil in inventory to flower pot 1`
    *   **Observation**: `You move the soil to the flower pot 1.`
2.  **Prepare Flower Pot 3** (Note: Out-of-order preparation is acceptable)
    *   **Action**: `move soil in inventory to flower pot 3`
    *   **Observation**: `You move the soil to the flower pot 3.`
3.  **Prepare Flower Pot 2**
    *   **Action**: `move soil in inventory to flower pot 2`
    *   **Observation**: `You move the soil to the flower pot 2.`

## Key Insight
The skill performs the same core `move` action regardless of the pot's initial state (observed as containing only `water` in the trajectory). The outcome is a pot containing `soil`, ready for the next step (planting a seed).
