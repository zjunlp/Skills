# Skill Derivation from Trajectory

## Source Trajectory Summary
**Task**: Put a toiletpaper in toiletpaperhanger.
**Successful Navigation Sequence**:
1.  `go to toiletpaperhanger 1` - Initial approach to check state
2.  `go to toilet 1` - Navigate to object source
3.  `go to toiletpaperhanger 1` - Return for final placement

## Extracted Navigation Logic
1.  **Target Identification**: Agent first identifies the target location from the task description.
2.  **State Checking**: Agent navigates to target to inspect current state (empty/full).
3.  **Path Planning**: Based on inspection, agent determines needed object and navigates to its location.
4.  **Return Navigation**: After obtaining object, agent returns to original target.

## Skill Abstraction
The trajectory shows a repeated pattern of `go to` actions for:
- Initial target inspection
- Moving to object sources
- Returning to placement locations

This pattern is abstracted into the `alfworld-location-navigator` skill that encapsulates the fundamental "move to location" capability used in all three trajectory steps.
